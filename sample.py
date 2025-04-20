import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from training_store import TrainingDataStore
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# -----------------------------------------------------------------------------
# DDP setup
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(local_rank)
    master = (rank == 0)
else:
    rank = local_rank = 0
    world_size = 1
    device = 'cuda'
    master = True

# -----------------------------------------------------------------------------
init_from = 'resume'  # 'resume' or 'gpt2*'
# out_dir = 'out_training_attribution'
out_dir = 'out-shakespeare-finetune'
out_dir = 'out-owt-gpt2'

start = "Tell me about communism in Vietnam!\n"
num_samples = 3
max_new_tokens = 500
temperature = 0.9
top_k = 200
seed = 1337
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
training_data_attribution = False

# -----------------------------------------------------------------------------
torch.manual_seed(seed + rank)
torch.cuda.manual_seed(seed + rank)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32,
           'bfloat16': torch.bfloat16,
           'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
import hashlib
import json

def dprint(*args, **kwargs):
    if master:
        print(*args, **kwargs)

def get_checkpoint_hash(checkpoint):
    metadata = {'val_loss': float(checkpoint['best_val_loss'].cpu().item()),
                'model_args': checkpoint['model_args'],
                'iter_num': checkpoint['iter_num']}
    metadata_str = json.dumps(metadata, sort_keys=True)
    return hashlib.md5(metadata_str.encode()).hexdigest()

def get_cache_path(checkpoint):
    cache_dir = 'embedding_cache'
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"embeddings_{get_checkpoint_hash(checkpoint)}.pkl")

# -----------------------------------------------------------------------------
def load_or_compute_embeddings(model_ref, training_data, checkpoint):
    cache_path = get_cache_path(checkpoint)
    if os.path.exists(cache_path):
        dprint(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    dprint("Computing fresh embeddings...")
    training_store = compute_training_store(model_ref, training_data)
    dprint(f"Caching embeddings to {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(training_store, f)
    return training_store

# -----------------------------------------------------------------------------
num_neighbors = 3

def print_relevant_examples(examples):
    if not master:
        return
    print("\nMost relevant training examples:")
    for idx, example in enumerate(examples, 1):
        clean_text = decode(example)
        print(f"\n{idx}. Example text:")
        print(clean_text)
        print("-" * 40)

# -----------------------------------------------------------------------------
def load_training_data():
    if 'config' not in checkpoint or 'dataset' not in checkpoint['config']:
        raise ValueError("Checkpoint missing dataset config")
    dataset_path = os.path.join('data', checkpoint['config']['dataset'], 'train.bin')
    return np.memmap(dataset_path, dtype=np.uint16, mode='r')

# -----------------------------------------------------------------------------
def compute_training_store(model_ref, data, chunk_size=256):
    training_store = TrainingDataStore()
    total_chunks = (len(data) - 1) // chunk_size + 1
    if master:
        pbar = tqdm(total=total_chunks, desc="Computing embeddings")
    for idx in range(total_chunks):
        if ddp and idx % world_size != rank:
            continue
        start_i = idx * chunk_size
        end_i = min(start_i + chunk_size, len(data))
        chunk = data[start_i:end_i]
        input_ids = torch.tensor(chunk, dtype=torch.long, device=device)[None, ...]
        with torch.no_grad():
            with ctx:
                _, _, emb = model_ref(input_ids)
        training_store.add_example(chunk.tolist(), emb[-1].cpu())
        if master:
            pbar.update(1)
    if master:
        pbar.close()
    if ddp:
        torch.distributed.barrier()
    return training_store

# -----------------------------------------------------------------------------
# MODEL LOADING
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'original.pt')
    dprint(f"Loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    for k, v in list(state_dict.items()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    if training_data_attribution:
        data = load_training_data()
        model.to(device)
else:
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    if training_data_attribution:
        data = None

model.eval()
model.to(device)

# wrap in DDP if requested
if ddp:
    model = DDP(model, device_ids=[local_rank])
# unwrap for generation and embedding functions
gen_model = model.module if ddp else model

if compile:
    gen_model = torch.compile(gen_model)

# ENCODER SETUP
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# Compute or load embeddings
if training_data_attribution and init_from == 'resume':
    training_store = load_or_compute_embeddings(gen_model, data, checkpoint)

# GENERATION (master only)
if master:
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    dprint(f"\nPrompt: {start}")
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                dprint(f"\n******************************************************")
                dprint(f"\nSample {k+1}:")
                y = gen_model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                dprint("\nGenerated text:")
                dprint(decode(y[0].tolist()))

                if training_data_attribution:
                    combined = torch.cat([x, y], dim=1)
                    emb = combined[:, -gen_model.config.block_size:]
                    combined_emb = emb.mean(dim=1)
                    examples = training_store.find_nearest_neighbors(combined_emb, k=num_neighbors)
                    print_relevant_examples(examples)
                    dprint("=" * 60)

# CLEANUP
if ddp:
    destroy_process_group()
