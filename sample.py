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
from collections import Counter
import json
import sys
import argparse

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
# Configuration flags
init_from = 'resume'  # 'resume' or 'gpt2*'
out_dir = 'out-owt-gpt2'

num_samples = 1  # Generate 3 answers per prompt
max_new_tokens = 500
temperature = 0.9
top_k = 200
seed = 1337
# bfloat16 support check
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
training_data_attribution = False

###
### The neuron trace procedure efficiently identifies the most
# influential neurons during model generation by attaching forward hooks
# to each transformer layer that capture per-neuron activations for each new token.
# As the model generates text autoregressively, these hooks record activations
# at each timestep. After generation completes, the algorithm computes each
# neuron's average activation across all tokens, then ranks and selects the
# top K most active neurons as the "trace."
###
# Flags for neuron tracing
neuron_trace = True
neuron_trace_topk = 10  # number of top activations to record
# concept_name = "tulip"  # Default concept
# prompts_file = f"{concept_name}.json"  # JSON file containing concept-related prompts
neuron_trace_all_positions = False  # Whether to trace all positions or just the last position

# parse CLI args
parser = argparse.ArgumentParser()
parser.add_argument('--concept_name', type=str, default='tulip',
                    help='Name of the concept (also used to find <concept>.json)')
args = parser.parse_args()
concept_name = args.concept_name
prompts_file = f"{concept_name}.json"
print(f"The concept we are interested in is {concept_name}")


# -----------------------------------------------------------------------------
# Set random seeds and precision context
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
# Utility functions
import hashlib


def dprint(*args, **kwargs):
    if master:
        print(*args, **kwargs)


def get_checkpoint_hash(checkpoint):
    metadata = {
        'val_loss': float(checkpoint['best_val_loss'].cpu().item()),
        'model_args': checkpoint['model_args'],
        'iter_num': checkpoint['iter_num']
    }
    metadata_str = json.dumps(metadata, sort_keys=True)
    return hashlib.md5(metadata_str.encode()).hexdigest()


def get_cache_path(checkpoint):
    cache_dir = 'embedding_cache'
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"embeddings_{get_checkpoint_hash(checkpoint)}.pkl")


# -----------------------------------------------------------------------------
# Data attribution helpers
# ----------------------------------------------------------------------------
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


def load_training_data():
    if 'config' not in checkpoint or 'dataset' not in checkpoint['config']:
        raise ValueError("Checkpoint missing dataset config")
    dataset_path = os.path.join('data', checkpoint['config']['dataset'], 'train.bin')
    return np.memmap(dataset_path, dtype=np.uint16, mode='r')


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
# -----------------------------------------------------------------------------
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'original.pt')
    dprint(f"Loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    if training_data_attribution:
        data = load_training_data()
else:
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    if training_data_attribution:
        data = None

model.eval()
model.to(device)

if ddp:
    model = DDP(model, device_ids=[local_rank])
gen_model = model.module if ddp else model
if compile:
    gen_model = torch.compile(gen_model)


#### STERRING GENERATIONS VIA TURNING ON/OFF NEURONS
# 1) list of (layer_idx, neuron_idx) you want to disable
disable_neurons = [(11, 197), (11, 373), (11, 680)] # for violent

# 2) register hooks that zero out just those neuron channels
_disable_hooks = []
for layer_idx, neuron_idx in disable_neurons:
    layer = gen_model.transformer.h[layer_idx]
    def _make_hook(n_idx):
        def hook_fn(module, inp, out):
            # out: [batch, seq_len, hidden_size]
            out = out.clone()                     # detach so we don't break other paths
            out[:, :, n_idx] = 0                  # zero that neuron across all positions
            return out
        return hook_fn
    _disable_hooks.append(layer.register_forward_hook(_make_hook(neuron_idx)))


# -----------------------------------------------------------------------------
# ENCODER SETUP
# -----------------------------------------------------------------------------
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

if training_data_attribution and init_from == 'resume':
    training_store = load_or_compute_embeddings(gen_model, data, checkpoint)

# -----------------------------------------------------------------------------
# Hook function for neuron tracing
# -----------------------------------------------------------------------------
def make_hooks(activations):
    hooks = []
    try:
        layers = gen_model.transformer.h
    except AttributeError:
        layers = getattr(gen_model, 'blocks', [])

    total_layers = len(layers)
    try:
        if hasattr(layers[0], 'mlp'):
            hidden_size = layers[0].mlp.c_proj.out_features
        else:
            hidden_size = getattr(layers[0], 'out_proj',
                                  getattr(layers[0], 'output_dense',
                                          getattr(layers[0], 'c_proj', None))).out_features
    except (AttributeError, IndexError):
        hidden_size = "unknown"

    dprint(f"\nNeuron Trace Configuration:")
    dprint(f"- Inspecting {total_layers} layers with approx. {hidden_size} neurons each")
    dprint(f"- Total neurons to analyze: ~{total_layers * hidden_size if isinstance(hidden_size, int) else 'unknown'}")
    dprint(f"- Will show top {neuron_trace_topk} most active neurons")
    dprint(f"- Tracing {'all token positions' if neuron_trace_all_positions else 'only the last token position'}\n")

    for layer_idx, layer in enumerate(layers):
        def hook_fn(module, inp, out, idx=layer_idx):
            all_acts = out.detach().squeeze(0)
            prompt_len = len(start_ids) if 'start_ids' in globals() else 0
            if neuron_trace_all_positions and all_acts.shape[0] > prompt_len:
                new_acts = all_acts[prompt_len:]
            else:
                new_acts = all_acts[-1:, :]
            for t in range(new_acts.shape[0]):
                for neuron_idx, val in enumerate(new_acts[t].tolist()):
                    activations.setdefault((idx, neuron_idx), []).append(val)
        hooks.append(layer.register_forward_hook(hook_fn))
    return hooks

# -----------------------------------------------------------------------------
# Baseline‐Normalization Helpers
# -----------------------------------------------------------------------------
baseline_prompts_file = "baseline.json"

def compute_average_activations(prompts):
    agg = {}
    counts = {}
    for prompt in tqdm(prompts, desc="Baseline/Concept activations"):
        start_ids_loc = encode(prompt)
        x_loc = torch.tensor(start_ids_loc, dtype=torch.long, device=device)[None, ...]
        acts = {}
        hooks = make_hooks(acts)
        with torch.no_grad():
            _ = gen_model.generate(x_loc,
                                   max_new_tokens=max_new_tokens,
                                   temperature=temperature,
                                   top_k=top_k)
        for h in hooks: h.remove()
        # average per neuron for this prompt
        for neuron, vals in acts.items():
            avg_val = float(np.mean(vals))
            agg[neuron] = agg.get(neuron, 0.0) + avg_val
            counts[neuron] = counts.get(neuron, 0) + 1
    # finalize average
    return {n: agg[n] / counts[n] for n in agg}

# -----------------------------------------------------------------------------
# GENERATION logic (with baseline subtraction)
# -----------------------------------------------------------------------------
if master:
    # load concept prompts
    if os.path.exists(prompts_file):
        with open(prompts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            prompts = data.get("prompts", [])
            concept = data.get("concept", concept_name)
    else:
        dprint(f"Error: {prompts_file} not found.")
        sys.exit(1)

    # load baseline prompts
    if os.path.exists(baseline_prompts_file):
        with open(baseline_prompts_file, 'r', encoding='utf-8') as f:
            bdata = json.load(f)
            baseline_prompts = bdata.get("prompts", [])
    else:
        dprint(f"Error: {baseline_prompts_file} not found.")
        sys.exit(1)

    dprint(f"Loaded {len(prompts)} concept prompts and {len(baseline_prompts)} baseline prompts")

    # compute baseline activations once
    dprint("Computing baseline activations...")
    baseline_acts = compute_average_activations(baseline_prompts)
    dprint("Baseline activations computed.")

    # initialize occurrence counter
    neuron_occurrences = Counter()

    # process each concept prompt as before, but rank by (avg - baseline)
    for prompt_idx, prompt in enumerate(prompts):
        dprint(f"\n{'='*40}\nPrompt {prompt_idx+1}/{len(prompts)}: {prompt}\n{'='*40}")
        start_ids = encode(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

        for k in range(num_samples):
            activations = {}
            hooks = make_hooks(activations)
            with torch.no_grad():
                y = gen_model.generate(x,
                                       max_new_tokens=max_new_tokens,
                                       temperature=temperature,
                                       top_k=top_k)
            for h in hooks: h.remove()
            for h in _disable_hooks:
                h.remove()

            # compute per-neuron avg and diff
            records = []
            for (layer, neuron), vals in activations.items():
                avg_val = float(np.mean(vals))
                base = baseline_acts.get((layer, neuron), 0.0)
                diff = avg_val - base
                records.append((layer, neuron, avg_val, diff))
            # sort by diff
            records.sort(key=lambda x: x[3], reverse=True)
            top_records = records[:neuron_trace_topk]

            # count occurrences
            for layer, neuron, _, _ in top_records:
                neuron_occurrences[(layer, neuron)] += 1

            # save individual trace
            concept_dir = os.path.join(out_dir, concept)
            os.makedirs(concept_dir, exist_ok=True)
            trace_path = os.path.join(concept_dir,
                                      f'neuron_trace_prompt_{prompt_idx+1}_sample_{k+1}.pkl')
            with open(trace_path, 'wb') as f:
                pickle.dump({
                    'prompt': prompt,
                    'prompt_idx': prompt_idx+1,
                    'sample': k+1,
                    'trace': top_records
                }, f)

            # show generated text
            dprint(f"\nGenerated text sample {k+1}:")
            dprint(decode(y[0].tolist()))

    # final ranking by occurrence (still concept-specific)
    max_occ = len(prompts) * num_samples
    dprint(f"\n{'='*20}\nMost consistent {concept} neurons\n{'='*20}")
    dprint(f"Max possible occurrences: {max_occ}")
    top_neurons = neuron_occurrences.most_common(neuron_trace_topk)
    dprint("Rank  Layer  Neuron  Occurrences  %")
    for rank, ((layer, neuron), cnt) in enumerate(top_neurons, 1):
        pct = cnt/max_occ*100
        dprint(f"{rank:4d}  {layer:5d}  {neuron:6d}  {cnt:11d}  {pct:6.2f}%")

    # save overall results
    concept_neurons_path = os.path.join(concept_dir, f'{concept}_neurons.pkl')
    with open(concept_neurons_path, 'wb') as f:
        pickle.dump({
            'concept': concept,
            'baseline_acts': baseline_acts,
            'neuron_occurrences': dict(neuron_occurrences),
            'top_neurons': top_neurons,
            'prompts': prompts,
            'baseline_prompts': baseline_prompts
        }, f)
    dprint(f"\nAnalysis complete! Results at {concept_neurons_path}")

# CLEANUP
if ddp:
    destroy_process_group()
