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
concept_name = "tulip"  # Default concept
prompts_file = f"{concept_name}.json"  # JSON file containing concept-related prompts
neuron_trace_all_positions = True  # Whether to trace all positions or just the last position

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
import json


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
# def make_hooks(activations):
#     hooks = []
#     try:
#         layers = gen_model.transformer.h
#     except AttributeError:
#         layers = getattr(gen_model, 'blocks', [])
#     for layer_idx, layer in enumerate(layers):
#         def hook_fn(module, inp, out, idx=layer_idx):
#             # capture last-token activations: [batch, hidden]
#             # the size of out will increase over time.
#             breakpoint()
#             last_act = out[:, -1, :].detach().squeeze(0)
#             for neuron_idx, val in enumerate(last_act.tolist()):
#                 activations.setdefault((idx, neuron_idx), []).append(val)
#
#         hooks.append(layer.register_forward_hook(hook_fn))
#     return hooks

# Modify the make_hooks function to trace all positions if requested
def make_hooks(activations):
    hooks = []
    try:
        layers = gen_model.transformer.h
    except AttributeError:
        layers = getattr(gen_model, 'blocks', [])

    # Print information about the number of neurons being inspected
    total_layers = len(layers)
    # Assuming the hidden size is the same for all layers, we can check the first layer
    try:
        if hasattr(layers[0], 'mlp'):
            hidden_size = layers[0].mlp.c_proj.out_features
        else:
            # Try different attribute names depending on model architecture
            hidden_size = getattr(layers[0], 'out_proj',
                                  getattr(layers[0], 'output_dense',
                                          getattr(layers[0], 'c_proj', None))).out_features
    except (AttributeError, IndexError):
        # If we can't determine it, use a placeholder
        hidden_size = "unknown"

    dprint(f"\nNeuron Trace Configuration:")
    dprint(f"- Inspecting {total_layers} layers with approximately {hidden_size} neurons each")
    dprint(f"- Total neurons to analyze: ~{total_layers * hidden_size if isinstance(hidden_size, int) else 'unknown'}")
    dprint(f"- Will show top {neuron_trace_topk} most active neurons")
    dprint(f"- Tracing {'all token positions' if neuron_trace_all_positions else 'only the last token position'}\n")

    for layer_idx, layer in enumerate(layers):
        def hook_fn(module, inp, out, idx=layer_idx):
            if neuron_trace_all_positions:
                # Capture activations for all positions in the sequence
                # Shape of out is [batch, sequence_length, hidden_size]
                all_acts = out.detach().squeeze(0)  # Remove batch dimension

                # Only look at newly generated tokens (skip prompt tokens)
                # We can identify this by checking if the sequence length is greater than our prompt length
                current_seq_len = all_acts.shape[0]
                prompt_len = len(start_ids) if 'start_ids' in globals() else 0

                if current_seq_len > prompt_len:
                    # Process activations for all newly generated tokens
                    new_tokens_acts = all_acts[prompt_len:]

                    # For each neuron, add its activations for all new tokens
                    for neuron_idx in range(all_acts.shape[1]):  # Loop over hidden dimension
                        neuron_acts = new_tokens_acts[:, neuron_idx].tolist()
                        for val in neuron_acts:
                            activations.setdefault((idx, neuron_idx), []).append(val)
            else:
                # Original behavior: capture only last-token activations
                last_act = out[:, -1, :].detach().squeeze(0)
                for neuron_idx, val in enumerate(last_act.tolist()):
                    activations.setdefault((idx, neuron_idx), []).append(val)

        hooks.append(layer.register_forward_hook(hook_fn))
    return hooks

# -----------------------------------------------------------------------------
# GENERATION logic
# -----------------------------------------------------------------------------
if master:
    # Read prompts from JSON file
    if os.path.exists(prompts_file):
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                prompts = data.get("prompts", [])
                # If concept is in the file, use it; otherwise use filename
                concept = data.get("concept", concept_name)
        except json.JSONDecodeError:
            dprint(f"Error: {prompts_file} is not a valid JSON file")
            sys.exit(1)
    else:
        dprint(f"Error: {prompts_file} not found. Please create this file with prompts for the concept.")
        sys.exit(1)

    dprint(f"Loaded {len(prompts)} prompts for concept '{concept}'")

    # Initialize counter for neuron occurrences
    neuron_occurrences = Counter()

    # Process each prompt
    for prompt_idx, prompt in enumerate(prompts):
        dprint(f"\n{'=' * 80}\nProcessing prompt {prompt_idx + 1}/{len(prompts)}:\n{prompt}\n{'=' * 80}")

        start_ids = encode(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

        # Generate multiple samples per prompt
        for k in range(num_samples):
            dprint(f"\nGenerating sample {k + 1}/{num_samples} for prompt {prompt_idx + 1}")

            activations = {}
            hooks = make_hooks(activations)

            with torch.no_grad():
                y = gen_model.generate(
                    x,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k
                )

            for h in hooks:
                h.remove()

            # Average activations per neuron
            records = []
            for (layer_idx, neuron_idx), vals in activations.items():
                avg_val = sum(vals) / len(vals)
                records.append((layer_idx, neuron_idx, avg_val))

            dprint(f"\nAnalyzed activations for {len(records)} neurons")
            records.sort(key=lambda x: x[2], reverse=True)
            top_records = records[:neuron_trace_topk]

            # Count occurrences of each neuron in top-k
            for layer, neuron, _ in top_records:
                neuron_occurrences[(layer, neuron)] += 1

            # Save individual trace data
            concept_dir = os.path.join(out_dir, concept)
            os.makedirs(concept_dir, exist_ok=True)
            trace_path = os.path.join(concept_dir, f'neuron_trace_prompt_{prompt_idx + 1}_sample_{k + 1}.pkl')
            with open(trace_path, 'wb') as f:
                pickle.dump({'prompt': prompt, 'prompt_idx': prompt_idx + 1, 'sample': k + 1, 'trace': top_records}, f)

            # Output sample generation text
            dprint(f"\nGenerated text (sample {k + 1}):")
            dprint(decode(y[0].tolist()))

    # Identify the most consistent "concept neurons"
    max_possible_occurrences = len(prompts) * num_samples
    dprint(f"\n{'=' * 80}\nMost consistent {concept} neurons\n{'=' * 80}")
    dprint(f"Total prompts: {len(prompts)}, samples per prompt: {num_samples}")
    dprint(f"Maximum possible occurrences: {max_possible_occurrences}")

    top_neurons = neuron_occurrences.most_common(neuron_trace_topk)
    dprint(f"\nTop {neuron_trace_topk} neurons most consistently activated by {concept} content:")
    dprint("Rank  Layer  Neuron  Occurrences  % of Samples")

    for rank, ((layer, neuron), count) in enumerate(top_neurons, 1):
        percentage = (count / max_possible_occurrences) * 100
        dprint(f"{rank:4d}  {layer:5d}  {neuron:6d}  {count:11d}  {percentage:11.2f}%")

    # Save the overall results
    concept_neurons_path = os.path.join(concept_dir, f'{concept}_neurons.pkl')
    with open(concept_neurons_path, 'wb') as f:
        pickle.dump({
            'concept': concept,
            'neuron_occurrences': dict(neuron_occurrences),
            'top_neurons': top_neurons,
            'prompts': prompts,
            'num_samples': num_samples,
            'max_occurrences': max_possible_occurrences
        }, f)

    dprint(f"\n{concept.capitalize()} neuron analysis complete! Results saved to {concept_neurons_path}")

# CLEANUP
if ddp:
    destroy_process_group()