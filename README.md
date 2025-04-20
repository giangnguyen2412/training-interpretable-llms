
# nanoGPT with explanations

![assets/img.png](assets/img.png)

## quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to train a character-level GPT on the works of Shakespeare. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```sh
python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

**I have a GPU**. Great, we can quickly train a baby GPT with the settings provided in the [config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file:

```sh
python train.py config/train_shakespeare_char.py
```

If you peek inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
python sample.py --out_dir=out-shakespeare-char
```

This generates a few samples, for example:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

lol  `¯\_(ツ)_/¯`. Not bad for a character-level model after 3 minutes of training on a GPU. Better results are quite likely obtainable by instead finetuning a pretrained GPT-2 model on this dataset (see finetuning section later).

**I only have a macbook** (or other cheap computer). No worries, we can still train a GPT but we want to dial things down a notch. I recommend getting the bleeding edge PyTorch nightly ([select it here](https://pytorch.org/get-started/locally/) when installing) as it is currently quite likely to make your code more efficient. But even without it, a simple train run could look as follows:

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here, since we are running on CPU instead of GPU we must set both `--device=cpu` and also turn off PyTorch 2.0 compile with `--compile=False`. Then when we evaluate we get a bit more noisy but faster estimate (`--eval_iters=20`, down from 200), our context size is only 64 characters instead of 256, and the batch size only 12 examples per iteration, not 64. We'll also use a much smaller Transformer (4 layers, 4 heads, 128 embedding size), and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around max_iters with `--lr_decay_iters`). Because our network is so small we also ease down on regularization (`--dropout=0.0`). This still runs in about ~3 minutes, but gets us a loss of only 1.88 and therefore also worse samples, but it's still good fun:

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
Generates samples like this:

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

Not bad for ~3 minutes on a CPU, for a hint of the right character gestalt. If you're willing to wait longer, feel free to tune the hyperparameters, increase the size of the network, the context length (`--block_size`), the length of training, etc.

Finally, on Apple Silicon Macbooks and with a recent PyTorch version make sure to add `--device=mps` (short for "Metal Performance Shaders"); PyTorch then uses the on-chip GPU that can *significantly* accelerate training (2-3X) and allow you to use larger networks. See [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) for more.

## reproducing GPT-2

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

This will run for about 4 days using PyTorch Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```sh
# Run on the first (master) node with example IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# Run on the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepend `NCCL_IB_DISABLE=1` to the above launches. Your multinode training will work, but most likely _crawl_. By default checkpoints are periodically written to the `--out_dir`. We can sample from the model by simply `python sample.py`.

Finally, to train on a single GPU simply run the `python train.py` script. Have a look at all of its args, the script tries to be very readable, hackable and transparent. You'll most likely want to tune a number of those variables depending on your needs.

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

However, we have to note that GPT-2 was trained on (closed, never released) WebText, while OpenWebText is just a best-effort open reproduction of this dataset. This means there is a dataset domain gap. Indeed, taking the GPT-2 (124M) checkpoint and finetuning on OWT directly for a while reaches loss down to ~2.85. This then becomes the more appropriate baseline w.r.t. reproduction.

## finetuning

Finetuning is no different than training, we just make sure to initialize from a pretrained model and train with a smaller learning rate. For an example of how to finetune a GPT on new text go to `data/shakespeare` and run `prepare.py` to download the tiny shakespeare dataset and render it into a `train.bin` and `val.bin`, using the OpenAI BPE tokenizer from GPT-2. Unlike OpenWebText this will run in seconds. Finetuning can take very little time, e.g. on a single GPU just a few minutes. Run an example finetuning like:

```sh
python train.py config/finetune_shakespeare.py
```

Having multitple GPUs?

```sh
$ torchrun --standalone --nproc_per_node=4 train.py config/finetune_shakespeare.py
```

This will load the config parameter overrides in `config/finetune_shakespeare.py` (I didn't tune them much though). Basically, we initialize from a GPT2 checkpoint with `init_from` and train as normal, except shorter and with a small learning rate. If you're running out of memory try decreasing the model size (they are `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) or possibly decreasing the `block_size` (context length). The best checkpoint (lowest validation loss) will be in the `out_dir` directory, e.g. in `out-shakespeare` by default, per the config file. You can then run the code in `sample.py --out_dir=out-shakespeare`:

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

Whoa there, GPT, entering some dark place over there. I didn't really tune the hyperparameters in the config too much, feel free to try!

## sampling / inference

Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. ```python sample.py --start=FILE:prompt.txt```.

## efficiency notes

For simple model benchmarking and profiling, `bench.py` might be useful. It's identical to what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!

## todos

- Investigate and add FSDP instead of DDP
- Eval zero-shot perplexities on standard evals (e.g. LAMBADA? HELM? etc.)
- Finetune the finetuning script, I think the hyperparams are not great
- Schedule for linear batch size increase during training
- Incorporate other embeddings (rotary, alibi)
- Separate out the optim buffers from model params in checkpoints I think
- Additional logging around network health (e.g. gradient clip events, magnitudes)
- Few more investigations around better init etc.

## troubleshooting

Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile=False` flag. This will slow down the code but at least it will run.

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!

## discussions

> Why loss values of training and validation often have values in range of 2-3.

In the context of GPT-2 and language models, the loss value (both training and validation) represents the cross-entropy loss, which indicates how well the model predicts the next token in a sequence.
Here's what these numbers mean:
1. **Cross-entropy Loss Interpretation**:
    - The loss represents the average negative log likelihood of predicting the correct next token
    - Lower values indicate better predictions
    - A loss of 2.0-3.0 means the model is, on average, assigning:
        - `exp(-2.0) ≈ 0.135` to `exp(-3.0) ≈ 0.050` probability to the correct token
     
2. **Why this range (2-3)?**:
    - GPT models typically work with a vocabulary size of around 50k tokens
    - If predictions were completely random:
        - Loss would be `ln(50000) ≈ 10.82`

    - A loss of 2-3 means the model is performing much better than random
    - However, perfect prediction (loss = 0) is impossible because language has inherent uncertainty
  
3. **Context and Benchmarks**:
    - Professional LLMs often achieve losses around 2.0-2.5 on general text
    - Lower losses (like 1.5) might indicate:
        - Very specialized/predictable text
        - Potential overfitting

    - Higher losses (above 3.0) might indicate:
        - Poor model performance
        - Very complex or random text

So when you see losses in the 2-3 range, it indicates the model is learning meaningful patterns in the text, significantly better than random guessing but not unrealistically perfect.

loss = -log(probability_of_correct_token)  # log == ln or natural logarithm

## training insights

### adding training data attribution

- Insight 1 – Decoding Tokenized Texts

> During training, the raw texts are tokenized and encoded into integer sequences. At inference time, I need to use decode() to map these integers back to their original text form.

- Insight 2 - Evolving Embeddings
> Ensure the embeddings are reliable. Currently, embeddings seem to evolve with model training. However, it's important to ensure that the feature extractor (i.e., embedding model) for input queries and training data remains consistent to support effective KNN retrieval.

- Insight 3 - Window size
Warning: Could not compute fresh training store: Cannot forward sequence of length 2048, block size is only 256
Falling back to stored training store

> The block_size (256 in my case) is a fundamental architecture limitation defined during model training/initialization. It determines:
> 1. **Position Embeddings**: The model has position embeddings from 0 to 255 (256 positions). It can't handle tokens beyond this as it doesn't have position encodings for them.
> 2. **Attention Mechanism**: The self-attention layers are built to handle sequences up to this length. The size of Q K V matrices are fixed to 256

- Insight 4 - Is training nearest neighbor meaningful for LLMs?

> Observation: I think this question boils down to: "Is training nearest neighbor meaningful for generative tasks?"
> For example, in classification tasks, using NNs may help users compare and contrast the input vs. the NNs so see if the model is correct or not (sorry but I am so overfit into model [verification task](https://proceedings.neurips.cc/paper_files/paper/2021/file/de043a5e421240eb846da8effe472ff1-Paper.pdf))
> By contrast, in generative tasks, there is no clear boundary between input and output in the training as the training manner is 
> text continuation/completion. Hence, the explanation "the model is generating this Y from X because in the training set there is a pair of (X_train, Y_train) that is similar to (X, Y)" does not trivial translate.

> Proposed solution: Concatenate the input and output text together (during test time) and use this chunk of text to generate the embeddings for kNN retrieval.
> The explanation now is: "The model is generating this Y from X because in the training set there is a text chunk that is similar to (X + Y)".

- Insight 4.1 - The concatenated text is exceeding the context length
> In the beginning, we only use X to generate the embeddings for kNN retrieval.
> Now, we have X+Y, which can often exceed the context length of the model.
> For example, original X has 21 tokens and Y has 521 tokens, so the concatenated text has 542 tokens > 256 (context length).
> I solved this by using sliding windows of 256 tokens and then perform the average pooling on the embeddings to get the representative embedding for the whole text chunk.

- Insight 4.2 - Can we go back to training and simply increase the context length?
> When increasing the context length, only the position embeddings are changed to keep track of where tokens are in the text.
> For Q, K, V, because they have size of NxD where N is the size of token embeddings and tokens are processed one-by-one --> there is no changes in these matrices for Q, K and V.
> the only change is the position embeddings, which are added to Q, K and V.

> So the answer is Yes, we can go back and expand the context length. I tried it with nanoGPT to increase the context length from 256 -> 512 but did not see improvements in 
> validation loss. Yet, the model can now process longer text; yay!

- Insight 5 - Yes, LLMs generate one **character** at a time.

> In the beginning, I thought we were using a word-based model. 
> But after some time, I realized that we are using a character-based model because the vocab size is only 65 (it makes me confused why is it so small). 
> I go back to prepare.py and recognize that we do character-based tokenization.
> For example, the vocab for training on shakespear writing is:
> ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
> Then, the model generates one character at a time, and the training data is also tokenized into characters. 
> This means that the model is learning to predict the next character in a sequence, rather than the next word.

- Insight 6 - Training data attribution vs. influence.
> Training Data Attribution: Pinpointing the specific training examples that directly cause or contain a fact in a model’s output, focusing on factual entailment. For example, if an LLM correctly states, "The capital of France is Paris," attribution would point to the training data that explicitly contains this fact (e.g., a sentence like "The capital of France is Paris").
> For instance, an influential example might teach the model a pattern or association that indirectly affects its output, like learning that certain names are commonly associated with specific roles or entities. --> kNN may be a post-hoc instantiation of this.

- Insight 7: Increasing model size
> with this: python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0 --> model size = 0.8M

> now I try: block size 2048; batchsize 512; 8 layers of transformer, each has 8 heads; n_embed 512 and max_iter 30k
> python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=2048 --batch_size=512 --n_layer=8 --n_head=8 --n_embd=512 --max_iters=30000 --lr_decay_iters=2000 --dropout=0.0 --> 25.2M
> the model quickly overfits.

- Insight 8: OlmoTrace https://allenai.org/blog/olmotrace

> Highlighting the verbatim spans in the training data.

- Insight 9: Is using the token embedding for kNN retrieval meaningful?

> Current, here is how we extract the embedding of a paragraph from the model:
```python
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # TRAINING_NN
        # Get sequence embedding (mean pooling)
        sequence_embedding = x.mean(dim=1)  # (b, n_embd)
        
        # Input embedding shape: torch.Size([1, 1024, 768])
        # Output embedding shape: torch.Size([1, 768])
```

> The embedding is extracted at the last layer of the transformer. For example, with the context length of 1024, the embedding for the whole context is torch.Size([1, 1024, 768]).
> So what I am doing is that I pool over 1024 tokens to get just one vector of torch.Size([1, 768]) that represents both (input prompt + output) and retrieve NNs based on this vector.
> However, as the LLMs are not trained for any retrieval tasks, the embedding here is not meaningful for similarity retrieval (or at least not working well with cosine function).
> Indeed, if using BLMs, and a sense vector having the concept of "toxic". We can leverage the sense vector to retrieve the training data NNs and remove them.
> In this case, the sense vectors may have "cleaner" representations of "toxic" than the token embeddings.

- Insight 10: Can we trace neurons that highly activate during the generation process? During multiple forward pass, we may be able to identify the role of specific neurons.

>  The neuron trace procedure efficiently identifies the most influential neurons during model generation by attaching forward hooks
to each transformer layer that capture per-neuron activations for each new token.
As the model generates text autoregressively, these hooks record activations
at each timestep. After generation completes, the algorithm computes each
neuron's average activation across all tokens, then ranks and selects the
top K most active neurons as the "trace." 

> Lesson: Based on the observation that the neuron trace algorithm yields nearly 
> identical "highly activating neurons" across different concepts (e.g., Vietnam and tulips), 
> we learn an important lesson about transformer-based language models. While we 
> might expect concept-specific neurons for distinct topics, these results suggest 
> that modern language models likely organize information in more distributed and 
> entangled ways than the theory we would predict. 
> Rather than dedicated neurons for specific concepts, these models appear to rely 
> on patterns of activation across shared neural pathways. This explains why the 
> same neurons in layers 9-11 activate strongly regardless of topic - they may 
> represent higher-level linguistic functions essential to coherent text generation, 
> such as maintaining grammatical consistency, tracking context, or managing output 
> formatting, rather than encoding specific semantic content. This insight challenges 
> our intuition about how neural networks represent knowledge and suggests we need 
> more sophisticated techniques to truly untangle concept-specific representations 
> within these models.
