# T5-Large Distributed Training Benchmark

Benchmarking suite comparing distributed training strategies on **T5-Large** using **Megatron-DeepSpeed**. All strategies run the same pre-training objective (T5 span-corruption masked language modelling) on the **XSum corpus**, with identical model architecture and hyperparameters, so throughput numbers are directly comparable.

Project slides are available in the [T5-Parallelism presentation](docs/T5-Parallelism.pdf).

## Strategies

| Strategy | Script | Default port | Parallelism |
|---|---|---|---|
| DDP (baseline) | `run_megatron_t5_ddp.sh` | 29800 | Megatron native DDP, no model sharding |
| ZeRO-1 | `run_megatron_t5_zero1.sh` | 29801 | Optimizer state sharding |
| ZeRO-2 | `run_megatron_t5_zero2.sh` | 29802 | + Gradient sharding |
| ZeRO-3 | `run_megatron_t5_zero3.sh` | 29803 | + Parameter sharding |
| ZeRO-3 + CPU Offload | `run_megatron_t5_zero3_offload.sh` | 29804 | + Optimizer & param offload to CPU |
| Tensor Parallelism | `run_megatron_t5_tensor.sh` | 29811 | Column/row-parallel across 2 GPUs (TP=2) |
| Pipeline Parallelism | `run_megatron_t5_pipeline.sh` | 29810 | Encoder on GPU 0, decoder on GPU 1 (PP=2) |
| Hybrid (TP+PP) | `run_megatron_t5_hybrid.sh` | 29812 | TP=2 × PP=2, requires 4 GPUs total |

**Fixed across all runs:** T5-Large architecture (24+24 layers, hidden=1024, heads=16, FFN=2816), XSum dataset, `global_batch_size=16`, `micro_batch_size=1`, `train_iters=1000`, bf16, AdamW (`lr=1e-4`, cosine decay).

**Hardware target:** 2 nodes, 1 GPU each (RTX 5000 GPUs on a SLURM cluster). Tensor/pipeline parallelism use both nodes as a single model-parallel group.

---

## Repository Layout

```
T5-Parallelism/
├── requirements.txt                    # Python deps (deepspeed, transformers, wandb, …)
├── MEGATRON.md                         # Detailed Apex / DeepSpeed build guide
├── docs/
│   └── T5-Parallelism.pdf              # Project presentation slides
├── patches/
│   ├── megatron_t5_fixes.patch         # Bug fixes applied to Megatron-DeepSpeed
│   └── 0001-fix-make-Megatron-compatible-with-t5-pipeline-parall.patch
├── ds_configs/
│   ├── megatron_zero1.json
│   ├── megatron_zero2.json
│   ├── megatron_zero3.json
│   └── megatron_zero3_offload.json
└── scripts/
    ├── common.sh                        # Shared shell utilities (host resolution, etc.)
    ├── megatron_common.sh               # Megatron-specific utilities + W&B env vars
    ├── export_xsum_corpus.py            # Downloads XSum → JSONL
    ├── prepare_xsum_megatron.sh         # Builds Megatron indexed dataset from JSONL
    ├── run_megatron_t5_ddp.sh
    ├── run_megatron_t5_zero1.sh
    ├── run_megatron_t5_zero2.sh
    ├── run_megatron_t5_zero3.sh
    ├── run_megatron_t5_zero3_offload.sh
    ├── run_megatron_t5_tensor.sh
    ├── run_megatron_t5_pipeline.sh
    └── run_megatron_t5_hybrid.sh
```

---

## Prerequisites

- Linux with CUDA 12.4
- Two nodes each with at least 1 NVIDIA GPU (RTX 5000 / A100 / etc.)
- Conda
- Nodes must be able to reach each other over TCP (NCCL communication)
- A shared filesystem between nodes **or** willingness to copy data manually

---

## Step 1 — Get Your Nodes

```bash
salloc --nodes=2 --gres=gpu:1 --time=04:00:00 --partition=<your-partition>
```

SLURM prints the two hostnames. Export them — you will use them for every launch:

```bash
export NODE0=ws-l4-007   # master node (replace with your actual hostnames)
export NODE1=ws-l4-008
export HOSTS="${NODE0} ${NODE1}"
```

> All commands below are run **on both nodes** unless stated otherwise. SSH into each in a separate terminal.

---

## Step 2 — Clone This Repository

On **both nodes**:

```bash
git clone <this-repo-url> T5-Parallelism
cd T5-Parallelism
```

---

## Step 3 — Create and Activate the Conda Environment

On **both nodes**:

```bash
conda create -n megatron python=3.10 -y
conda activate megatron
```

---

## Step 4 — Install PyTorch

On **both nodes**:

```bash
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# Expected: True  12.4
```

---

## Step 5 — Install the CUDA Toolchain

On **both nodes**:

```bash
conda install cuda-toolkit cudnn -c nvidia/label/cuda-12.4.0 -y
conda install -c nvidia cuda-nvcc -y
conda install pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install -c conda-forge pybind11 -y
```

---

## Step 6 — Build and Install Apex

On **both nodes**:

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
cd ..
```

> **If Apex fails with a CUDA version mismatch:** set `export CUDA_HOME=/usr/local/cuda-12.4` and retry. If it still fails, open `apex/setup.py`, find `check_cuda_torch_binary_vs_bare_metal`, and comment out the `raise RuntimeError(...)` line — then retry. See [MEGATRON.md](MEGATRON.md) for the exact lines.

---

## Step 7 — Clone Megatron-DeepSpeed

Clone it **next to** (not inside) this repository, on **both nodes**:

```bash
cd ..   # go up one level from T5-Parallelism
git clone https://github.com/deepspeedai/Megatron-DeepSpeed.git
cd Megatron-DeepSpeed
```

Your directory tree should look like:

```
parent-dir/
├── T5-Parallelism/    ← this repo
└── Megatron-DeepSpeed/
```

---

## Step 8 — Apply the Bug-Fix Patch

On **both nodes**, from inside `Megatron-DeepSpeed/`:

```bash
git apply ../T5-Parallelism/patches/megatron_t5_fixes.patch
```

This patch fixes three bugs in the upstream Megatron-DeepSpeed that prevent T5 from training:

| File | Fix |
|---|---|
| `megatron/tokenizer/tokenizer.py` | `_HFTokenizer.vocab_size` now returns `len(tokenizer)` instead of `tokenizer.vocab_size`, so the 100 T5 sentinel tokens (`<extra_id_0>`–`<extra_id_99>`) are included in the embedding table |
| `megatron/model/t5_model.py` | Fixes an operator-precedence bug in the `return_moe_loss` conditional and adds the missing `encoder-only` forward branch |
| `megatron/model/language_model.py` | Initialises `encoder_moe_losses = []` in the hidden-state branch to prevent `UnboundLocalError` |

Then install Megatron as an editable package and add it to `PYTHONPATH`:

```bash
pip install -e .
export PYTHONPATH=$(pwd)   # add this to ~/.bashrc to persist across sessions
cd ../T5-Parallelism
```

> Re-export `PYTHONPATH` in every new terminal on every node, or add `export PYTHONPATH=/path/to/Megatron-DeepSpeed` to `~/.bashrc`.

---

## Step 9 — Install Python Dependencies

On **both nodes**, from inside `T5-Parallelism/`:

```bash
pip install deepspeed==0.17.6
pip install -r requirements.txt
```

Install FlashAttention last (it compiles CUDA extensions):

```bash
pip install flash-attn --no-build-isolation
```

> FlashAttention is optional. If it fails to build, the scripts fall back to standard attention automatically.

Verify DeepSpeed can see your GPU:

```bash
ds_report
```

Look for `[CUDA]` and `[NCCL]` showing as available.

---

## Step 10 — Set Your W&B API Key

Training metrics are logged to [Weights & Biases](https://wandb.ai) under the project `ml710-t5-parallelism`.

On **both nodes**:

```bash
export WANDB_API_KEY=<your-key>   # get it from https://wandb.ai/authorize
```

To make it permanent:

```bash
echo 'export WANDB_API_KEY=<your-key>' >> ~/.bashrc
```

To use a different project name:

```bash
export WANDB_PROJECT=my-project-name
```

Each script sets a distinct run name by default (`megatron-ddp`, `megatron-zero1`, etc.) so all runs appear in the same project and are easy to compare in the W&B dashboard.

---

## Step 11 — Prepare the Dataset

Run this **once**, on **one node only**. The output files are read by all training scripts.

```bash
bash scripts/prepare_xsum_megatron.sh
```

This script:
1. Downloads XSum from HuggingFace via `export_xsum_corpus.py` → `megatron_data/xsum_text.jsonl`
2. Tokenises with `google-t5/t5-large` and builds Megatron indexed binary files

Confirm the files exist and are non-empty:

```bash
ls -lh megatron_data/xsum_text_document.bin megatron_data/xsum_text_document.idx
```

> **Shared filesystem (NFS/Lustre):** if `megatron_data/` is on a shared mount, both nodes see it automatically.
>
> **No shared filesystem:** copy the two files to the same path on the other node:
> ```bash
> scp megatron_data/xsum_text_document.{bin,idx} ${NODE1}:$(pwd)/megatron_data/
> ```

---

## Running Experiments

Every script takes the same two positional arguments:

```bash
bash scripts/<script>.sh <master-node-hostname> <node-rank>
```

Open **two terminals** (one SSH session per node) and run both commands at roughly the same time — the processes rendezvous via torchrun before training starts.

You must export the following in every terminal before launching:

```bash
export HOSTS="${NODE0} ${NODE1}"
export PYTHONPATH=/path/to/Megatron-DeepSpeed
export WANDB_API_KEY=<your-key>
conda activate megatron
```

---

### DDP (Baseline)

Megatron native data-parallel training. One full model replica per GPU, gradients all-reduced every step.

**Node 0:**
```bash
bash scripts/run_megatron_t5_ddp.sh ${NODE0} 0
```
**Node 1:**
```bash
bash scripts/run_megatron_t5_ddp.sh ${NODE0} 1
```

---

### ZeRO-1

Optimizer states sharded across GPUs. Each GPU holds a full copy of parameters and gradients but only 1/N of the Adam moments.

**Node 0:**
```bash
bash scripts/run_megatron_t5_zero1.sh ${NODE0} 0
```
**Node 1:**
```bash
bash scripts/run_megatron_t5_zero1.sh ${NODE0} 1
```

---

### ZeRO-2

Optimizer states + gradients sharded. Reduces gradient memory by 1/N versus ZeRO-1.

**Node 0:**
```bash
bash scripts/run_megatron_t5_zero2.sh ${NODE0} 0
```
**Node 1:**
```bash
bash scripts/run_megatron_t5_zero2.sh ${NODE0} 1
```

---

### ZeRO-3

Optimizer states + gradients + parameters all sharded. Maximum memory savings; requires an all-gather before each forward pass to reconstruct parameters.

**Node 0:**
```bash
bash scripts/run_megatron_t5_zero3.sh ${NODE0} 0
```
**Node 1:**
```bash
bash scripts/run_megatron_t5_zero3.sh ${NODE0} 1
```

---

### ZeRO-3 + CPU Offload

Same as ZeRO-3 but optimizer states and parameters are additionally offloaded to CPU RAM. Enables fitting very large models on small GPUs at the cost of PCIe bandwidth.

**Node 0:**
```bash
bash scripts/run_megatron_t5_zero3_offload.sh ${NODE0} 0
```
**Node 1:**
```bash
bash scripts/run_megatron_t5_zero3_offload.sh ${NODE0} 1
```

---

### Tensor Parallelism (TP=2)

Each GPU holds **half** the model — attention heads and FFN columns/rows are split across both GPUs. Both nodes jointly execute every layer in parallel with an all-reduce per layer.

**Node 0:**
```bash
bash scripts/run_megatron_t5_tensor.sh ${NODE0} 0
```
**Node 1:**
```bash
bash scripts/run_megatron_t5_tensor.sh ${NODE0} 1
```

---

### Pipeline Parallelism (PP=2)

GPU 0 runs the full encoder stack; GPU 1 runs the full decoder stack. Activations are streamed between nodes between micro-batches.

**Node 0:**
```bash
bash scripts/run_megatron_t5_pipeline.sh ${NODE0} 0
```
**Node 1:**
```bash
bash scripts/run_megatron_t5_pipeline.sh ${NODE0} 1
```

---

### Hybrid Parallelism (TP=2 × PP=2) — requires 4 GPUs

Combines tensor and pipeline parallelism. Requires **4 GPUs total** (2 per node) or **4 nodes** (1 GPU each). Not runnable in the default 2-node/1-GPU-each setup without adjusting `NPROC_PER_NODE` and `NNODES`.

```bash
# 4 nodes, 1 GPU each:
NNODES=4 bash scripts/run_megatron_t5_hybrid.sh ${NODE0} 0   # node 0
NNODES=4 bash scripts/run_megatron_t5_hybrid.sh ${NODE0} 1   # node 1
NNODES=4 bash scripts/run_megatron_t5_hybrid.sh ${NODE0} 2   # node 2
NNODES=4 bash scripts/run_megatron_t5_hybrid.sh ${NODE0} 3   # node 3
```

---

## Environment Variable Reference

All settings can be overridden by exporting before the script call. Nothing inside the scripts needs to be edited.

| Variable | Default | Description |
|---|---|---|
| `HOSTS` | `ws-l4-002 ws-l4-010` | Space-separated node hostnames in rank order |
| `MEGATRON_DIR` | `../Megatron-DeepSpeed` | Path to your Megatron-DeepSpeed checkout |
| `MEGATRON_DATA_DIR` | `./megatron_data` | Directory containing indexed dataset files |
| `WANDB_PROJECT` | `ml710-t5-parallelism` | W&B project name |
| `WANDB_API_KEY` | _(unset)_ | Your W&B API key |
| `WANDB_RUN_NAME` | strategy-specific | W&B run name (e.g. `megatron-ddp`) |
| `TRAIN_ITERS` | `1000` | Total training iterations |
| `GLOBAL_BATCH_SIZE` | `16` | Global batch size |
| `MICRO_BATCH_SIZE` | `1` | Per-GPU micro-batch size |
| `ENCODER_NUM_LAYERS` | `24` | Encoder depth |
| `DECODER_NUM_LAYERS` | `24` | Decoder depth |
| `HIDDEN_SIZE` | `1024` | Hidden dimension |
| `NUM_ATTENTION_HEADS` | `16` | Attention heads |
| `FFN_HIDDEN_SIZE` | `2816` | FFN intermediate size |
| `ENCODER_SEQ_LENGTH` | `512` | Encoder input length |
| `DECODER_SEQ_LENGTH` | `128` | Decoder output length |
| `LR` | `0.0001` | Peak learning rate |
| `PRECISION_FLAG` | `--bf16` | `--bf16`, `--fp16`, or omit for fp32 |
| `MASTER_PORT` | script-specific | torchrun rendezvous port |
| `OUTPUT_DIR` | `./outputs/<strategy>` | Checkpoint and log directory |

Quick example — smoke-test with 50 iterations and a custom run name:

```bash
TRAIN_ITERS=50 WANDB_RUN_NAME=smoke-ddp bash scripts/run_megatron_t5_ddp.sh ${NODE0} 0
```

---

## What Gets Logged

### Terminal output

Megatron prints an iteration log every `LOG_INTERVAL=20` steps:

```
iteration       20/    1000 | consumed samples:         320 | elapsed time per iteration (ms): 1234.5 |
  learning rate: 9.800E-05 | global batch size:    16 | lm loss: 3.456789E+00 |
  number of skipped iterations:   0 | number of nan iterations:   0 |
  samples per second: 12.345 | TFLOPs: 56.78
```

### Weights & Biases

All runs are logged to your W&B project. Key metrics logged automatically:

| Metric | W&B key | Notes |
|---|---|---|
| Training loss | `lm loss` | Span-corruption cross-entropy |
| Throughput | `samples per second` | Primary benchmark metric |
| Iteration time | `elapsed time per iteration` | In milliseconds |
| Learning rate | `learning rate` | |
| Loss scale | `loss scale` | bf16 dynamic scaling |

To compare strategies: open the project in W&B, select all runs, and plot `samples per second` vs `iteration` on a single chart.

### GPU memory (manual)

Run this in a separate terminal on each node during training:

```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 5
```

---

## Troubleshooting

**`Missing or empty Megatron indexed dataset files`**
Run `bash scripts/prepare_xsum_megatron.sh` first and confirm `.bin`/`.idx` exist under `megatron_data/`.

**`srcIndex < srcSelectDimSize` CUDA assertion (out-of-bounds embedding lookup)**
The patch was not applied. Follow Step 8 — the `vocab_size` fix is required for T5 sentinel tokens to fit in the embedding table.

**`assert isinstance(model[0], deepspeed.PipelineEngine)` AssertionError**
The `--no-pipeline-parallel` flag is missing from a ZeRO script. This is fixed in the scripts in this repo; make sure you have the latest version.

**ZeRO-2 / ZeRO-3 hangs after "training …" with no iteration output**
`overlap_comm` was enabled. This is already set to `false` in the configs here. If you edited the configs, set `"overlap_comm": false` in the relevant `ds_configs/*.json`.

**`Unable to infer NODE_RANK`**
Pass the rank explicitly: `bash scripts/run_megatron_t5_ddp.sh ${NODE0} 0` on node 0, and `... 1` on node 1.

**`NPROC_PER_NODE > LOCAL_GPU_COUNT`**
Each node only has 1 GPU. Do not set `NPROC` above 1 for any script except `run_megatron_t5_hybrid.sh`.

**Port already in use**
Each script uses a unique default port (29800–29812). Override if needed:
```bash
MASTER_PORT=29900 bash scripts/run_megatron_t5_ddp.sh ${NODE0} 0
```

**Apex build fails**
Set `export CUDA_HOME=/usr/local/cuda-12.4` and retry. See [MEGATRON.md](MEGATRON.md) for the full workaround including how to bypass the version guard.

**`WARNING: WANDB writing requested but no legit wandb project`**
`WANDB_API_KEY` is not exported. Set it with `export WANDB_API_KEY=<your-key>` before launching.

**DeepSpeed version incompatibility**
Use exactly `deepspeed==0.17.6`. Newer versions have API changes that break the Megatron-DeepSpeed integration used here.
