# T5-Large Distributed Training Benchmark

A benchmarking suite for comparing **distributed training strategies** on `t5-large` across multiple GPUs and nodes. The goal is to measure and compare **training throughput**, **GPU memory usage**, and **convergence** for five parallelism strategies:

| Strategy | Script | Backend |
|---|---|---|
| DDP Baseline | `run_trainer.sh` | PyTorch `DistributedDataParallel` |
| FSDP | `run_fsdp.sh` | PyTorch `FullyShardedDataParallel` |
| ZeRO-2 | `run_zero2.sh` | DeepSpeed ZeRO Stage 2 |
| ZeRO-3 | `run_zero3.sh` | DeepSpeed ZeRO Stage 3 |
| ZeRO-3 + CPU Offload | `run_zero3_offload.sh` | DeepSpeed ZeRO Stage 3 + CPU offload |

All strategies share a single training entrypoint (`train.py`) and are configured purely through CLI flags and config files — no code changes needed to switch strategies.

In addition to these strategies, **Megatron-DeepSpeed** is also utilized to implement three parallelism strategies:

| Strategy | Script | Backend |
|---|---|---|
| Tensor / Model Parallelism | `run_megatron_t5_tensor.sh` | Megatron tensor model parallelism |
| Pipeline Parallelism | `run_megatron_t5_pipeline.sh` | Megatron pipeline model parallelism |
| Hybrid Parallelism | `run_megatron_t5_hybrid.sh` | Megatron tensor + pipeline parallelism |

---

## Task

Training is performed on [GLUE](https://gluebenchmark.com/) ([HuggingFace dataset](https://huggingface.co/datasets/nyu-mll/glue)) classification benchmarks, reformulated as **text-to-text** tasks in the T5 style. Integer labels are converted to text strings and the model is trained to generate the correct label token.

| GLUE Task | Input Prompt Format | Labels |
|---|---|---|
| SST-2 (sentiment) | `sst2 sentence: <text> sentiment:` | `positive` / `negative` |
| MRPC (paraphrase) | `mrpc sentence1: <s1> sentence2: <s2> equivalent:` | `equivalent` / `not_equivalent` |
| QNLI (inference) | `qnli question: <q> sentence: <s> answer:` | `entailment` / `not_entailment` |

The default and recommended task is **SST-2** (`--task-name sst2`), which is fast to run and gives a clean benchmark signal.

---

## Parallelism Strategies

### DDP — Distributed Data Parallel (Baseline)
Each GPU holds a **full copy** of the model. Gradients are averaged across GPUs via all-reduce after each backward pass. Simple and well-understood, but the most memory-hungry strategy since every GPU pays the full cost of optimizer states, gradients, and parameters.

### FSDP — Fully Sharded Data Parallel
Each GPU holds only a **shard of the model parameters, gradients, and optimizer states**. Parameters are gathered just-in-time for each forward/backward pass and discarded afterward. Configured via `fsdp_config.json` to wrap at the `T5Block` layer granularity. More memory-efficient than DDP at the cost of higher inter-GPU communication.

### DeepSpeed ZeRO-2
Shards **optimizer states and gradients** across GPUs, but each GPU still holds a full copy of the model parameters. A middle ground: significant memory savings over DDP with lower communication overhead than ZeRO-3. Configured via `ds_configs/zero2.json`.

### DeepSpeed ZeRO-3
Extends ZeRO-2 by also sharding **model parameters** across GPUs. The most memory-efficient GPU-only strategy. Requires more inter-GPU communication (all-gather for parameters) but enables training models that would not fit on a single GPU at all. Configured via `ds_configs/zero3.json`.

### DeepSpeed ZeRO-3 + CPU Offload
Builds on ZeRO-3 by offloading both **parameters and optimizer states to CPU RAM**. Dramatically reduces GPU memory at the cost of CPU↔GPU data transfers, which can significantly reduce throughput. Best used when GPU memory is the hard constraint. Configured via `ds_configs/zero3_offload.json`.

### Megatron Tensor / Model Parallelism
Splits individual layer weights across GPUs using **tensor model parallelism**, so each GPU stores only a partition of the large matrix operations inside the model instead of a full replica. In this repository the default Megatron layout uses `tensor-model-parallel-size=2` and `pipeline-model-parallel-size=1`. This reduces per-GPU model memory and enables larger model shards, but introduces communication inside each transformer layer.

### Megatron Pipeline Parallelism
Splits the model itself into **pipeline stages** placed on different GPUs, with different micro-batches flowing through the stages in sequence. For T5, the default configuration uses `pipeline-model-parallel-size=2` with `pipeline-model-parallel-split-rank=1`, which creates a natural encoder/decoder split. This reduces the amount of model state each GPU must host, but can introduce pipeline bubbles and stage-balance challenges.

### Megatron Hybrid Parallelism
Combines **tensor parallelism and pipeline parallelism** in the same run. In this repository the default layout uses `tensor-model-parallel-size=2` together with `pipeline-model-parallel-size=2`, so tensor parallelism is applied within each pipeline stage. When total processes exceed `tensor x pipeline` model-parallel size, Megatron also forms additional data-parallel groups automatically. This is the most flexible Megatron setup for scaling up, but it is also the most communication-heavy and operationally complex.

---

## Project Layout

```
T5-Parallelism/
├── train.py                  # Shared training entrypoint (all strategies)
├── data.py                   # GLUE dataset loading and text-to-text preprocessing
├── fsdp_config.json          # FSDP T5Block wrapping configuration
├── requirements.txt          # Python dependencies
├── ds_configs/
│   ├── zero2.json            # DeepSpeed ZeRO Stage 2 config
│   ├── zero3.json            # DeepSpeed ZeRO Stage 3 config
│   └── zero3_offload.json    # DeepSpeed ZeRO Stage 3 + CPU offload config
└── scripts/
    ├── common.sh             # Shared shell utilities (host resolution, env setup)
    ├── megatron_common.sh    # Shared Megatron shell utilities
    ├── prepare_xsum_megatron.sh # Export + preprocess XSum for Megatron
    ├── run_trainer.sh        # Launch: DDP baseline
    ├── run_fsdp.sh           # Launch: FSDP
    ├── run_zero2.sh          # Launch: DeepSpeed ZeRO-2
    ├── run_zero3.sh          # Launch: DeepSpeed ZeRO-3
    ├── run_zero3_offload.sh  # Launch: DeepSpeed ZeRO-3 + CPU offload
    ├── run_megatron_t5_tensor.sh   # Launch: Megatron tensor/model parallelism
    ├── run_megatron_t5_pipeline.sh # Launch: Megatron pipeline parallelism
    └── run_megatron_t5_hybrid.sh   # Launch: Megatron hybrid parallelism
```

---

## Installation

Install a CUDA-compatible PyTorch build first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

If your cluster image requires a specific wheel:

```bash
pip uninstall -y torch torchvision
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r requirements.txt
```

For the Megatron-based strategies, follow the dedicated setup guide in [MEGATRON.md](./MEGATRON.md) first.

After that, make sure you also have a sibling checkout of `Megatron-DeepSpeed`:

```bash
cd ..
git clone https://github.com/deepspeedai/Megatron-DeepSpeed
```

The Megatron integration in this repository depends on a small local patch containing the fixes developed during this project. After creating the patch file in this repo, apply it with:

```bash
cd Megatron-DeepSpeed
git apply <path-to-T5-parallelism>/patches/0001-fix-make-Megatron-compatible-with-t5-pipeline-parall.patch
```

---

## Megatron Additions

The three Megatron launchers were added fpr distributed strategies:

- `run_megatron_t5_tensor.sh`: model parallelism via **tensor parallelism**
- `run_megatron_t5_pipeline.sh`: **pipeline parallelism** with an encoder/decoder split
- `run_megatron_t5_hybrid.sh`: **hybrid parallelism** combining tensor and pipeline parallelism

These scripts use the external `Megatron-DeepSpeed` checkout instead of `train.py`, and they operate on an **XSum** corpus preprocessed into Megatron indexed dataset format.

### Preparing the Megatron Dataset

All three Megatron strategies share the same preprocessing pipeline:

```bash
bash scripts/prepare_xsum_megatron.sh
```

This script:

- exports XSum to JSONL
- tokenizes it using the configured Megatron tokenizer
- builds `mmap` `.bin/.idx` dataset files under `megatron_data/`

By default it prepares the **document-level** dataset used by the Megatron launchers.

#### Step-by-Step Dataset Generation

1. Verify the Megatron preprocessing entrypoint exists:

```bash
test -f ../Megatron-DeepSpeed/tools/preprocess_data.py
```

2. Generate the Megatron-ready XSum dataset:

```bash
bash scripts/prepare_xsum_megatron.sh
```

3. Confirm the indexed dataset files were created:

```bash
ls -lh megatron_data/xsum_text_document.bin megatron_data/xsum_text_document.idx
```

The resulting files are the default dataset inputs for:

- `run_megatron_t5_tensor.sh`
- `run_megatron_t5_pipeline.sh`
- `run_megatron_t5_hybrid.sh`

If you want to limit preprocessing for quick smoke tests, you can subsample before building the dataset:

```bash
MAX_TRAIN_SAMPLES=4000 \
MAX_VALIDATION_SAMPLES=1000 \
MAX_TEST_SAMPLES=500 \
bash scripts/prepare_xsum_megatron.sh
```

If you need to regenerate the Megatron dataset from scratch, remove the old indexed files first:

```bash
rm -f megatron_data/xsum_text_document.bin megatron_data/xsum_text_document.idx
bash scripts/prepare_xsum_megatron.sh
```

### Choosing the Parallel Stages

The default layouts are:

- **Tensor parallelism**: `tensor-model-parallel-size=2`, `pipeline-model-parallel-size=1`
- **Pipeline parallelism**: `tensor-model-parallel-size=1`, `pipeline-model-parallel-size=2`, `pipeline-model-parallel-split-rank=1`
- **Hybrid parallelism**: `tensor-model-parallel-size=2`, `pipeline-model-parallel-size=2`, `pipeline-model-parallel-split-rank=1`

For T5, the natural pipeline split is between the **encoder** and **decoder**:

- stage 0 runs the encoder stack
- stage 1 runs the decoder stack

Hybrid parallelism then applies tensor parallelism **within each of those pipeline stages**.

### Running the Three New Strategies

As with the other scripts, run the same launcher on each node manually. The first positional argument is the master hostname and the second is the node rank.

#### 1. Tensor / Model Parallelism

This is the model-parallel-only strategy.

Default layout:

- `tensor-model-parallel-size=2`
- `pipeline-model-parallel-size=1`

Example:

```bash
export HOSTS="ws-lx-xxy ws-lx-xxz"
bash scripts/run_megatron_t5_tensor.sh ws-lx-xxy 0
bash scripts/run_megatron_t5_tensor.sh ws-lx-xxy 1
```

#### 2. Pipeline Parallelism

This is the pipeline-only strategy.

Default layout:

- `tensor-model-parallel-size=1`
- `pipeline-model-parallel-size=2`
- `pipeline-model-parallel-split-rank=1`

Example:

```bash
export HOSTS="ws-lx-xxy ws-lx-xxz"
bash scripts/run_megatron_t5_pipeline.sh ws-lx-xxy 0
bash scripts/run_megatron_t5_pipeline.sh ws-lx-xxy 1
```

#### 3. Hybrid Parallelism

This combines tensor parallelism and pipeline parallelism.

Default layout:

- `tensor-model-parallel-size=2`
- `pipeline-model-parallel-size=2`
- `pipeline-model-parallel-split-rank=1`

This requires at least **4 total GPU processes**.

Example on 2 nodes with 2 GPU processes per node:

```bash
export HOSTS="ws-lx-xxy ws-lx-xxz"
NPROC_PER_NODE=2 bash scripts/run_megatron_t5_hybrid.sh ws-lx-xxy 0
NPROC_PER_NODE=2 bash scripts/run_megatron_t5_hybrid.sh ws-lx-xxy 1
```

Example on 1 node with 4 GPU processes:

```bash
export HOSTS="ws-lx-xxy"
NPROC_PER_NODE=4 bash scripts/run_megatron_t5_hybrid.sh ws-lx-xxy 0
```

### Megatron Outputs

The Megatron launchers write to separate directories:

- `outputs/megatron_tp`
- `outputs/megatron_pp`
- `outputs/megatron_hybrid`

Each one stores checkpoints under its own `checkpoints/` subdirectory.

---

## Configuration

All scripts are controlled through **environment variables** so you can override any setting without editing the scripts. The full list of supported overrides:

| Variable | Default | Description |
|---|---|---|
| `HOSTS` | `ws-l4-002 ws-l4-010` | Space-separated list of node hostnames, in rank order |
| `TASK_NAME` | `sst2` | GLUE task: `sst2`, `mrpc`, or `qnli` |
| `OUTPUT_DIR` | `./outputs/<strategy>` | Where checkpoints and metrics are saved |
| `MODEL_NAME` | `t5-large` | HuggingFace model name or local path |
| `EPOCHS` | `1` | Number of training epochs |
| `PER_DEVICE_BATCH_SIZE` | `1` | Per-GPU micro-batch size |
| `EVAL_BATCH_SIZE` | `2` | Per-GPU eval batch size |
| `GRAD_ACCUM` | `16` | Gradient accumulation steps |
| `OPTIMIZER` | `adafactor` | Optimizer (`adafactor` or `adamw_torch`) |
| `MAX_TRAIN_SAMPLES` | `4000` | Subsample training set (0 = use full dataset) |
| `MAX_EVAL_SAMPLES` | `1000` | Subsample eval set (0 = use full dataset) |
| `PRECISION_FLAG` | `--bf16` | `--bf16`, `--fp16`, or empty for fp32 |
| `GRADIENT_CHECKPOINTING` | `--gradient-checkpointing` | Pass empty string to disable |
| `MASTER_PORT` | `29700` | `torchrun` rendezvous port |
| `NNODES` | inferred from `HOSTS` | Number of nodes (override if needed) |

### Recommended Defaults for Benchmarking

These settings fit within typical shared-cluster time and memory limits:

```bash
export TASK_NAME=sst2
export EPOCHS=1
export MAX_TRAIN_SAMPLES=4000
export MAX_EVAL_SAMPLES=1000
export PRECISION_FLAG=--bf16
export GRADIENT_CHECKPOINTING=--gradient-checkpointing
export PER_DEVICE_BATCH_SIZE=1
export GRAD_ACCUM=16
```

---

## Running (Manual Multi-Node Launch)

Each launch script must be run **manually on every node** via SSH. The first positional argument is the **master node hostname**; the second is the **node rank** (0-indexed).

### Step 1: Pick your nodes and export `HOSTS`

```bash
export HOSTS="ws-lx-xxy ws-lx-xxz"   # space-separated, rank order
```

### Step 2: SSH into each node and run the same script

**Node 0 (rank 0):**
```bash
ssh ws-lx-xxy
conda activate <env>
export HOSTS="ws-lx-xxy ws-lx-xxz"
bash scripts/run_trainer.sh ws-lx-xxy 0
```

**Node 1 (rank 1):**
```bash
ssh ws-lx-xxz
conda activate <env>
export HOSTS="ws-lx-xxy ws-lx-xxz"
bash scripts/run_trainer.sh ws-lx-xxy 1
```

Replace `run_trainer.sh` with the desired strategy script:

```bash
bash scripts/run_fsdp.sh          ws-lx-xxy <node_rank>   # FSDP
bash scripts/run_zero2.sh         ws-lx-xxy <node_rank>   # ZeRO-2
bash scripts/run_zero3.sh         ws-lx-xxy <node_rank>   # ZeRO-3
bash scripts/run_zero3_offload.sh ws-lx-xxy <node_rank>   # ZeRO-3 + CPU offload
```

---

## W&B Tracking

All launch scripts have W&B tracking **enabled by default**. Runs are automatically logged to the `t5_mlsys` entity under the `deepseed` project, with a run name derived from the strategy and task (e.g., `ddp-sst2`, `fsdp-sst2`, `zero3-offload-mrpc`).

### Setup

```bash
pip install wandb
wandb login   # paste your API key from https://wandb.ai/authorize
```

### Defaults

| Variable | Default | Description |
|---|---|---|
| `WANDB_ENTITY` | `t5_mlsys` | W&B team / entity |
| `WANDB_PROJECT` | `deepseed` | W&B project name |
| `WANDB_RUN_NAME` | `<strategy>-<task>` | Auto-named per script (e.g. `zero3-sst2`) |

### Overriding

```bash
export WANDB_PROJECT=my-project
export WANDB_RUN_NAME=custom-run-name
bash scripts/run_zero3.sh ws-lx-xxy 0
```

### Disabling W&B

```bash
export WANDB_PROJECT=""
bash scripts/run_trainer.sh ws-lx-xxy 0
```

### What Gets Logged

The HuggingFace Trainer reports the following to W&B automatically:

| Metric | Description |
|---|---|
| `train/loss` | Training loss per step |
| `train/learning_rate` | LR schedule |
| `train/samples_per_second` | Throughput |
| `train/steps_per_second` | Step rate |
| `eval/loss` | Eval loss per epoch |
| `eval/accuracy` | Exact-match accuracy |
| `eval/runtime` | Eval wall-clock time |

---

## What Gets Measured

After each run, the trainer saves the following to `OUTPUT_DIR`:
- `eval_results.json` — evaluation accuracy
- `trainer_state.json` — per-step loss and runtime logs (use for throughput / samples-per-second)
- `final_model/` — the fine-tuned model checkpoint

**Key metrics for benchmarking across strategies:**
- **Training throughput** (samples/sec) — from `trainer_state.json` → `train_samples_per_second`
- **Peak GPU memory** — monitor with `nvidia-smi` or `torch.cuda.max_memory_allocated()`
- **Eval accuracy** — to confirm all strategies converge to equivalent results
- **Wall-clock time** — total runtime per epoch

---

## Useful Overrides

```bash
export TASK_NAME=sst2
export OUTPUT_DIR=/shared/t5_runs/ddp
export EPOCHS=1
export PER_DEVICE_BATCH_SIZE=1
export GRAD_ACCUM=16
export OPTIMIZER=adafactor
export MAX_TRAIN_SAMPLES=4000
export MAX_EVAL_SAMPLES=1000
export MASTER_PORT=29700
```

If memory is tight:

```bash
export PER_DEVICE_BATCH_SIZE=1
export GRAD_ACCUM=16
```

---

## Hardware

Tested configuration:
- **2 nodes**, 1 GPU per node
- **GPU**: NVIDIA RTX 5000 (32 GB VRAM)
- **Interconnect**: SSH-based manual launch (no InfiniBand assumed)

---

## Strategy Summary for Benchmarking

| Strategy | Memory Efficiency | Communication Overhead | Notes |
|---|---|---|---|
| DDP | ❌ Lowest | ✅ Lowest (grad all-reduce only) | Baseline |
| FSDP | ✅ High | 🔶 Medium | PyTorch-native; good for mid-size models |
| ZeRO-2 | ✅ Medium | ✅ Low-Medium | Good throughput, meaningful memory savings |
| ZeRO-3 | ✅✅ Very High | 🔶 High | Best GPU memory; more comm |
| ZeRO-3 Offload | ✅✅✅ Highest | ❌ Highest | For extreme memory constraints; slowest |
