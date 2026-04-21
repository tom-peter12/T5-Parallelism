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
    ├── run_trainer.sh        # Launch: DDP baseline
    ├── run_fsdp.sh           # Launch: FSDP
    ├── run_zero2.sh          # Launch: DeepSpeed ZeRO-2
    ├── run_zero3.sh          # Launch: DeepSpeed ZeRO-3
    └── run_zero3_offload.sh  # Launch: DeepSpeed ZeRO-3 + CPU offload
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
