# T5 Large Parallel Training

This directory lists a distributed training stack for `t5-large`. The stack is:

- Hugging Face `transformers` Trainer
- `torchrun` for multi-node launch
- DeepSpeed for ZeRO strategies
- FSDP through the Trainer integration

The training task is `GLUE/SST-2` reformulated as text-to-text. The current readme is based on a cluster which requires manual execution on each node. This has been tested with the following parameters:

- 2 nodes
- 1 GPU per node
- RTX 5000 32 GB
- manual node allocation
- `ssh` into each node and run the same script with a different node rank

## Implemented Strategies

- `Trainer / DDP baseline`
- `FSDP full_shard`
- `DeepSpeed ZeRO-2`
- `DeepSpeed ZeRO-3`
- `DeepSpeed ZeRO-3 + CPU offload`

## Layout

- `train.py`: shared training entrypoint
- `data.py`: GLUE text-to-text preprocessing
- `fsdp_config.json`: FSDP wrapping config for `T5Block`
- `ds_configs/*.json`: DeepSpeed ZeRO configs
- `scripts/*.sh`: manual launchers for your `ssh + torchrun` workflow

## Install

Install a CUDA-compatible PyTorch build first, then:

```bash
pip install -r requirements.txt
```

If your cluster image needs a specific wheel:

```bash
pip uninstall -y torch torchvision
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r requirements.txt
```

## Recommended Defaults

These settings are chosen to stay under your shared-cluster runtime and memory limits:

- task: `sst2`
- epochs: `1`
- max train samples: `4000`
- max eval samples: `1000`
- precision: `bf16`
- gradient checkpointing: enabled

Suggested starting points:

- `DDP`: batch `1`, grad accum `16`, optimizer `adafactor`
- `FSDP`: batch `1`, grad accum `16`
- `ZeRO-2`: batch `1`, grad accum `16`
- `ZeRO-3`: batch `1`, grad accum `16`
- `ZeRO-3 Offload`: batch `1`, grad accum `16`

## Manual Multi-Node Launch

Pick the allocated hosts in a fixed order and export the same `HOSTS` value on every node.

Example:

```bash
export HOSTS="ws-lx-xxy ws-lx-xxz"
```

On node 0:

```bash
ssh ws-lx-xxy
conda activate <env>
export HOSTS="ws-lx-xxy ws-lx-xxz"
bash scripts/run_trainer.sh ws-lx-xxy 0
```

On node 1:

```bash
ssh ws-lx-xxz
conda activate <env>
export HOSTS="ws-lx-xxy ws-lx-xxz"
bash scripts/run_trainer.sh ws-lx-xxy 1
```

Use the same pattern for the other strategies:

```bash
bash scripts/run_fsdp.sh ws-lx-xxy <node_rank>
bash scripts/run_zero2.sh ws-lx-xxy <node_rank>
bash scripts/run_zero3.sh ws-lx-xxy <node_rank>
bash scripts/run_zero3_offload.sh ws-lx-xxy <node_rank>
```

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

If you want slightly richer task comparisons later, the trainer also supports:

- `TASK_NAME=mrpc`
- `TASK_NAME=qnli`

## Strategy Mapping for Your Slides

- `Trainer / DDP`: required naive baseline
- `FSDP`: non-trivial sharded parallel strategy
- `ZeRO-2`, `ZeRO-3`, `ZeRO-3 Offload`: three advanced data-parallel strategies
