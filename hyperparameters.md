# T5 Parallelism Hyperparameters

Defaults below are taken from the run scripts and config files as written, assuming no environment overrides.

## Launch and Parallelism

| Hyperparameter | Hybrid | Pipeline | Tensor | ZeRO-2 | ZeRO-3 | ZeRO-3 Offload | FSDP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Script | `scripts/run_megatron_t5_hybrid.sh` | `scripts/run_megatron_t5_pipeline.sh` | `scripts/run_megatron_t5_tensor.sh` | `scripts/run_zero2.sh` | `scripts/run_zero3.sh` | `scripts/run_zero3_offload.sh` | `scripts/run_fsdp.sh` |
| Backend | Megatron | Megatron | Megatron | HF Trainer + DeepSpeed | HF Trainer + DeepSpeed | HF Trainer + DeepSpeed | HF Trainer + FSDP |
| Model | T5-large architecture; tokenizer `google-t5/t5-large` | T5-large architecture; tokenizer `google-t5/t5-large` | T5-large architecture; tokenizer `google-t5/t5-large` | `t5-large` | `t5-large` | `t5-large` | `t5-large` |
| Nodes default | `4` | `2` | `2` | `2` | `2` | `2` | `2` |
| Processes per node | `1` | `1` | `1` | `1` | `1` | `1` | `1` |
| Total processes default | `4` | `2` | `2` | `2` | `2` | `2` | `2` |
| Tensor parallel size | `2` | `1` | `2` | n/a | n/a | n/a | n/a |
| Pipeline parallel size | `2` | `2` | `1` | n/a | n/a | n/a | n/a |
| Pipeline split rank | `1` | `1` | n/a | n/a | n/a | n/a | n/a |
| Data parallel groups default | `1` | effectively `1` | effectively `1` | DDP/ZeRO across `2` procs | DDP/ZeRO across `2` procs | DDP/ZeRO across `2` procs | FSDP across `2` procs |
| Master port | `29812` | `29810` | `29811` | `29702` | `29703` | `29704` | `29701` |
| Precision | `bf16` | `bf16` | `bf16` | `bf16` | `bf16` | `bf16` | `bf16` |

## Training Hyperparameters

| Hyperparameter | Hybrid | Pipeline | Tensor | ZeRO-2 | ZeRO-3 | ZeRO-3 Offload | FSDP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Epochs | `1` | `1` | `1` | `1` | `1` | `1` | `1` |
| Micro/per-device train batch | `1` | `1` | `1` | `8` | `8` | `8` | `8` |
| Global/effective batch | `16` | `16` | `16` | `16` (`8 * 2 * 1`) | `16` (`8 * 2 * 1`) | `16` (`8 * 2 * 1`) | `16` (`8 * 2 * 1`) |
| Eval batch size | Megatron eval iters based | same | same | `8` | `8` | `8` | `8` |
| Gradient accumulation | derived by Megatron to reach global batch `16` | same | same | `1` | `1` | `1` | `1` |
| Learning rate | `1e-4` | `1e-4` | `1e-4` | `1e-4` | `1e-4` | `1e-4` | `1e-4` |
| Min LR | `1e-5` | `1e-5` | `1e-5` | n/a | n/a | n/a | n/a |
| LR decay style | `linear` | `linear` | `linear` | `linear` HF default | `linear` HF default | `linear` HF default | `linear` HF default |
| LR warmup | `0.01` fraction | `0.01` fraction | `0.01` fraction | `0.01` ratio | `0.01` ratio | `0.01` ratio | `0.01` ratio |
| Weight decay | `0.01` | `0.01` | `0.01` | `0.01` from `train.py` | `0.01` from `train.py` | `0.01` from `train.py` | `0.01` from `train.py` |
| Optimizer | Megatron default, not set in script | same | same | `adamw_torch` | `adamw_torch` | `adamw_torch` | `adamw_torch` |
| Gradient clipping | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` |
| Gradient checkpointing | disabled | disabled | disabled | disabled | disabled | disabled | disabled |
| Train iterations | computed from XSum JSONL, split, global batch, epochs | same | same | epoch-based | epoch-based | epoch-based | epoch-based |
| LR decay iters | defaults to `TRAIN_ITERS` | same | same | n/a | n/a | n/a | n/a |
| Save interval/strategy | every 20% of iterations | same | same | every 20% of steps | every 20% of steps | every 20% of steps | every 20% of steps |
| Eval interval/strategy | every 10% of iterations | same | same | every 10% of steps | every 10% of steps | every 10% of steps | every 10% of steps |
| Eval iters | `20` | `20` | `20` | n/a | n/a | n/a | n/a |
| Log interval/steps | every iteration | every iteration | every iteration | every step | every step | every step | every step |
| Save total limit | not set | not set | not set | `2` from `train.py` | `2` from `train.py` | `2` from `train.py` | `2` from `train.py` |
| Seed | not set in script | not set in script | not set in script | `42` from `train.py` | `42` from `train.py` | `42` from `train.py` | `42` from `train.py` |

## Model and Sequence/Data Settings

| Hyperparameter | Megatron Hybrid/Pipeline/Tensor | ZeRO-2 / ZeRO-3 / Offload / FSDP |
|---|---:|---:|
| Encoder layers | `24` | HF `t5-large` architecture |
| Decoder layers | `24` | HF `t5-large` architecture |
| Hidden size | `1024` | HF `t5-large` architecture |
| Attention heads | `16` | HF `t5-large` architecture |
| KV channels | `64` | HF `t5-large` architecture |
| FFN hidden size | `2816` | HF `t5-large` architecture |
| Encoder/source length | `512` | `512` |
| Decoder/target length | `128` | `128` |
| Generation max length | n/a | `128` |
| Max position embeddings | `512` | HF model config |
| Task/data | Megatron indexed XSum corpus | `xsum` |
| Data split | `949,50,1` | train/eval subsets |
| Max train samples | full prepared Megatron corpus | `5000` |
| Max eval samples | split/eval iters | `1000` |
| Num workers | `2` | `2` |
| Mask prob | `0.15` | n/a |
| Short seq prob | `0.1` | n/a |
| Vocab extra IDs | `100` | HF tokenizer/model default |
| Tokenizer | `HFTokenizer` | `AutoTokenizer` |
| Tokenizer model | `google-t5/t5-large` | `t5-large` |

## DeepSpeed and FSDP Config

| Config knob | ZeRO-2 | ZeRO-3 | ZeRO-3 Offload | FSDP |
|---|---:|---:|---:|---:|
| Config file | `ds_configs/zero2.json` | `ds_configs/zero3.json` | `ds_configs/zero3_offload.json` | `fsdp_config.json` |
| ZeRO stage | `2` | `3` | `3` | n/a |
| BF16/FP16 | `auto` | `auto` | `auto` | script passes `bf16` |
| Train micro batch | `auto` | `auto` | `auto` | script passes `8` |
| Gradient accumulation | `auto` | `auto` | `auto` | script passes `1` |
| Overlap communication | `true` | `true` | `true` | n/a |
| Contiguous gradients | `true` | `true` | `true` | n/a |
| Reduce bucket size | `200000000` | `100000000` | `100000000` | n/a |
| Allgather bucket size | `200000000` | n/a | n/a | n/a |
| Stage 3 prefetch bucket | n/a | `50000000` | `50000000` | n/a |
| Param persistence threshold | n/a | `10000` | `10000` | n/a |
| Max live parameters | n/a | `1000000000` | not set | n/a |
| Max reuse distance | n/a | `1000000000` | not set | n/a |
| Gather 16-bit weights on save | n/a | `true` | `true` | n/a |
| Optimizer offload | none | none | CPU, pinned memory | n/a |
| Parameter offload | none | none | CPU, pinned memory | n/a |
| FSDP mode | n/a | n/a | n/a | `full_shard auto_wrap` |
| FSDP wrap class | n/a | n/a | n/a | `T5Block` |
| FSDP backward prefetch | n/a | n/a | n/a | `backward_pre` |
| FSDP forward prefetch | n/a | n/a | n/a | `false` |
| FSDP limit all gathers | n/a | n/a | n/a | `true` |
| FSDP use original parameters | n/a | n/a | n/a | `true` |
| FSDP sync module states | n/a | n/a | n/a | `false` |
| FSDP activation checkpointing | n/a | n/a | n/a | `false` |
