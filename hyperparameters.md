# T5 Parallelism Hyperparameters

Defaults below are taken from the run scripts and config files as written, assuming no environment overrides.

## Launch and Parallelism

| Strategy | Script | Backend | Model | Nodes | Procs/GPU per node | Total procs/GPUs | TP | PP | PP split | Data-parallel degree | Port | Precision |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Hybrid | `scripts/run_megatron_t5_hybrid.sh` | Megatron | T5-large architecture; tokenizer `google-t5/t5-large` | `4` | `1` | `4` | `2` | `2` | `1` | `1` | `29812` | `bf16` |
| Pipeline | `scripts/run_megatron_t5_pipeline.sh` | Megatron | T5-large architecture; tokenizer `google-t5/t5-large` | `2` | `1` | `2` | `1` | `2` | `1` | `1` | `29810` | `bf16` |
| Tensor | `scripts/run_megatron_t5_tensor.sh` | Megatron | T5-large architecture; tokenizer `google-t5/t5-large` | `2` | `1` | `2` | `2` | `1` | n/a | `1` | `29811` | `bf16` |
| Megatron ZeRO-1 | `scripts/run_megatron_t5_zero1.sh` | Megatron + DeepSpeed | T5-large architecture; tokenizer `google-t5/t5-large` | `2` | `1` | `2` | `1` | `1` | n/a | `2` | `29801` | `bf16` |
| Megatron ZeRO-2 | `scripts/run_megatron_t5_zero2.sh` | Megatron + DeepSpeed | T5-large architecture; tokenizer `google-t5/t5-large` | `2` | `1` | `2` | `1` | `1` | n/a | `2` | `29802` | `bf16` |
| Megatron ZeRO-3 | `scripts/run_megatron_t5_zero3.sh` | Megatron + DeepSpeed | T5-large architecture; tokenizer `google-t5/t5-large` | `2` | `1` | `2` | `1` | `1` | n/a | `2` | `29803` | `bf16` |
| Megatron ZeRO-3 Offload | `scripts/run_megatron_t5_zero3_offload.sh` | Megatron + DeepSpeed | T5-large architecture; tokenizer `google-t5/t5-large` | `2` | `1` | `2` | `1` | `1` | n/a | `2` | `29804` | `bf16` |
| HF ZeRO-2 | `scripts/run_zero2.sh` | HF Trainer + DeepSpeed | `t5-large` | `2` | `1` | `2` | n/a | n/a | n/a | `2` | `29702` | `bf16` |
| HF ZeRO-3 | `scripts/run_zero3.sh` | HF Trainer + DeepSpeed | `t5-large` | `2` | `1` | `2` | n/a | n/a | n/a | `2` | `29703` | `bf16` |
| HF ZeRO-3 Offload | `scripts/run_zero3_offload.sh` | HF Trainer + DeepSpeed | `t5-large` | `2` | `1` | `2` | n/a | n/a | n/a | `2` | `29704` | `bf16` |
| FSDP | `scripts/run_fsdp.sh` | HF Trainer + FSDP | `t5-large` | `2` | `1` | `2` | n/a | n/a | n/a | `2` | `29701` | `bf16` |

## Training Hyperparameters

| Strategy group | Micro/per-device train batch | Global/effective batch | Gradient accumulation | LR | Min LR | LR decay | Warmup | Weight decay | Grad clip | Grad checkpointing | Save | Eval | Log |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---|---|---|---|
| Megatron parallel: hybrid/pipeline/tensor | `1` | `16` | derived by Megatron | `1e-4` | `1e-5` | `linear` | `0.01` fraction | `0.01` | `1.0` | disabled | every 20% of iterations | every 10% of iterations; `20` eval iters | every iteration |
| Megatron ZeRO: ZeRO-1/2/3/offload | `1` | `16` | DeepSpeed config: `8` | `1e-4` | `1e-5` | `linear` | `0.01` fraction | `0.01` | `1.0` | disabled | every 20% of iterations | every 10% of iterations; `20` eval iters | every iteration |
| HF ZeRO/FSDP | `8` | `16` (`8 * 2 * 1`) | `1` | `1e-4` | n/a | `linear` HF default | `0.01` ratio | `0.01` | `1.0` | disabled | every 20% of steps | every 10% of steps | every step |

Common values:

| Hyperparameter | Value |
|---|---:|
| Epochs | `1` |
| Megatron train iterations | computed from XSum JSONL, data split, global batch, and epochs |
| HF train steps | epoch-based, with launcher-computed save/eval step intervals |
| HF eval batch size | `8` |
| HF save total limit | `2` from `train.py` |
| HF seed | `42` from `train.py` |

## Model and Sequence/Data Settings

| Hyperparameter | Megatron strategies | HF ZeRO/FSDP strategies |
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
| Data/objective | T5 span-corruption over exported XSum text | supervised XSum summarization |
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

| Strategy | Config file | Stage/mode | BF16 | Grad accumulation | Micro batch/GPU | Grad clip | Overlap comm | Reduce bucket | Allgather bucket | Stage 3 prefetch | Stage 3 persistence | Stage 3 max live/reuse | Save gathered 16-bit weights | Optimizer offload | Parameter offload |
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|
| Megatron ZeRO-1 | `ds_configs/megatron_zero1.json` | ZeRO stage `1` | `true` | `8` | `1` | `1.0` | `true` | `200000000` | `200000000` | n/a | n/a | n/a | n/a | none | none |
| Megatron ZeRO-2 | `ds_configs/megatron_zero2.json` | ZeRO stage `2` | `true` | `8` | `1` | `1.0` | `false` | `200000000` | `200000000` | n/a | n/a | n/a | n/a | none | none |
| Megatron ZeRO-3 | `ds_configs/megatron_zero3.json` | ZeRO stage `3` | `true` | `8` | `1` | `1.0` | `false` | `100000000` | n/a | `50000000` | `10000` | `1000000000` | `true` | none | none |
| Megatron ZeRO-3 Offload | `ds_configs/megatron_zero3_offload.json` | ZeRO stage `3` | `true` | `8` | `1` | `1.0` | `false` | `100000000` | n/a | `50000000` | `10000` | `1000000000` | `true` | CPU, pinned memory | CPU, pinned memory |
| HF ZeRO-2 | `ds_configs/zero2.json` | ZeRO stage `2` | `auto` | `auto` | `auto` | `1.0` | `true` | `200000000` | `200000000` | n/a | n/a | n/a | n/a | none | none |
| HF ZeRO-3 | `ds_configs/zero3.json` | ZeRO stage `3` | `auto` | `auto` | `auto` | `1.0` | `true` | `100000000` | n/a | `50000000` | `10000` | `1000000000` | `true` | none | none |
| HF ZeRO-3 Offload | `ds_configs/zero3_offload.json` | ZeRO stage `3` | `auto` | `auto` | `auto` | `1.0` | `true` | `100000000` | n/a | `50000000` | `10000` | not set | `true` | CPU, pinned memory | CPU, pinned memory |
| FSDP | `fsdp_config.json` | `full_shard auto_wrap` | script passes `bf16` | script passes `1` | script passes `8` | script passes `1.0` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

FSDP-specific settings:

| Config knob | Value |
|---|---:|
| Wrap class | `T5Block` |
| Backward prefetch | `backward_pre` |
| Forward prefetch | `false` |
| Limit all gathers | `true` |
| Use original parameters | `true` |
| Sync module states | `false` |
| Activation checkpointing | `false` |
