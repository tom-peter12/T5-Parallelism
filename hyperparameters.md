# T5 Parallelism Hyperparameters

Defaults below are taken from the Megatron run scripts and DeepSpeed config files as written, assuming no environment overrides. All listed strategies run Megatron T5 pre-training on the exported XSum corpus.

## Launch and Parallelism

| Strategy | Script | Backend | Model | Nodes | Procs/GPU per node | Total procs/GPUs | TP | PP | PP split | Data-parallel degree | Port | Precision |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DDP | `scripts/run_megatron_t5_ddp.sh` | Megatron DDP | T5-large architecture; tokenizer `google-t5/t5-large` | `2` | `1` | `2` | `1` | `1` | n/a | `2` | `29800` | `bf16` |
| Pipeline | `scripts/run_megatron_t5_pipeline.sh` | Megatron pipeline parallel | T5-large architecture; tokenizer `google-t5/t5-large` | `2` | `1` | `2` | `1` | `2` | `1` | `1` | `29810` | `bf16` |
| Tensor | `scripts/run_megatron_t5_tensor.sh` | Megatron tensor parallel | T5-large architecture; tokenizer `google-t5/t5-large` | `2` | `1` | `2` | `2` | `1` | n/a | `1` | `29811` | `bf16` |
| Hybrid | `scripts/run_megatron_t5_hybrid.sh` | Megatron tensor + pipeline parallel | T5-large architecture; tokenizer `google-t5/t5-large` | `4` | `1` | `4` | `2` | `2` | `1` | `1` | `29812` | `bf16` |
| Megatron ZeRO-1 | `scripts/run_megatron_t5_zero1.sh` | Megatron + DeepSpeed | T5-large architecture; tokenizer `google-t5/t5-large` | `2` | `1` | `2` | `1` | `1` | n/a | `2` | `29801` | `bf16` |
| Megatron ZeRO-2 | `scripts/run_megatron_t5_zero2.sh` | Megatron + DeepSpeed | T5-large architecture; tokenizer `google-t5/t5-large` | `2` | `1` | `2` | `1` | `1` | n/a | `2` | `29802` | `bf16` |
| Megatron ZeRO-3 | `scripts/run_megatron_t5_zero3.sh` | Megatron + DeepSpeed | T5-large architecture; tokenizer `google-t5/t5-large` | `2` | `1` | `2` | `1` | `1` | n/a | `2` | `29803` | `bf16` |
| Megatron ZeRO-3 Offload | `scripts/run_megatron_t5_zero3_offload.sh` | Megatron + DeepSpeed | T5-large architecture; tokenizer `google-t5/t5-large` | `2` | `1` | `2` | `1` | `1` | n/a | `2` | `29804` | `bf16` |

## Training Hyperparameters

| Strategy group | Micro batch | Global batch | Gradient accumulation | LR | Min LR | LR decay | Warmup | Weight decay | Grad clip | Grad checkpointing | Save | Eval | Log |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---|---|---|---|
| DDP / pipeline / tensor / hybrid | `1` | `16` | derived by Megatron | `1e-4` | `1e-5` | `linear` | `0.01` fraction | `0.01` | `1.0` | disabled | every 20% of iterations | every 10% of iterations; `20` eval iters | every iteration |
| Megatron ZeRO-1/2/3/offload | `1` | `16` | DeepSpeed config: `8` | `1e-4` | `1e-5` | `linear` | `0.01` fraction | `0.01` | `1.0` | disabled | every 20% of iterations | every 10% of iterations; `20` eval iters | every iteration |

Common values:

| Hyperparameter | Value |
|---|---:|
| Epochs | `1` |
| Train iterations | computed from XSum JSONL, data split, global batch, and epochs |
| LR decay iters | defaults to `TRAIN_ITERS` |
| Data split | `949,50,1` |
| Num workers | `2` |

## Model and Sequence/Data Settings

| Hyperparameter | Value |
|---|---:|
| Objective | Megatron T5 pre-training on exported XSum text |
| Encoder layers | `24` |
| Decoder layers | `24` |
| Hidden size | `1024` |
| Attention heads | `16` |
| KV channels | `64` |
| FFN hidden size | `2816` |
| Encoder/source length | `512` |
| Decoder/target length | `128` |
| Max position embeddings | `512` |
| Task/data | Megatron indexed XSum corpus |
| Data path default | `${MEGATRON_DATA_DIR}/xsum_text_document` |
| Sentence-split data path | `${MEGATRON_DATA_DIR}/xsum_text_sentence` when `SPLIT_SENTENCES=1` |
| Mask prob | `0.15` |
| Short seq prob | `0.1` |
| Vocab extra IDs | `100` |
| Tokenizer | `HFTokenizer` |
| Tokenizer model | `google-t5/t5-large` |

## DeepSpeed Config

| Strategy | Config file | ZeRO stage | BF16 | Grad accumulation | Micro batch/GPU | Grad clip | Overlap comm | Reduce bucket | Allgather bucket | Stage 3 prefetch | Stage 3 persistence | Stage 3 max live/reuse | Save gathered 16-bit weights | Optimizer offload | Parameter offload |
|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|
| Megatron ZeRO-1 | `ds_configs/megatron_zero1.json` | `1` | `true` | `8` | `1` | `1.0` | `true` | `200000000` | `200000000` | n/a | n/a | n/a | n/a | none | none |
| Megatron ZeRO-2 | `ds_configs/megatron_zero2.json` | `2` | `true` | `8` | `1` | `1.0` | `false` | `200000000` | `200000000` | n/a | n/a | n/a | n/a | none | none |
| Megatron ZeRO-3 | `ds_configs/megatron_zero3.json` | `3` | `true` | `8` | `1` | `1.0` | `false` | `100000000` | n/a | `50000000` | `10000` | `1000000000` | `true` | none | none |
| Megatron ZeRO-3 Offload | `ds_configs/megatron_zero3_offload.json` | `3` | `true` | `8` | `1` | `1.0` | `false` | `100000000` | n/a | `50000000` | `10000` | `1000000000` | `true` | CPU, pinned memory | CPU, pinned memory |
