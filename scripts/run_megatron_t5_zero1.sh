#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/megatron_common.sh"

require_megatron_repo
build_host_array

MASTER_ADDR="$(resolve_master_addr "${1:-}")"
NODE_RANK="$(resolve_node_rank "${2:-}")"
MASTER_PORT="${MASTER_PORT:-29801}"
NNODES="${NNODES:-${#HOST_ARRAY[@]}}"
NPROC="${NPROC:-1}"

export CUDA_DEVICE_MAX_CONNECTIONS=1

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/megatron_zero1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${OUTPUT_DIR}/checkpoints}"
SPLIT_SENTENCES="${SPLIT_SENTENCES:-0}"
DS_CONFIG="${DS_CONFIG:-${PROJECT_DIR}/ds_configs/megatron_zero1.json}"

DATA_PREFIX_BASE="${DATA_PREFIX_BASE:-${MEGATRON_DATA_DIR}/xsum}"
if [[ "${SPLIT_SENTENCES}" == "1" ]]; then
  DATA_PATH="${DATA_PATH:-${DATA_PREFIX_BASE}_text_sentence}"
else
  DATA_PATH="${DATA_PATH:-${DATA_PREFIX_BASE}_text_document}"
fi

TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"

NUM_LAYERS="${NUM_LAYERS:-24}"
ENCODER_NUM_LAYERS="${ENCODER_NUM_LAYERS:-${NUM_LAYERS}}"
DECODER_NUM_LAYERS="${DECODER_NUM_LAYERS:-${ENCODER_NUM_LAYERS}}"
HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-16}"
KV_CHANNELS="${KV_CHANNELS:-64}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-2816}"

ENCODER_SEQ_LENGTH="${ENCODER_SEQ_LENGTH:-512}"
DECODER_SEQ_LENGTH="${DECODER_SEQ_LENGTH:-128}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-512}"

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
TRAIN_ITERS="${TRAIN_ITERS:-1000}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-1000}"
LR="${LR:-0.0001}"
MIN_LR="${MIN_LR:-0.00001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.01}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
MASK_PROB="${MASK_PROB:-0.15}"
SHORT_SEQ_PROB="${SHORT_SEQ_PROB:-0.1}"

NUM_WORKERS="${NUM_WORKERS:-2}"
DATA_SPLIT="${DATA_SPLIT:-949,50,1}"
EVAL_ITERS="${EVAL_ITERS:-20}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
VOCAB_EXTRA_IDS="${VOCAB_EXTRA_IDS:-100}"

PRECISION_FLAG="${PRECISION_FLAG:---bf16}"
ADDITIONAL_ARGS="${ADDITIONAL_ARGS:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-megatron-zero1}"

mkdir -p "${OUTPUT_DIR}" "${CHECKPOINT_PATH}"

if [[ ! -s "${DATA_PATH}.bin" || ! -s "${DATA_PATH}.idx" ]]; then
  echo "Missing or empty Megatron indexed dataset files for prefix: ${DATA_PATH}" >&2
  echo "Run scripts/prepare_xsum_megatron.sh first." >&2
  exit 1
fi

TOKENIZER_ARGS=(--tokenizer-type "${TOKENIZER_TYPE}")
if [[ "${TOKENIZER_TYPE}" == "HFTokenizer" ]]; then
  TOKENIZER_ARGS+=(--tokenizer-model "${TOKENIZER_MODEL}")
else
  if [[ -z "${VOCAB_FILE:-}" ]]; then
    echo "VOCAB_FILE is required when TOKENIZER_TYPE is not HFTokenizer." >&2
    exit 1
  fi
  TOKENIZER_ARGS+=(--vocab-file "${VOCAB_FILE}")
fi

if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  TOKENIZER_ARGS+=(--trust-remote-code)
fi

TOTAL_PROCS=$((NNODES * NPROC))
echo "=== Megatron T5 ZeRO-1 (XSum corpus) ==="
echo "host                  : $(hostname -s)"
echo "master_addr           : ${MASTER_ADDR}:${MASTER_PORT}"
echo "node_rank             : ${NODE_RANK} / ${NNODES}"
echo "megatron_dir          : ${MEGATRON_DIR}"
echo "data_path             : ${DATA_PATH}"
echo "checkpoint_path       : ${CHECKPOINT_PATH}"
echo "deepspeed_config      : ${DS_CONFIG}"
echo "tensor_mp_size        : ${TENSOR_MODEL_PARALLEL_SIZE}"
echo "pipeline_mp_size      : ${PIPELINE_MODEL_PARALLEL_SIZE}"
echo "data_parallel_degree  : ${TOTAL_PROCS}"
echo "encoder/decoder layers: ${ENCODER_NUM_LAYERS}/${DECODER_NUM_LAYERS}"
echo "micro/global batch    : ${MICRO_BATCH_SIZE}/${GLOBAL_BATCH_SIZE}"
echo "train_iters           : ${TRAIN_ITERS}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC}" \
  --rdzv_id=t5_megatron_zero1_job \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --node_rank="${NODE_RANK}" \
  "${MEGATRON_DIR}/pretrain_t5.py" \
  "${TOKENIZER_ARGS[@]}" \
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}" \
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}" \
  --deepspeed \
  --deepspeed_config "${DS_CONFIG}" \
  --encoder-num-layers "${ENCODER_NUM_LAYERS}" \
  --decoder-num-layers "${DECODER_NUM_LAYERS}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --num-attention-heads "${NUM_ATTENTION_HEADS}" \
  --kv-channels "${KV_CHANNELS}" \
  --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
  --encoder-seq-length "${ENCODER_SEQ_LENGTH}" \
  --decoder-seq-length "${DECODER_SEQ_LENGTH}" \
  --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}" \
  --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --lr "${LR}" \
  --train-iters "${TRAIN_ITERS}" \
  --lr-decay-iters "${LR_DECAY_ITERS}" \
  --lr-decay-style linear \
  --min-lr "${MIN_LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --lr-warmup-fraction "${LR_WARMUP_FRACTION}" \
  --clip-grad "${CLIP_GRAD}" \
  --mask-prob "${MASK_PROB}" \
  --short-seq-prob "${SHORT_SEQ_PROB}" \
  --vocab-extra-ids "${VOCAB_EXTRA_IDS}" \
  --data-path "${DATA_PATH}" \
  --data-impl mmap \
  --split "${DATA_SPLIT}" \
  --num-workers "${NUM_WORKERS}" \
  --log-interval "${LOG_INTERVAL}" \
  --save-interval "${SAVE_INTERVAL}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --eval-iters "${EVAL_ITERS}" \
  --distributed-backend nccl \
  --no-pipeline-parallel \
  --save "${CHECKPOINT_PATH}" \
  --load "${CHECKPOINT_PATH}" \
  ${PRECISION_FLAG} \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-exp-name "${WANDB_RUN_NAME}" \
  ${ADDITIONAL_ARGS}
