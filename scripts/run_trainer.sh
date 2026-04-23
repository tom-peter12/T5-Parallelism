#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

build_host_array
MASTER_ADDR="$(resolve_master_addr "${1:-}")"
NODE_RANK="$(resolve_node_rank "${2:-}")"
MASTER_PORT="${MASTER_PORT:-29700}"
NNODES="${NNODES:-${#HOST_ARRAY[@]}}"
NPROC=1

TASK_NAME="${TASK_NAME:-xsum}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/ddp}"
MODEL_NAME="${MODEL_NAME:-t5-large}"
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-512}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-128}"
GENERATION_MAX_LENGTH="${GENERATION_MAX_LENGTH:-128}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-10000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1000}"
PRECISION_FLAG="${PRECISION_FLAG:---bf16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:---gradient-checkpointing}"
OPTIMIZER="${OPTIMIZER:-adafactor}"
WANDB_ENTITY="${WANDB_ENTITY:-t5_mlsys}"
WANDB_PROJECT="${WANDB_PROJECT:-deepseed}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-ddp-${TASK_NAME}}"

export HF_DATASETS_CACHE="/tmp/hf_cache_$(hostname)"
export HF_HOME="/tmp/hf_cache_$(hostname)"
mkdir -p "/tmp/hf_cache_$(hostname)"

echo "=== T5 Trainer / DDP Baseline ==="
print_summary

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC}" \
  --rdzv_id=t5_trainer_job \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --node_rank="${NODE_RANK}" \
  "${PROJECT_DIR}/train.py" \
    --model-name "${MODEL_NAME}" \
    --task-name "${TASK_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --optim "${OPTIMIZER}" \
    --num-workers "${NUM_WORKERS}" \
    --max-source-length "${MAX_SOURCE_LENGTH}" \
    --max-target-length "${MAX_TARGET_LENGTH}" \
    --generation-max-length "${GENERATION_MAX_LENGTH}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-eval-samples "${MAX_EVAL_SAMPLES}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${WANDB_RUN_NAME}" \
    --learning-rate "${LEARNING_RATE}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    ${GRADIENT_CHECKPOINTING} \
    ${PRECISION_FLAG}
