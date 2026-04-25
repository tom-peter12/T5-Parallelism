#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

export MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_DIR}/../Megatron-DeepSpeed}"
export MEGATRON_DATA_DIR="${MEGATRON_DATA_DIR:-${PROJECT_DIR}/megatron_data}"
export TOKENIZER_TYPE="${TOKENIZER_TYPE:-HFTokenizer}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-google-t5/t5-large}"

mkdir -p "${MEGATRON_DATA_DIR}"

require_megatron_repo() {
  if [[ ! -d "${MEGATRON_DIR}" ]]; then
    echo "Megatron directory not found: ${MEGATRON_DIR}" >&2
    echo "Set MEGATRON_DIR to your Megatron-DeepSpeed checkout path." >&2
    exit 1
  fi

  if [[ ! -f "${MEGATRON_DIR}/pretrain_t5.py" ]]; then
    echo "Missing ${MEGATRON_DIR}/pretrain_t5.py" >&2
    exit 1
  fi

  if [[ ! -f "${MEGATRON_DIR}/tools/preprocess_data.py" ]]; then
    echo "Missing ${MEGATRON_DIR}/tools/preprocess_data.py" >&2
    exit 1
  fi
}

build_megatron_wandb_args() {
  MEGATRON_WANDB_ARGS=()

  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    export WANDB_ENTITY="${WANDB_ENTITY:-t5_mlsys}"
    MEGATRON_WANDB_ARGS=(
      --wandb-project "${WANDB_PROJECT}"
      --wandb-exp-name "${WANDB_RUN_NAME:-}"
      --wandb-save-dir "${WANDB_SAVE_DIR:-${OUTPUT_DIR}/wandb}"
    )
  fi
}

resolve_megatron_training_schedule() {
  EPOCHS="${EPOCHS:-1}"
  MEGATRON_XSUM_JSONL="${MEGATRON_XSUM_JSONL:-${MEGATRON_DATA_DIR}/xsum_text.jsonl}"

  if [[ -z "${TRAIN_ITERS:-}" ]]; then
    if [[ ! -s "${MEGATRON_XSUM_JSONL}" ]]; then
      echo "Unable to compute one-epoch TRAIN_ITERS because ${MEGATRON_XSUM_JSONL} is missing or empty." >&2
      echo "Run scripts/prepare_xsum_megatron.sh first, or set TRAIN_ITERS explicitly." >&2
      exit 1
    fi

    TRAIN_ITERS="$(
      python - "${MEGATRON_XSUM_JSONL}" "${DATA_SPLIT}" "${GLOBAL_BATCH_SIZE}" "${EPOCHS}" <<'PY'
import math
import sys

jsonl_path, data_split, global_batch_size, epochs = sys.argv[1:]
with open(jsonl_path, "r", encoding="utf-8") as handle:
    total_rows = sum(1 for _ in handle)

split_parts = [float(part) for part in data_split.split(",")]
if not split_parts or split_parts[0] <= 0:
    raise ValueError(f"DATA_SPLIT must start with a positive train split: {data_split}")

train_fraction = split_parts[0] / sum(split_parts)
train_rows = total_rows * train_fraction * float(epochs)
print(max(1, math.ceil(train_rows / int(global_batch_size))))
PY
    )"
  fi

  LR_DECAY_ITERS="${LR_DECAY_ITERS:-${TRAIN_ITERS}}"
}
