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
