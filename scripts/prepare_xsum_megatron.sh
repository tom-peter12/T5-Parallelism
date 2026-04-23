#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/megatron_common.sh"

require_megatron_repo

XSUM_JSONL="${XSUM_JSONL:-${MEGATRON_DATA_DIR}/xsum_text.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${MEGATRON_DATA_DIR}/xsum}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-8}"
SPLIT_SENTENCES="${SPLIT_SENTENCES:-0}"
INCLUDE_SUMMARY="${INCLUDE_SUMMARY:-1}"
PREPROCESS_PARTITIONS="${PREPROCESS_PARTITIONS:-}"
PREPROCESS_SEQ_LENGTH="${PREPROCESS_SEQ_LENGTH:-1000000}"

MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VALIDATION_SAMPLES="${MAX_VALIDATION_SAMPLES:-0}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-0}"

EXPORT_ARGS=(
  --output "${XSUM_JSONL}"
  --max-train-samples "${MAX_TRAIN_SAMPLES}"
  --max-validation-samples "${MAX_VALIDATION_SAMPLES}"
  --max-test-samples "${MAX_TEST_SAMPLES}"
)

if [[ "${INCLUDE_SUMMARY}" == "1" ]]; then
  EXPORT_ARGS+=(--include-summary)
else
  EXPORT_ARGS+=(--no-include-summary)
fi

echo "=== Export XSum corpus ==="
python "${PROJECT_DIR}/scripts/export_xsum_corpus.py" "${EXPORT_ARGS[@]}"

PREPROCESS_ARGS=(
  --input "${XSUM_JSONL}"
  --output-prefix "${OUTPUT_PREFIX}"
  --tokenizer-type "${TOKENIZER_TYPE}"
  --json-keys text
  --workers "${PREPROCESS_WORKERS}"
  --dataset-impl mmap
  --append-eod
)

if [[ "${TOKENIZER_TYPE}" == "HFTokenizer" ]]; then
  PREPROCESS_ARGS+=(--tokenizer-model "${TOKENIZER_MODEL}")
  PREPROCESS_ARGS+=(--seq-length "${PREPROCESS_SEQ_LENGTH}")
fi

if [[ "${SPLIT_SENTENCES}" == "1" ]]; then
  if [[ -z "${PREPROCESS_PARTITIONS}" ]]; then
    PREPROCESS_PARTITIONS=2
  fi
  PREPROCESS_ARGS+=(--split-sentences)
  DATA_SUFFIX="text_sentence"
else
  if [[ -z "${PREPROCESS_PARTITIONS}" ]]; then
    PREPROCESS_PARTITIONS=1
  fi
  DATA_SUFFIX="text_document"
fi

PREPROCESS_ARGS+=(--partitions "${PREPROCESS_PARTITIONS}")

if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  PREPROCESS_ARGS+=(--trust-remote-code)
fi

echo "=== Build Megatron indexed dataset ==="
python "${MEGATRON_DIR}/tools/preprocess_data.py" "${PREPROCESS_ARGS[@]}"

OUTPUT_BIN="${OUTPUT_PREFIX}_${DATA_SUFFIX}.bin"
OUTPUT_IDX="${OUTPUT_PREFIX}_${DATA_SUFFIX}.idx"

if [[ ! -s "${OUTPUT_BIN}" || ! -s "${OUTPUT_IDX}" ]]; then
  echo "Megatron preprocessing produced empty dataset artifacts." >&2
  echo "  BIN: ${OUTPUT_BIN}" >&2
  echo "  IDX: ${OUTPUT_IDX}" >&2
  echo "Check the preprocessing error above and rerun after fixing it." >&2
  exit 1
fi

echo "Done."
echo "Data prefix: ${OUTPUT_PREFIX}_${DATA_SUFFIX}"
echo "  BIN: ${OUTPUT_BIN}"
echo "  IDX: ${OUTPUT_IDX}"
