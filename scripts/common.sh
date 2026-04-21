#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}-triton-cache}"

if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
  export CUDA_HOME="/usr/local/cuda"
elif [[ -x "/usr/local/cuda-12.2/bin/nvcc" ]]; then
  export CUDA_HOME="/usr/local/cuda-12.2"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"

DEFAULT_HOSTS=(
  ws-l4-002
  ws-l4-010
)

build_host_array() {
  if [[ -n "${HOSTS:-}" ]]; then
    read -r -a HOST_ARRAY <<< "${HOSTS}"
  else
    HOST_ARRAY=("${DEFAULT_HOSTS[@]}")
  fi
}

resolve_master_addr() {
  local provided="${1:-}"
  if [[ -n "${provided}" ]]; then
    echo "${provided}"
    return
  fi
  echo "${HOST_ARRAY[0]}"
}

resolve_node_rank() {
  local provided="${1:-}"
  if [[ -n "${provided}" ]]; then
    echo "${provided}"
    return
  fi

  local this_host
  this_host="$(hostname -s)"
  local i
  for i in "${!HOST_ARRAY[@]}"; do
    if [[ "${HOST_ARRAY[$i]}" == "${this_host}" ]]; then
      echo "${i}"
      return
    fi
  done

  echo "Unable to infer NODE_RANK for host ${this_host}. Pass it explicitly." >&2
  exit 1
}

print_summary() {
  echo "host        : $(hostname -s)"
  echo "master_addr : ${MASTER_ADDR}:${MASTER_PORT}"
  echo "node_rank   : ${NODE_RANK} / ${NNODES}"
  echo "cuda_home   : ${CUDA_HOME:-unset}"
  echo "task        : ${TASK_NAME}"
  echo "output_dir  : ${OUTPUT_DIR}"
  echo "epochs      : ${EPOCHS}"
  echo "batch_size  : ${PER_DEVICE_BATCH_SIZE}"
  echo "grad_accum  : ${GRAD_ACCUM}"
  echo "train_sub   : ${MAX_TRAIN_SAMPLES}"
  echo "eval_sub    : ${MAX_EVAL_SAMPLES}"
  echo "wandb_entity: ${WANDB_ENTITY:-t5_mlsys}"
  echo "wandb_proj  : ${WANDB_PROJECT:-deepseed}"
  echo "wandb_run   : ${WANDB_RUN_NAME:-(auto)}"
}
