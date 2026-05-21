#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=$PWD:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.9"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m baselines.cudaforge.run_cudaforge \
  --dataset kernelbench \
  --task "${1:-kernelbench/level1/1_Square_matrix_multiplication_.py}" \
  --gpu "${GPU_NAME:-RTX 4090}" \
  --server_type "${SERVER_TYPE:-deepseek}" \
  --model_name "${MODEL_NAME:-deepseek-coder}" \
  --round "${ROUND:-3}" \
  --warmup "${WARMUP:-3}" \
  --repeat "${REPEAT:-5}" \
  ${EXTRA_ARGS:-}
