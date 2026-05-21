#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=$PWD:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.9"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m baselines.kernelmem.run_kernelmem \
  --bench kernelbench \
  --task "${1:-kernelbench/level1/1_Square_matrix_multiplication_.py}" \
  --warmup 1 \
  --repeat 2 \
  --rounds 1 \
  --mock
