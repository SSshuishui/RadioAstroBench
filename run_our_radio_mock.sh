#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=$PWD:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.9"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m our_method.run_our_method \
  --bench radio \
  --task radio_bench/level1/05-ws-build-nm1.py \
  --scale smoke \
  --warmup 1 \
  --repeat 2 \
  --rounds 1 \
  --mock
