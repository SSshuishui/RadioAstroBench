#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=$PWD:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.9"}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m baselines.llm_direct.run_llm_direct \
  --task radio_bench/level1/05-ws-build-nm1.py \
  --scale smoke \
  --warmup 3 \
  --repeat 5 \
  --max-iters 3
