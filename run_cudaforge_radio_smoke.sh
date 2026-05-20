#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=$PWD:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST="8.9"

: "${LLM_API_KEY:?Set LLM_API_KEY}"
: "${LLM_API_BASE:?Set LLM_API_BASE, e.g. https://.../v1}"
: "${LLM_MODEL:?Set LLM_MODEL, e.g. deepseek-v4-pro}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m baselines.cudaforge.run_cudaforge \
  --dataset radio \
  --task radio_bench/level1/05-ws-build-nm1.py \
  --scale smoke \
  --warmup 3 \
  --repeat 5 \
  --max-iters 3 \
  --judge-after-success
