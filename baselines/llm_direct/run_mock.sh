#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=$PWD:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.9"}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m baselines.llm_direct.run_llm_direct \
  --task "${1:-radio_bench/level1/05-ws-build-nm1.py}" \
  --scale "${SCALE:-smoke}" \
  --warmup 1 \
  --repeat 2 \
  --max-iters 1 \
  --mock
