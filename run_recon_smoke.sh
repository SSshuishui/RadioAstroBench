#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=$PWD:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST="8.9"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m radio_astronomy_cuda_bench.run_smoke \
  --task radio_bench/level2/03-ws-recon-representative.py \
  --scale smoke \
  --warmup 10 \
  --repeat 50
