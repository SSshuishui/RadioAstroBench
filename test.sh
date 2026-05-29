#!/usr/bin/env bash
set -euo pipefail

export LLM_API_KEY="sk-0f7bda33f62e4413ae233c93b8492973"

export LLM_API_BASE="https://api.deepseek.com"
export LLM_MODEL="deepseek-v4-pro"
export LLM_MAX_TOKENS=12000

export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.9"

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"

python -m baselines.llm_direct.run_llm_direct \
  --task radio_bench/level2/03-ws-recon-representative.py \
  --scale nside512_full \
  --fixture-profile nside512_day1_10m_ring \
  --warmup 1 \
  --repeat 5 \
  --max-iters 3