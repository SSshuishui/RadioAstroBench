#!/usr/bin/env bash
set -euo pipefail

# export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"


# export LLM_API_KEY="${LLM_API_KEY:-}"
export LLM_API_BASE="${LLM_API_BASE:-https://api.deepseek.com}"
export LLM_MODEL="${LLM_MODEL:-deepseek-v4-pro}"
# Set LLM_API_KEY before running.


CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m baselines.kernelmem.run_kernelmem \
  --bench radio \
  --task "${1:-radio_bench/level1/05-ws-build-nm1.py}" \
  --scale "${SCALE:-smoke}" \
  --warmup "${WARMUP:-3}" \
  --repeat "${REPEAT:-5}" \
  --max-iters "${MAX_ITERS:-${ROUNDS:-3}}" \
  --continue-after-success \
  --model "${LLM_MODEL}" \
  --api-base "${LLM_API_BASE}" \
  ${EXTRA_ARGS:-}
