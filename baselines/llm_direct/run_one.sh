#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=$PWD:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.9"}

# Required for real LLM calls:
#   export LLM_API_KEY="..."
#   export LLM_API_BASE="https://.../v1"
#   export LLM_MODEL="..."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m baselines.llm_direct.run_llm_direct \
  --task "${1:-radio_bench/level1/05-ws-build-nm1.py}" \
  --scale "${SCALE:-smoke}" \
  --warmup "${WARMUP:-3}" \
  --repeat "${REPEAT:-5}" \
  --max-iters "${MAX_ITERS:-3}" \
  ${EXTRA_ARGS:-}
