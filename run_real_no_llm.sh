#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-nside512_day1_10m_ring}"
SCALE="${SCALE:-nside512_full}"
WARMUP="${WARMUP:-1}"
REPEAT="${REPEAT:-2}"

export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.9"

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

# export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:--allow-unsupported-compiler}"
# export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export RKB_RADIO_ASTRO_DATA_ROOT="${RKB_RADIO_ASTRO_DATA_ROOT:-$PWD/radio_astro_data}"


mkdir -p real_logs_no_llm
LOG="real_logs_no_llm/${PROFILE}_${SCALE}.log"

cat <<EOF
[REAL NO-LLM]
  profile=${PROFILE}
  scale=${SCALE}
  warmup=${WARMUP}
  repeat=${REPEAT}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
  RKB_RADIO_ASTRO_DATA_ROOT=${RKB_RADIO_ASTRO_DATA_ROOT}
  log=${LOG}
EOF

python -m radio_astronomy_cuda_bench.run_bench \
  --task all \
  --scale "${SCALE}" \
  --fixture-profile "${PROFILE}" \
  --warmup "${WARMUP}" \
  --repeat "${REPEAT}" \
  2>&1 | tee "${LOG}"
