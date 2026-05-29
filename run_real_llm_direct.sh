#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-nside512_day1_10m_ring}"
SCALE="${SCALE:-nside512_full}"
WARMUP="${WARMUP:-1}"
REPEAT="${REPEAT:-5}"
MAX_ITERS="${MAX_ITERS:-5}"


export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.9"

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

# export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:--allow-unsupported-compiler}"
# export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export RKB_RADIO_ASTRO_DATA_ROOT="${RKB_RADIO_ASTRO_DATA_ROOT:-$PWD/radio_astro_data}"

export LLM_API_KEY=""
export LLM_API_BASE="https://api.deepseek.com"
export LLM_MODEL="deepseek-v4-pro"

if [ -z "${LLM_API_KEY:-}" ]; then
  echo "[ERROR] LLM_API_KEY is not set"
  exit 1
fi

mkdir -p real_logs_llm
fail_count=0
total_count=0

mapfile -t TASKS < <(find radio_bench/level1 radio_bench/level2 radio_bench/level3 -maxdepth 1 -name "*.py" | sort)

cat <<EOF
[REAL LLM DIRECT]
  profile=${PROFILE}
  scale=${SCALE}
  warmup=${WARMUP}
  repeat=${REPEAT}
  max_iters=${MAX_ITERS}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
  RKB_RADIO_ASTRO_DATA_ROOT=${RKB_RADIO_ASTRO_DATA_ROOT}
  LLM_API_BASE=${LLM_API_BASE}
  LLM_MODEL=${LLM_MODEL}
  task_count=${#TASKS[@]}
EOF

for task in "${TASKS[@]}"; do
  total_count=$((total_count + 1))
  name=$(echo "${task}" | sed 's#/#__#g' | sed 's#.py$##')
  log_file="real_logs_llm/${name}.log"

  echo
  echo "============================================================"
  echo "[REAL LLM DIRECT] ${task}"
  echo "============================================================"

  if python -m baselines.llm_direct.run_llm_direct \
      --task "${task}" \
      --scale "${SCALE}" \
      --fixture-profile "${PROFILE}" \
      --warmup "${WARMUP}" \
      --repeat "${REPEAT}" \
      --max-iters "${MAX_ITERS}" \
      > "${log_file}" 2>&1; then
    echo "[OK] ${task}"
  else
    echo "[FAIL] ${task}"
    echo "  log: ${log_file}"
    tail -80 "${log_file}" || true
    fail_count=$((fail_count + 1))
  fi
done

echo
echo "============================================================"
echo "REAL LLM summary: total=${total_count}, failed=${fail_count}"
echo "============================================================"

if [ "${fail_count}" -ne 0 ]; then
  exit 1
fi