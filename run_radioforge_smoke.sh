#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LLM_API_KEY="${LLM_API_KEY:-}"
export LLM_API_BASE="${LLM_API_BASE:-https://api.deepseek.com}"
export LLM_MODEL="${LLM_MODEL:-deepseek-v4-pro}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:--allow-unsupported-compiler}"

if [ -z "${LLM_API_KEY:-}" ]; then
  echo "[ERROR] LLM_API_KEY is not set"
  echo "Example: export LLM_API_KEY=..."
  exit 1
fi

SCALE="${SCALE:-smoke}"
WARMUP="${WARMUP:-3}"
REPEAT="${REPEAT:-5}"
MAX_ITERS="${MAX_ITERS:-3}"
LOG_DIR="${LOG_DIR:-radioforge/runs/smoke_logs}"
mkdir -p "${LOG_DIR}"

fail_count=0
total_count=0
mapfile -t TASKS < <(find radio_bench/level1 radio_bench/level2 radio_bench/level3 -maxdepth 1 -name "*.py" | sort)

cat <<EOF2
[RADIOFORGE RADIO SMOKE]
  scale=${SCALE}
  warmup=${WARMUP}
  repeat=${REPEAT}
  max_iters=${MAX_ITERS}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
  LLM_API_BASE=${LLM_API_BASE}
  LLM_MODEL=${LLM_MODEL}
  task_count=${#TASKS[@]}
  log_dir=${LOG_DIR}
EOF2

for task in "${TASKS[@]}"; do
  total_count=$((total_count + 1))
  name=$(echo "${task}" | sed 's#/#__#g' | sed 's#.py$##')
  log_file="${LOG_DIR}/${name}.log"

  echo
  echo "============================================================"
  echo "[RADIOFORGE SMOKE] ${task}"
  echo "============================================================"

  if python -m radioforge.run_radioforge \
      --bench radio \
      --task "${task}" \
      --scale "${SCALE}" \
      --warmup "${WARMUP}" \
      --repeat "${REPEAT}" \
      --max-iters "${MAX_ITERS}" \
      --continue-after-success \
      --model "${LLM_MODEL}" \
      --api-base "${LLM_API_BASE}" \
      ${EXTRA_ARGS:-} > "${log_file}" 2>&1; then
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
echo "RadioForge smoke summary: total=${total_count}, failed=${fail_count}"
echo "============================================================"

if [ "${fail_count}" -ne 0 ]; then
  exit 1
fi
