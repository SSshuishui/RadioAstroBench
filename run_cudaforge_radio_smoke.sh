#!/usr/bin/env bash
set -euo pipefail

# export LLM_API_KEY="${LLM_API_KEY:-}"
export LLM_API_KEY="sk-98edded97d6b42ceb0b676b5b5702d5b"
export LLM_API_BASE="${LLM_API_BASE:-https://api.deepseek.com}"
export LLM_MODEL="${LLM_MODEL:-deepseek-v4-pro}"

export CUDA_VISIBLE_DEVICES=1
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

mkdir -p smoke_logs_cudaforge

fail_count=0
total_count=0

mapfile -t TASKS < <(find radio_bench/level1 radio_bench/level2 radio_bench/level3 -maxdepth 1 -name "*.py" | sort)

for task in "${TASKS[@]}"; do
  total_count=$((total_count + 1))
  name=$(echo "${task}" | sed 's#/#__#g' | sed 's#.py$##')
  log_file="smoke_logs_cudaforge/${name}.log"

  echo
  echo "============================================================"
  echo "[CUDAFORGE SMOKE] ${task}"
  echo "============================================================"

  if python -m baselines.cudaforge.run_cudaforge \
        --dataset radio \
        --task "${task}" \
        --scale smoke \
        --warmup 3 \
        --repeat 5 \
        --max-iters 3 
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
echo "CUDAForge smoke summary: total=${total_count}, failed=${fail_count}"
echo "============================================================"

if [ "${fail_count}" -ne 0 ]; then
  exit 1
fi
