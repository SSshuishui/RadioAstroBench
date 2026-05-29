#!/usr/bin/env bash
set -euo pipefail

export LLM_API_KEY=""
export LLM_API_BASE="https://api.deepseek.com"
export LLM_MODEL="deepseek-v4-pro"

export CUDA_VISIBLE_DEVICES=1
export TORCH_CUDA_ARCH_LIST="8.9"

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"

mkdir -p smoke_logs_llm

fail_count=0
total_count=0

for task in $(find radio_bench/level1 radio_bench/level2 radio_bench/level3 -maxdepth 1 -name "*.py" | sort); do
  total_count=$((total_count + 1))
  name=$(echo "$task" | sed 's#/#__#g' | sed 's#.py$##')
  log_file="smoke_logs_llm/${name}.log"

  echo
  echo "============================================================"
  echo "[LLM DIRECT] ${task}"
  echo "============================================================"

  if python -m baselines.llm_direct.run_llm_direct \
      --task "${task}" \
      --scale smoke \
      --warmup 3 \
      --repeat 5 \
      --max-iters 3 \
      > "${log_file}" 2>&1; then
    echo "[OK] ${task}"
  else
    echo "[FAIL] ${task}"
    echo "  log: ${log_file}"
    fail_count=$((fail_count + 1))
  fi
done

echo
echo "============================================================"
echo "LLM smoke summary: total=${total_count}, failed=${fail_count}"
echo "============================================================"

if [ "$fail_count" -ne 0 ]; then
  exit 1
fi


# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m baselines.llm_direct.run_llm_direct \
#   --task radio_bench/level1/05-ws-build-nm1.py \
#   --scale smoke \
#   --warmup 3 \
#   --repeat 5 \
#   --max-iters 3