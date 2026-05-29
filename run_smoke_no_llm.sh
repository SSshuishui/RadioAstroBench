#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.9"

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++

# 可先保留
export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"

mkdir -p smoke_logs_no_llm

fail_count=0
total_count=0

for task in $(find radio_bench/level1 radio_bench/level2 radio_bench/level3 -maxdepth 1 -name "*.py" | sort); do
  total_count=$((total_count + 1))
  name=$(echo "$task" | sed 's#/#__#g' | sed 's#.py$##')

  echo
  echo "============================================================"
  echo "[NO-LLM SMOKE] $task"
  echo "============================================================"

  if python -m radio_astronomy_cuda_bench.run_smoke \
      --task "$task" \
      --scale smoke \
      --warmup 3 \
      --repeat 5 \
      > "smoke_logs_no_llm/${name}.log" 2>&1; then
    echo "[OK] $task"
  else
    echo "[FAIL] $task"
    echo "  log: smoke_logs_no_llm/${name}.log"
    fail_count=$((fail_count + 1))
  fi
done

echo
echo "============================================================"
echo "NO-LLM smoke summary: total=${total_count}, failed=${fail_count}"
echo "============================================================"

if [ "$fail_count" -ne 0 ]; then
  exit 1
fi