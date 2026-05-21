# KernelMem baseline adapter

This directory contains a framework-local KernelMem-style baseline extracted from the uploaded `KernelMem-main.zip`.

It keeps the core KernelMem ideas:

- long-term memory from `memorybank/` and prompt/memory files;
- short-term memory from previous local runs;
- LLM generation of a candidate `ModelNew`;
- compile/runtime/correctness feedback repair loop;
- optional NCU profiling feedback for `radio_bench` tasks;
- unified support for `radio_bench/` and `kernelbench/`.

## Benchmarks

Supported benchmark roots:

```text
radio_bench/     # existing scientific CUDA kernel benchmark
kernelbench/     # copied from KernelMem-main/KernelBench
```

## Mock harness test

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST="8.9"

CUDA_VISIBLE_DEVICES=0 python -m baselines.kernelmem.run_kernelmem \
  --bench radio \
  --task radio_bench/level1/05-ws-build-nm1.py \
  --scale smoke \
  --warmup 1 \
  --repeat 2 \
  --rounds 1 \
  --mock
```

Or:

```bash
bash baselines/kernelmem/run_radio_mock.sh
```

## Real LLM run on radio_bench

```bash
export LLM_API_KEY="..."
export LLM_API_BASE="https://.../v1"
export LLM_MODEL="deepseek-v4-pro"
export PYTHONPATH=$PWD:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST="8.9"

CUDA_VISIBLE_DEVICES=0 python -m baselines.kernelmem.run_kernelmem \
  --bench radio \
  --task radio_bench/level1/05-ws-build-nm1.py \
  --scale smoke \
  --warmup 3 \
  --repeat 5 \
  --rounds 3
```

## Real LLM run on KernelBench

```bash
CUDA_VISIBLE_DEVICES=0 python -m baselines.kernelmem.run_kernelmem \
  --bench kernelbench \
  --task kernelbench/level1/1_Square_matrix_multiplication_.py \
  --warmup 3 \
  --repeat 5 \
  --rounds 3
```

## Outputs

Runs are written to:

```text
baselines/kernelmem/runs/<timestamp>/
```

Each task/round stores:

```text
prompt.txt
llm_raw.txt
candidate_snippet.py
candidate_task.py
bench_result.json
stdout.txt
stderr.txt
result.json
```
