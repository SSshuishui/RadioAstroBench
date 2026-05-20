# Radio Astronomy Agent Bench

This repository contains the radio astronomy CUDA benchmark and agent-baseline adapters.

## Top-level layout

```text
radio_bench/                    # Radio Astronomy CUDA Bench tasks
radio_astronomy_cuda_bench/     # shared harness, configs, original CUDA snapshots
kernelbench/                    # KernelBench tasks copied from CudaForge package
baselines/
  llm_direct/                   # LLM-only direct candidate baseline; radio_bench only
  cudaforge/                    # CudaForge-style baseline; supports radio_bench and KernelBench
  kernelmem/                    # reserved for KernelMem adapter
our_method/                     # reserved for our proposed method; not inside baselines
```

Naming policy for future packages:

```text
radio_astronomy_agent_v1_llm.zip
radio_astronomy_agent_v2_cudaforge.zip
radio_astronomy_agent_v3_kernelmem.zip
...
```

## Bench task naming

Inside each level, task numbers are sequential. The suffix indicates the algorithm path:

- `ws`: short-time stacking / 2D local-frame reconstruction path.
- `3d`: 3D inverse reconstruction path.

Example:

```text
radio_bench/level2/03-ws-recon-representative.py
radio_bench/level2/06-3d-recon-direct.py
```

## Quick sanity checks

Radio benchmark smoke:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST="8.9"
bash run_all_smoke.sh
```

LLM-only baseline mock, radio_bench only:

```bash
bash baselines/llm_direct/run_mock.sh
```

CudaForge-style mock on radio_bench:

```bash
bash run_cudaforge_radio_mock.sh
```

CudaForge-style mock on KernelBench:

```bash
bash run_cudaforge_kernelbench_mock.sh
```

## LLM-only direct baseline

This baseline is intentionally simple and only supports `radio_bench`.

Run with a real OpenAI-compatible model:

```bash
export LLM_API_KEY="..."
export LLM_API_BASE="https://.../v1"
export LLM_MODEL="deepseek-v4-pro"

CUDA_VISIBLE_DEVICES=0 python -m baselines.llm_direct.run_llm_direct \
  --task radio_bench/level1/05-ws-build-nm1.py \
  --scale smoke \
  --warmup 3 \
  --repeat 5 \
  --max-iters 3
```

## CudaForge-style baseline

The adapter extracts the core CudaForge loop into this repo:

```text
Coder prompt -> candidate ModelNew -> compile/run/correctness/latency feedback
            -> repair prompt on failure
            -> optional judge plan after success
            -> next candidate
```

It supports two datasets:

```text
--dataset radio        # uses radio_astronomy_cuda_bench.run_smoke
--dataset kernelbench  # uses CudaForge's compare_and_bench evaluator under baselines/cudaforge/vendor
```

Run CudaForge-style baseline on radio_bench:

```bash
export LLM_API_KEY="..."
export LLM_API_BASE="https://.../v1"
export LLM_MODEL="deepseek-v4-pro"

CUDA_VISIBLE_DEVICES=0 python -m baselines.cudaforge.run_cudaforge \
  --dataset radio \
  --task radio_bench/level1/05-ws-build-nm1.py \
  --scale smoke \
  --warmup 3 \
  --repeat 5 \
  --max-iters 3 \
  --judge-after-success
```

Run CudaForge-style baseline on KernelBench:

```bash
CUDA_VISIBLE_DEVICES=0 python -m baselines.cudaforge.run_cudaforge \
  --dataset kernelbench \
  --task kernelbench/level1/1_Square_matrix_multiplication_.py \
  --warmup 3 \
  --repeat 5 \
  --max-iters 3 \
  --judge-after-success
```

Run a subset:

```bash
CUDA_VISIBLE_DEVICES=0 python -m baselines.cudaforge.run_cudaforge \
  --dataset radio \
  --level level1 \
  --first-n 3 \
  --scale smoke \
  --warmup 3 \
  --repeat 5 \
  --max-iters 2
```

## Outputs

LLM direct outputs:

```text
baselines/llm_direct/runs/<timestamp>/...
```

CudaForge-style outputs:

```text
baselines/cudaforge/runs/<timestamp>_<dataset>/...
```

Each attempt stores:

```text
coder_prompt.txt
llm_raw.txt
candidate_snippet.py
candidate_task.py
bench_result.json
stdout.txt
stderr.txt
result.json
```

## Current boundaries

- `llm_direct` only targets `radio_bench`.
- `cudaforge` supports both `radio_bench` and `kernelbench`.
- `kernelmem` is only a placeholder until its source is added.
- `our_method` is a placeholder outside `baselines/`, as requested.
- Optional NCU/hardware feedback is represented by `--profile-text-file`; full automatic NCU collection can be added later.
