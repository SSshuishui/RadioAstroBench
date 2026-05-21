# Radio Astronomy Agent v2 — CudaForge baseline

This package keeps the project-level benchmark layout and implements a CudaForge-style baseline inside `baselines/cudaforge/`.

## Layout

```text
radio_bench/                      # our radio astronomy CUDA benchmark
radio_astronomy_cuda_bench/        # radio_bench harness and shared utilities
kernelbench/                      # outer KernelBench dataset, shared by cudaforge/kernelmem

baselines/
  llm_direct/                     # simple LLM-only baseline for radio_bench
  cudaforge/                      # CudaForge workflow implemented in this framework
  kernelmem/                      # placeholder for next baseline

our_method/                       # our future method, not under baselines/
```

`kernelbench/` is intentionally at the same level as `baselines/`, not nested under `baselines/cudaforge/`.

## What the CudaForge baseline does

`baselines/cudaforge/run_cudaforge.py` implements the CudaForge workflow in this repository structure:

1. seed candidate generation
2. compile/correctness/latency evaluation
3. repair loop after compilation/runtime/correctness failure
4. Nsight Compute profiling after runnable candidates
5. optimization judge prompt from NCU metrics
6. optimization prompt and next candidate generation
7. best-candidate tracking
8. score curve, per-round metrics, `summary.json`, and `summary.csv`

It supports:

```text
--dataset radio        -> tasks under radio_bench/
--dataset kernelbench  -> tasks under kernelbench/
```

## Quick checks

Run radio_bench CudaForge mock without LLM and without NCU:

```bash
bash run_cudaforge_radio_mock.sh
```

Run KernelBench CudaForge mock:

```bash
bash run_cudaforge_kernelbench_mock.sh
```

## Real LLM run

For DeepSeek-compatible usage:

```bash
export DEEPSEEK_API_KEY="..."
export PYTHONPATH=$PWD:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST="8.9"

CUDA_VISIBLE_DEVICES=0 python -m baselines.cudaforge.run_cudaforge \
  --dataset radio \
  --task radio_bench/level1/05-ws-build-nm1.py \
  --scale smoke \
  --gpu "RTX 4090" \
  --server_type deepseek \
  --model_name deepseek-coder \
  --round 3 \
  --warmup 3 \
  --repeat 5
```

KernelBench:

```bash
CUDA_VISIBLE_DEVICES=0 python -m baselines.cudaforge.run_cudaforge \
  --dataset kernelbench \
  --task kernelbench/level1/1_Square_matrix_multiplication_.py \
  --gpu "RTX 4090" \
  --server_type deepseek \
  --model_name deepseek-coder \
  --round 3 \
  --warmup 3 \
  --repeat 5
```

If `ncu` is not configured on the machine, add `--no-ncu` for debugging. Formal CudaForge comparisons should run with NCU enabled.
