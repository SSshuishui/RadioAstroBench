# Radio Astronomy Agent v2 — CudaForge baseline

This package keeps the project-level benchmark layout and implements a CudaForge-style baseline inside `baselines/cudaforge/`.

## Layout

```text
radio_bench/                      # our radio astronomy CUDA benchmark
radio_astronomy_cuda_bench/        # radio_bench harness and shared utilities
kernelbench/                      # outer KernelBench dataset, shared by cudaforge/kernelmem

baselines/
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

## 2026-06-02 update: CudaForge radio_bench compatibility

The CudaForge baseline was updated so `--dataset radio` evaluates generated `ModelNew` candidates through `radio_astronomy_cuda_bench.run_bench` and reads the current `candidate_ms`, `candidate_speedup`, and `candidate_correctness` fields rather than the older `identity_*` fields. All CudaForge reproduction code is kept under `baselines/cudaforge/`.

Additional radio_bench options are now supported by CudaForge:

```bash
python -m baselines.cudaforge.run_cudaforge \
  --dataset radio \
  --task radio_bench/level2/03-ws-recon-representative.py \
  --scale nside512_full \
  --fixture-profile nside512_day1_10m_ring \
  --server_type deepseek \
  --model_name deepseek-v4-pro \
  --round 3
```

For real-data runs, set `RKB_RADIO_ASTRO_DATA_ROOT` to the directory containing fixture profiles such as `radio_astro_data/nside512_day1_10m_ring`.  Pass `--require-fixture` if you want tasks without mapped fixtures to be skipped rather than falling back to synthetic scale inputs.
