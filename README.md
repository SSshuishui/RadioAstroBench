# Radio Astronomy CUDA Bench

A KernelBench-style benchmark wrapper for existing CUDA kernels used in the lunar-orbit radio astronomy imaging pipeline.

`radio_bench/` contains only benchmark task files.  
`radio_astronomy_cuda_bench/` contains the harness, shared helpers, configs, and archived original CUDA source snapshots.

## Naming convention

Task files are numbered sequentially inside each level:

```text
01-ws-*.py, 02-ws-*.py, ...
15-3d-*.py, 16-3d-*.py, ...
```

The number is the task order in that level. The tag after the number indicates the algorithm path:

- `ws`: short-time stacking / 2D local-frame reconstruction path.
- `3d`: 3D inverse reconstruction path.

## Benchmark levels

- `level1`: helper kernels, geometry preprocessing, data movement, small reductions, index/key construction.
- `level2`: single-stage core kernels, including visibility, reconstruction, task-list construction, and DCF/pair-weight calculation.
- `level3`: day-level all-10-segment workloads, closer to the original day-1 execution structure.

## Layout

```text
radio_bench/
  level1/   # helper kernels and preprocessing micro-kernels
  level2/   # single-stage core kernels
  level3/   # day-level all-10-segment workloads

radio_astronomy_cuda_bench/
  common.py
  run_smoke.py
  run_bench.py
  configs/
  kernels/original/ws/
  kernels/original/3d/
```

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST="8.9"

bash run_recon_smoke.sh
```

Run all smoke tasks:

```bash
bash run_all_smoke.sh
```

Equivalent explicit command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m radio_astronomy_cuda_bench.run_smoke \
  --task all \
  --scale smoke \
  --warmup 3 \
  --repeat 5
```

Run one WS task:

```bash
CUDA_VISIBLE_DEVICES=0 python -m radio_astronomy_cuda_bench.run_smoke \
  --task radio_bench/level2/03-ws-recon-representative.py \
  --scale smoke \
  --warmup 10 \
  --repeat 50
```

Run one 3D task:

```bash
CUDA_VISIBLE_DEVICES=0 python -m radio_astronomy_cuda_bench.run_smoke \
  --task radio_bench/level2/08-3d-pair-weight.py \
  --scale smoke \
  --warmup 3 \
  --repeat 10
```

Run a day-level all-10-segment task:

```bash
CUDA_VISIBLE_DEVICES=0 python -m radio_astronomy_cuda_bench.run_bench \
  --task radio_bench/level3/02-ws-day-recon-representative.py \
  --scale nside512_full \
  --segment-profile all10 \
  --warmup 1 \
  --repeat 3
```

## Task list

### level1: helper kernels and preprocessing micro-kernels

- `01-ws-pix2ang-nest.py`: legacy/reference NEST pixel-to-angle helper, returning `theta_heal/phi_heal`.
- `02-ws-pix2ang-ring.py`: legacy/reference RING pixel-to-angle helper, returning `theta_heal/phi_heal`.
- `03-ws-pix2lmn-nest.py`: optimized NEST pixel-to-direction helper, directly returning `l/m/n` and skipping intermediate `theta/phi`.
- `04-ws-pix2lmn-ring.py  # user-provided direct RING l/m/n kernel`: optimized RING pixel-to-direction helper, directly returning `l/m/n` and skipping intermediate `theta/phi`.
- `05-ws-build-nm1.py`: `nm1 = n - 1` helper.
- `06-ws-normalize-by-weight.py`: normalize reconstruction image by integer weight.
- `07-ws-generate-baselines-signed56.py`: online 8-node baseline generation, 28 unique + 28 signed counterpart records.
- `08-ws-iota.py`: integer index initialization.
- `09-ws-build-locg-half.py`: map half-baselines onto local reconstruction-grid keys.
- `10-ws-gather-viss-by-idx.py`: reorder visibility values by sorted/grouped indices.
- `11-ws-compute-avg.py`: average grouped complex visibility sums by counts.
- `12-ws-add-inplace-f32.py`: float32 vector accumulation helper.
- `13-ws-add-inplace-u32.py`: uint32 vector accumulation helper.
- `14-ws-scale-inplace-f32.py`: float32 vector scaling helper.
- `15-3d-compute-sat1-xyz.py`: generate a chief-satellite trajectory sample.
- `16-3d-diff-sat1.py`: adjacent-position difference for the chief satellite.
- `17-3d-norm3.py`: 3D vector norm.
- `18-3d-invnorm3.py`: inverse 3D vector norm.
- `19-3d-normalize3-inplace.py`: normalize 3D vectors.
- `20-3d-compute-other-sats.py`: generate seven additional node positions from a chief trajectory.
- `21-3d-gather-pos-range.py`: gather a time-range slice from node positions.
- `22-3d-compute-uvw-xyz.py`: generate 56 signed baselines and endpoint records.
- `23-3d-compute-uvw-xyz-half.py`: generate 28 half-baselines and endpoint records.
- `24-3d-healpix-lmn-from-theta-phi.py`: convert theta/phi direction angles to l/m/n.

### level2: single-stage core kernels

- `01-ws-tile-meta.py`: tile-level cone metadata for visibility fast paths.
- `02-ws-visibility-forward.py`: half-baseline visibility simulation with tile cone classification.
- `03-ws-recon-representative.py`: representative-group reconstruction with blockage check.
- `04-ws-recon-grid-average.py`: grouped half-grid reconstruction without representative blockage.
- `05-3d-visibility-forward.py`: 3D half-symmetry visibility forward kernel.
- `06-3d-recon-direct.py`: 3D direct inverse accumulation kernel.
- `07-3d-phase-reduce.py`: reduce partial visibility parts and apply half-symmetry phase correction.
- `08-3d-pair-weight.py`: compute half-baseline pair weights using the DCF table.
- `09-3d-task-flags.py`: build all-visible and mixed task flags for 3D reconstruction.
- `10-3d-recon-tasklist-vv.py`: 3D direct reconstruction for all-visible task-list entries.
- `11-3d-recon-tasklist-mixed.py`: 3D direct reconstruction for mixed task-list entries.

### level3: day-level all-10-segment workloads

- `01-ws-day-visibility-forward.py`: all-10-segment day-level WS visibility task.
- `02-ws-day-recon-representative.py`: all-10-segment day-level WS representative reconstruction task.
- `03-3d-day-visibility-forward.py`: all-10-segment day-level 3D visibility task.
- `04-3d-day-recon-direct.py`: all-10-segment day-level 3D direct reconstruction task.

## HEALPix pixel order notes

The original MATLAB-style path was:

```text
f_pix2ang_nest/ring(nside, ipix) -> theta_heal, phi_heal
theta_lat = pi/2 - theta_heal
phi = -wrap_to_pi(phi_heal)
n = sin(theta_lat)
l = cos(theta_lat) * cos(phi)
m = cos(theta_lat) * sin(phi)
```

The optimized CUDA tasks `03-ws-pix2lmn-nest.py` and `04-ws-pix2lmn-ring.py  # user-provided direct RING l/m/n kernel` fuse this chain and directly generate `l/m/n` online. The `pix2ang-*` tasks are kept as reference-style microbenchmarks and for compatibility checks.

## Notes

1. `Model` is the existing CUDA baseline; `ModelNew` is currently an identity candidate.
2. Torch is used for tensor management, inline CUDA extension compilation, and timing only.
3. Most tasks currently use deterministic synthetic fixtures. Later, real day-1 segment fixtures can replace these inputs.
4. `run_smoke.py` supports `--scale`, `--segment-profile`, and `--fixture`.
5. Original CUDA source snapshots are archived under:
   - `radio_astronomy_cuda_bench/kernels/original/ws/`
   - `radio_astronomy_cuda_bench/kernels/original/3d/`

## Baselines

All optimization methods are placed under `baselines/` so that future methods can share the same benchmark interface.

```text
baselines/
  llm_direct/     # LLM-only direct optimization baseline, implemented now
  cudaforge/      # placeholder for future CudaForge adaptation
  kernelmem/      # placeholder for future KernelMem adaptation
  ours/           # placeholder for the proposed method
```

### LLM-only direct baseline

This baseline reads one `radio_bench` task, asks an OpenAI-compatible chat model to append a new `ModelNew`, runs `run_smoke.py`, and optionally repairs the candidate using compile/runtime/correctness feedback.

Environment variables for real model calls:

```bash
export LLM_API_KEY="..."
export LLM_API_BASE="https://.../v1"
export LLM_MODEL="..."
```

Run one task:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST="8.9"

CUDA_VISIBLE_DEVICES=0 python -m baselines.llm_direct.run_llm_direct \
  --task radio_bench/level1/05-ws-build-nm1.py \
  --scale smoke \
  --warmup 3 \
  --repeat 5 \
  --max-iters 3
```

Harness-only mock test, without calling a model:

```bash
CUDA_VISIBLE_DEVICES=0 python -m baselines.llm_direct.run_llm_direct \
  --task radio_bench/level1/05-ws-build-nm1.py \
  --scale smoke \
  --warmup 1 \
  --repeat 2 \
  --max-iters 1 \
  --mock
```

Outputs are written to:

```text
baselines/llm_direct/runs/<timestamp>/
```

Each attempt stores:

```text
llm_raw.txt
candidate_snippet.py
candidate_task.py
bench_result.json
stdout.txt
stderr.txt
result.json
```
