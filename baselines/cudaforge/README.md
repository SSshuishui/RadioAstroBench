# CudaForge baseline adapter

This directory implements the CudaForge workflow inside the `radio_astronomy_agent` repository structure.

It keeps the CudaForge functionality in this framework rather than nesting the original repository:

- seed kernel generation
- correctness diagnosis
- repair loop
- Nsight Compute profiling
- optimization judge
- optimization prompt
- best-candidate tracking
- per-round metrics and score curve
- global `summary.json` / `summary.csv`

Datasets are selected by `--dataset`:

- `radio`: tasks under outer `radio_bench/`
- `kernelbench`: tasks under outer `kernelbench/`

The `kernelbench/` directory is intentionally at the repository root, not inside this baseline directory.

## Reproduction scope

This directory contains the CudaForge baseline adapted to `radio_bench`. It keeps the CudaForge seed / repair / Nsight Compute profile / judge / optimization loop, prompt structure, and KernelBench-compatible path, but it is not a byte-for-byte copy of upstream CudaForge `main.py`: the evaluator and candidate wrapper are adapted for `radio_bench`, real fixture profiles, and this repository's task format.

`--max-iters` is accepted as an alias for `--round` for the radio smoke/real scripts.
