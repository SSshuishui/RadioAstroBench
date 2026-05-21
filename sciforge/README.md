# Our Method: Domain-Aware Constrained CUDA Optimization Agent

This method is built on the KernelMem idea of memory-augmented candidate generation, but adds radio/scientific-kernel specific control:

1. **Kernel role analysis**: classifies tasks as geometry, visibility, reconstruction, task scheduling, baseline generation, weighting, or helper kernels.
2. **Constraint extraction**: explicitly protects HEALPix conventions, l/m/n geometry, uvw/baseline symmetry, blockage checks, adjoint/forward phase formulas, output shapes and dtypes.
3. **Skill routing**: selects optimization strategies based on the kernel role and data pattern.
4. **Long/short memory**: reuses KernelMem's memorybank and local run history.
5. **Multi-scale verification**: supports smoke -> nside512/nside4096 validation for `radio_bench`.
6. **Optional hardware feedback**: can call Nsight Compute (`ncu`) for `radio_bench` candidates.

## Mock run

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST="8.9"

CUDA_VISIBLE_DEVICES=0 python -m our_method.run_our_method \
  --bench radio \
  --task radio_bench/level1/05-ws-build-nm1.py \
  --scale smoke \
  --warmup 1 \
  --repeat 2 \
  --rounds 1 \
  --mock
```

## Real LLM run on radio_bench

```bash
export LLM_API_KEY="..."
export LLM_API_BASE="https://.../v1"
export LLM_MODEL="deepseek-v4-pro"
export PYTHONPATH=$PWD:$PYTHONPATH
export TORCH_CUDA_ARCH_LIST="8.9"

CUDA_VISIBLE_DEVICES=0 python -m our_method.run_our_method \
  --bench radio \
  --task radio_bench/level2/03-ws-recon-representative.py \
  --scale smoke \
  --validate-scales nside512_full \
  --warmup 3 \
  --repeat 5 \
  --rounds 4
```

## Real LLM run on KernelBench

```bash
CUDA_VISIBLE_DEVICES=0 python -m our_method.run_our_method \
  --bench kernelbench \
  --task kernelbench/level1/1_Square_matrix_multiplication_.py \
  --warmup 3 \
  --repeat 5 \
  --rounds 4
```

Outputs are written to:

```text
our_method/runs/<timestamp>/
```
