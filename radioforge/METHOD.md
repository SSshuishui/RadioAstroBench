# RadioForge-SEMS: Skill-conditioned Epilogue-Memory Search

## Motivation

The two existing baselines are useful but emphasize different aspects:

- `cudaforge`: good at iterative CUDA generation/evaluation.
- `kernelmem`: good at reusing prior successful/failure cases.

The uploaded projects suggest another direction:

- CODA: represent expensive transformer-style operations as GEMM plus fused epilogue.
- KDA: use a staged agent workflow and record evidence after each iteration.
- TileLang/CUDA skills: provide structured low-level optimization rules.
- AKO4ALL: end-to-end automated optimize/benchmark/promote loop.

For `radio_bench`, many tasks are small scientific kernels with shape-specific math, reductions, masking, complex-valued operations, gridding-like loops, or elementwise chains. The main opportunity is often **not** a new exotic kernel language; it is generating a simple correct CUDA/Triton solution while fusing adjacent epilogues and avoiding unnecessary intermediate tensors.

## Algorithm

For each task `T`:

1. Parse source and construct an operator signature `S`:
   - class/function names;
   - torch ops;
   - detected patterns: elementwise, reduction, matmul, conv, scatter/gather, complex, FFT, sort/topk, mask, shape transforms.
2. Retrieve memory items whose signatures overlap with `S`:
   - successful candidates;
   - compile/runtime failures;
   - reusable prompt hints.
3. Generate a candidate with a constrained prompt:
   - output a single Python solution file with `class Model(nn.Module)`;
   - use CUDA extension/Triton where beneficial;
   - target RTX 4090 only (`sm_89`);
   - fuse epilogue operations when safe;
   - preserve exact API and numerical tolerance.
4. Evaluate candidate.
5. Update memory with structured result.
6. Repair next candidate using the failure trace, or mutate the best correct candidate if available.
7. Promote the fastest correct candidate.

## Why this is our method

The key design is **signature-level memory plus epilogue-aware generation**. KernelMem-style memory normally remembers whole tasks/candidates. RadioForge-SEMS remembers both whole candidates and operator signatures, so a reduction+elementwise pattern from one radio task can influence a different task. CODA inspires the prompt to treat trailing normalization, activation, scaling, and reductions as epilogues to fuse into the producer kernel.

## 4090 specialization

This implementation intentionally supports only NVIDIA RTX 4090:

- `TORCH_CUDA_ARCH_LIST=8.9`;
- NVCC `-gencode=arch=compute_89,code=sm_89`;
- avoids Hopper-only WGMMA/CuTeDSL requirements;
- recommends CUDA C++ and Triton over CuTeDSL for portability on Ada.
