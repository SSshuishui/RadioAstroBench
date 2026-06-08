# RTX 4090 CUDA rules

- Target Ada Lovelace SM89: `TORCH_CUDA_ARCH_LIST=8.9` and NVCC `-gencode=arch=compute_89,code=sm_89`.
- Prefer block sizes 128/256 threads for elementwise and simple reductions.
- Use contiguous fast paths but keep fallback for non-contiguous tensors if reference may pass them.
- For fp32 reductions, accumulate in float. For fp16/bf16, use float accumulation unless tolerance permits otherwise.
- Avoid Hopper-only WGMMA, TMA, CuTeDSL assumptions.
- Avoid excessive template complexity; KernelBench-style tasks reward compile reliability.
