Generate an optimized solution for this task.

Task signature:
{signature}

Relevant memory:
{memory}

Optimization knowledge:
{knowledge}

Reference source excerpt:
```python
{source}
```

Strategy requirements:
1. Inspect the reference semantics and preserve numerical behavior.
2. Look for CODA-style epilogue fusion: combine trailing elementwise ops, scaling, bias, masks, reductions, and reshapes into the producer kernel when safe.
3. Use signature memory to avoid mistakes seen before.
4. For RTX 4090, use `TORCH_CUDA_ARCH_LIST=8.9` and NVCC flags for `sm_89` if compiling CUDA inline.
5. Return only code.
