# Radio astronomy pattern hints

Common useful patterns:

- Complex tensors: preserve real/imag semantics carefully. `torch.view_as_real` layout is last dimension of size 2.
- Gridding/scatter-like updates: atomics may be needed; correctness can be sensitive to accumulation order.
- Reductions over channels/frequency/time: map one row or output element per block, use shared memory or warp reductions.
- Small fixed dimensions: specialize branches based on shape in Python and call separate kernels.
- Broadcasting: explicitly compute strides or require contiguous inputs and call `.contiguous()` before custom kernels.
