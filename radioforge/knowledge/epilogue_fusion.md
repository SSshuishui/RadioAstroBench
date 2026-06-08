# Epilogue fusion hints

CODA's useful transferable idea is not the Hopper implementation, but the decomposition:
producer compute + epilogue visitors.

For radio_bench tasks, consider fusing:

- multiply/add/subtract after matmul or elementwise loops;
- clamp, relu, sigmoid/silu approximations only if exact tolerance is safe;
- mask application after computing a value;
- per-row or per-column scaling;
- simple sum/mean reductions with final scale;
- conversion between real/imag pairs if layout is simple.

Do not fuse if it changes broadcasting semantics, dtype promotion, or NaN behavior in a way that may fail correctness.
