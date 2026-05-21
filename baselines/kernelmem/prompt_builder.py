from __future__ import annotations

from textwrap import dedent

SYSTEM_PROMPT = dedent(
    """
    You are KernelMem-style CUDA optimization agent. Optimize an existing benchmark task
    by appending a candidate ModelNew implementation.

    Required output:
    - Return Python code only, no Markdown fences.
    - The code must define class ModelNew(nn.Module) or class ModelNew(Model).
    - You may define CPP_SRC_NEW, CUDA_SRC_NEW, get_candidate_extension, helper kernels, and helper classes.
    - The returned code will be appended to the original task file.

    Hard constraints:
    - Do not change baseline Model, get_inputs, get_init_inputs, TASK_ID, or SUPPORTED_SCALES.
    - Preserve output shapes, dtypes, and numerical semantics.
    - Do not remove physics/geometric checks such as visibility, blockage, l/m/n, uvw, or half-symmetry logic.
    - Prefer safe incremental optimizations first; repair correctness before optimizing speed.

    KernelMem behavior:
    - Use long-term memory to select optimization strategies.
    - Use short-term error/performance feedback to avoid repeated failures.
    - When generating CUDA, include full wrapper code needed by load_inline.
    """
).strip()


def build_generation_prompt(
    *,
    bench: str,
    task_path: str,
    task_source: str,
    task_structure: str,
    long_memory: str,
    short_memory: str,
    scale: str,
    warmup: int,
    repeat: int,
) -> str:
    return dedent(
        f"""
        Generate a KernelMem-style optimized candidate for this benchmark task.

        Benchmark type: {bench}
        Task path: {task_path}
        Evaluation scale: {scale}
        Timing: warmup={warmup}, repeat={repeat}

        Long-term optimization memory:
        <<<LONG_MEMORY_BEGIN>>>
        {long_memory}
        <<<LONG_MEMORY_END>>>

        Short-term local run memory:
        <<<SHORT_MEMORY_BEGIN>>>
        {short_memory}
        <<<SHORT_MEMORY_END>>>

        Task structure summary:
        <<<TASK_STRUCTURE_BEGIN>>>
        {task_structure}
        <<<TASK_STRUCTURE_END>>>

        Full task source:
        <<<TASK_SOURCE_BEGIN>>>
        {task_source}
        <<<TASK_SOURCE_END>>>

        Return only the appended Python candidate snippet defining ModelNew.
        """
    ).strip()


def build_repair_prompt(
    *,
    bench: str,
    task_path: str,
    task_source: str,
    previous_snippet: str,
    failure_summary: str,
    long_memory: str,
    short_memory: str,
    scale: str,
    warmup: int,
    repeat: int,
) -> str:
    return dedent(
        f"""
        Repair the previous KernelMem candidate.

        Benchmark type: {bench}
        Task path: {task_path}
        Evaluation scale: {scale}
        Timing: warmup={warmup}, repeat={repeat}

        Failure / benchmark feedback:
        <<<FAILURE_BEGIN>>>
        {failure_summary[-12000:]}
        <<<FAILURE_END>>>

        Previous candidate snippet:
        <<<PREVIOUS_SNIPPET_BEGIN>>>
        {previous_snippet}
        <<<PREVIOUS_SNIPPET_END>>>

        Long-term optimization memory:
        <<<LONG_MEMORY_BEGIN>>>
        {long_memory}
        <<<LONG_MEMORY_END>>>

        Short-term local run memory:
        <<<SHORT_MEMORY_BEGIN>>>
        {short_memory}
        <<<SHORT_MEMORY_END>>>

        Original task source:
        <<<TASK_SOURCE_BEGIN>>>
        {task_source}
        <<<TASK_SOURCE_END>>>

        Return a complete replacement appended Python snippet defining ModelNew.
        Return code only, no Markdown.
        """
    ).strip()
