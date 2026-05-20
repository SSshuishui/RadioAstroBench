from __future__ import annotations

from textwrap import dedent

SYSTEM_PROMPT = dedent(
    """
    You are a senior CUDA performance engineer in a CudaForge-style optimization loop.
    You optimize kernels iteratively using compile results, correctness feedback, latency feedback,
    and optional hardware/profile hints.

    Hard constraints:
    - Preserve public Python API, input/output shapes, dtypes, and semantics.
    - Do not modify get_inputs, get_init_inputs, TASK_ID, or benchmark configuration.
    - For radio_bench tasks, do not remove physics/geometry logic such as occultation checks,
      half-symmetry, l/m/n conventions, uvw phase terms, or representative-group semantics.
    - Return only Python code for an appended candidate snippet defining class ModelNew.
    - The snippet may define new CUDA/CPP strings, a new load_inline extension, and helper functions.
    - Do not include prose, markdown fences, tests, main functions, or unrelated files.
    """
).strip()

JUDGE_SYSTEM_PROMPT = dedent(
    """
    You are a CUDA optimization judge. Inspect a benchmark task, a candidate result, and optional
    profiling hints. Return a compact optimization plan. Do not generate code unless explicitly asked.
    """
).strip()


def build_coder_prompt(
    *,
    dataset: str,
    task_path: str,
    task_source: str,
    scale: str,
    warmup: int,
    repeat: int,
    history_block: str = "",
    judge_plan: str = "",
) -> str:
    return dedent(
        f"""
        Optimize this {dataset} CUDA benchmark task by appending a candidate implementation.

        Dataset: {dataset}
        Task path: {task_path}
        Evaluation scale: {scale}
        Timing: warmup={warmup}, repeat={repeat}

        The evaluator will append your snippet to the original task and compare baseline Model
        against candidate ModelNew.

        CudaForge-style strategy:
        1. Preserve semantics and public interface.
        2. Make one or a few safe CUDA performance improvements.
        3. Prefer changes such as block size tuning, shared-memory staging, vectorized loads,
           loop unrolling, fast math where safe, reduced redundant loads, branch separation,
           memory coalescing, or using existing symmetry more efficiently.
        4. Avoid deleting correctness-critical conditions.

        Previous attempts / history:
        <<<HISTORY_BEGIN>>>
        {history_block or '(none)'}
        <<<HISTORY_END>>>

        Judge optimization plan:
        <<<JUDGE_PLAN_BEGIN>>>
        {judge_plan or '(none)'}
        <<<JUDGE_PLAN_END>>>

        Return only Python code. The code must define class ModelNew.

        Original task file:
        <<<TASK_SOURCE_BEGIN>>>
        {task_source}
        <<<TASK_SOURCE_END>>>
        """
    ).strip()


def build_repair_prompt(
    *,
    dataset: str,
    task_path: str,
    task_source: str,
    previous_snippet: str,
    failure_text: str,
    scale: str,
    warmup: int,
    repeat: int,
) -> str:
    return dedent(
        f"""
        The previous candidate failed. Repair it using the error log.

        Dataset: {dataset}
        Task path: {task_path}
        Evaluation scale: {scale}
        Timing: warmup={warmup}, repeat={repeat}

        Previous candidate snippet:
        <<<PREVIOUS_SNIPPET_BEGIN>>>
        {previous_snippet}
        <<<PREVIOUS_SNIPPET_END>>>

        Failure log:
        <<<FAILURE_BEGIN>>>
        {failure_text[-12000:]}
        <<<FAILURE_END>>>

        Original task file:
        <<<TASK_SOURCE_BEGIN>>>
        {task_source}
        <<<TASK_SOURCE_END>>>

        Return only a corrected Python candidate snippet defining class ModelNew.
        """
    ).strip()


def build_judge_prompt(
    *,
    dataset: str,
    task_path: str,
    task_source: str,
    result_json: str,
    stdout_tail: str = "",
    stderr_tail: str = "",
    ncu_text: str = "",
) -> str:
    return dedent(
        f"""
        Analyze this CudaForge-style optimization state and produce the next optimization plan.

        Dataset: {dataset}
        Task path: {task_path}

        Benchmark/result JSON:
        <<<RESULT_JSON_BEGIN>>>
        {result_json[-12000:]}
        <<<RESULT_JSON_END>>>

        STDOUT tail:
        <<<STDOUT_BEGIN>>>
        {stdout_tail[-4000:]}
        <<<STDOUT_END>>>

        STDERR tail:
        <<<STDERR_BEGIN>>>
        {stderr_tail[-4000:]}
        <<<STDERR_END>>>

        Optional NCU / profile hints:
        <<<PROFILE_BEGIN>>>
        {ncu_text[-8000:] if ncu_text else '(none)'}
        <<<PROFILE_END>>>

        Original task source:
        <<<TASK_SOURCE_BEGIN>>>
        {task_source[:16000]}
        <<<TASK_SOURCE_END>>>

        Return a concise JSON object with keys:
        - bottleneck
        - optimization_method
        - modification_plan
        - risk
        """
    ).strip()
