from __future__ import annotations

from textwrap import dedent

SYSTEM_PROMPT = dedent(
    """
    You are an expert CUDA performance engineer. Your job is to optimize one existing
    PyTorch CUDA extension benchmark task by adding a replacement ModelNew.

    Hard rules:
    - Do not change the public task input/output semantics.
    - Do not remove correctness-critical physics or geometry logic.
    - Do not alter get_inputs, TASK_ID, SUPPORTED_SCALES, or the baseline Model.
    - Return only Python code for an appended candidate snippet.
    - The snippet must define class ModelNew(nn.Module) or class ModelNew(Model).
    - The snippet may define CPP_SRC_NEW, CUDA_SRC_NEW, get_candidate_extension, and helper functions.
    - The snippet must be self-contained when appended to the original task file.
    - The snippet may reuse imports already available in the task file: torch, nn, load_inline, os.
    - The snippet must not use external packages other than torch/PyTorch extension machinery.
    - Prefer safe, incremental optimizations over formula changes.
    - If you are uncertain, create a candidate that preserves semantics exactly.
    """
).strip()


def build_initial_prompt(task_path: str, task_source: str, scale: str, warmup: int, repeat: int) -> str:
    return dedent(
        f"""
        Optimize this CUDA benchmark task by appending a candidate implementation.

        Task path: {task_path}
        Evaluation scale: {scale}
        Timing: warmup={warmup}, repeat={repeat}

        The evaluator will create candidate_task.py by appending your returned snippet
        to the original task source. Then it will run:

          python -m radio_astronomy_cuda_bench.run_smoke \\
            --task candidate_task.py \\
            --scale {scale} \\
            --warmup {warmup} \\
            --repeat {repeat}

        The benchmark compares baseline Model vs ModelNew for correctness and speed.

        Return only Python code. Do not use Markdown. Do not wrap in ``` fences.

        Original task file:
        <<<TASK_SOURCE_BEGIN>>>
        {task_source}
        <<<TASK_SOURCE_END>>>
        """
    ).strip()


def build_repair_prompt(
    task_path: str,
    task_source: str,
    previous_snippet: str,
    failure_summary: str,
    scale: str,
    warmup: int,
    repeat: int,
) -> str:
    return dedent(
        f"""
        The previous candidate failed. Repair it.

        Task path: {task_path}
        Evaluation scale: {scale}
        Timing: warmup={warmup}, repeat={repeat}

        Failure summary:
        <<<FAILURE_BEGIN>>>
        {failure_summary[-8000:]}
        <<<FAILURE_END>>>

        Previous appended snippet:
        <<<PREVIOUS_SNIPPET_BEGIN>>>
        {previous_snippet}
        <<<PREVIOUS_SNIPPET_END>>>

        Original task file:
        <<<TASK_SOURCE_BEGIN>>>
        {task_source}
        <<<TASK_SOURCE_END>>>

        Return a full replacement appended snippet. Return only Python code.
        Do not use Markdown. Do not wrap in ``` fences.
        """
    ).strip()
