from __future__ import annotations

import json
from textwrap import dedent
from typing import List
from .task_analyzer import TaskAnalysis

SYSTEM_PROMPT = dedent(
    """
    You are a domain-aware CUDA optimization agent for scientific and ML kernels.
    Your task is to append a candidate ModelNew implementation to an existing benchmark task.

    Output requirements:
    - Return Python code only. No Markdown fences.
    - The code must define class ModelNew(nn.Module) or class ModelNew(Model).
    - You may define CPP_SRC_NEW, CUDA_SRC_NEW, get_candidate_extension, helper kernels, and helper Python functions.
    - The returned snippet is appended to the original task file.

    Hard safety/correctness constraints:
    - Do not change Model, get_inputs, get_init_inputs, TASK_ID, or SUPPORTED_SCALES.
    - Preserve output shapes and dtypes.
    - Preserve physics/geometric formulas, visibility/blockage checks, half-baseline symmetry, and HEALPix conventions.
    - Prefer conservative correctness-preserving optimization. If unsure, produce a correct candidate first.

    Method behavior:
    - Use the provided role analysis, protected semantics, and skill routing.
    - Use long-term and short-term memory to avoid repeated failures.
    - For high-risk scientific kernels, optimize implementation details, not formulas.
    """
).strip()


def build_prompt(
    *,
    task_path: str,
    bench: str,
    source: str,
    analysis: TaskAnalysis,
    skills: List[str],
    long_memory: str,
    short_memory: str,
    scale: str,
    warmup: int,
    repeat: int,
    previous_snippet: str = "",
    feedback: str = "",
    ncu_feedback: str = "",
) -> str:
    analysis_json = json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False)
    skills_text = "\n".join(f"- {s}" for s in skills)
    mode = "repair_or_improve" if previous_snippet or feedback else "initial_generation"
    return dedent(
        f"""
        Mode: {mode}
        Benchmark: {bench}
        Task: {task_path}
        Radio scale if applicable: {scale}
        Timing: warmup={warmup}, repeat={repeat}

        Domain role analysis:
        <<<ANALYSIS_BEGIN>>>
        {analysis_json}
        <<<ANALYSIS_END>>>

        Routed optimization skills:
        <<<SKILLS_BEGIN>>>
        {skills_text}
        <<<SKILLS_END>>>

        Long-term memory:
        <<<LONG_MEMORY_BEGIN>>>
        {long_memory}
        <<<LONG_MEMORY_END>>>

        Short-term local memory:
        <<<SHORT_MEMORY_BEGIN>>>
        {short_memory}
        <<<SHORT_MEMORY_END>>>

        Previous snippet, if any:
        <<<PREVIOUS_SNIPPET_BEGIN>>>
        {previous_snippet[-12000:]}
        <<<PREVIOUS_SNIPPET_END>>>

        Feedback from compiler/runtime/correctness/performance:
        <<<FEEDBACK_BEGIN>>>
        {feedback[-12000:]}
        <<<FEEDBACK_END>>>

        Optional NCU/hardware feedback:
        <<<NCU_BEGIN>>>
        {ncu_feedback[-12000:]}
        <<<NCU_END>>>

        Full original task source:
        <<<SOURCE_BEGIN>>>
        {source}
        <<<SOURCE_END>>>

        Return a full appended candidate snippet defining ModelNew. Return code only.
        """
    ).strip()
