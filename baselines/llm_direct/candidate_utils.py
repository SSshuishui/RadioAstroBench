from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional


def strip_code_fences(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```(?:python|py)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    # If model includes prose before/after a fenced block, recover the largest Python-looking fence.
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return max(blocks, key=len).strip()
    return text


def validate_snippet(snippet: str) -> None:
    if "class ModelNew" not in snippet:
        raise ValueError("Candidate snippet must define class ModelNew")
    forbidden = [
        "def get_inputs",
        "TASK_ID =",
        "SUPPORTED_SCALES =",
        "class Model("
    ]
    hits = [x for x in forbidden if x in snippet]
    if hits:
        raise ValueError(f"Candidate snippet changes forbidden task-level objects: {hits}")


def make_candidate_source(original_source: str, snippet: str, metadata_comment: Optional[str] = None) -> str:
    snippet = strip_code_fences(snippet)
    validate_snippet(snippet)
    comment = metadata_comment or "LLM direct candidate"
    return (
        original_source.rstrip()
        + "\n\n# ============================================================================\n"
        + f"# {comment}\n"
        + "# The code below is appended by baselines/llm_direct. It intentionally\n"
        + "# redefines ModelNew without changing baseline Model or get_inputs.\n"
        + "# ============================================================================\n\n"
        + snippet.strip()
        + "\n"
    )


def task_slug(task_path: str | Path) -> str:
    p = Path(task_path)
    raw = str(p.with_suffix(""))
    raw = raw.replace("/", "__").replace("\\", "__")
    raw = re.sub(r"[^0-9A-Za-z_.-]+", "_", raw)
    return raw.strip("_")


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
