from __future__ import annotations

import hashlib
import re
from pathlib import Path


def strip_code_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:python|py)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return max(blocks, key=len).strip()
    return text


def validate_snippet(snippet: str) -> None:
    if "class ModelNew" not in snippet:
        raise ValueError("Candidate must define class ModelNew")
    forbidden = ["def get_inputs", "def get_init_inputs", "TASK_ID =", "SUPPORTED_SCALES =", "class Model("]
    hits = [x for x in forbidden if x in snippet]
    if hits:
        raise ValueError(f"Candidate changes forbidden benchmark objects: {hits}")


def make_candidate_source(original_source: str, snippet: str, *, label: str) -> str:
    snippet = strip_code_fences(snippet)
    validate_snippet(snippet)
    return (
        original_source.rstrip()
        + "\n\n# ============================================================================\n"
        + f"# CudaForge-style candidate: {label}\n"
        + "# Appended candidate code below; baseline Model and benchmark inputs remain unchanged.\n"
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
