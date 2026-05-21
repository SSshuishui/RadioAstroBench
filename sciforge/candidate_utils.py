from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional


def strip_code_fences(text: str) -> str:
    text = text.strip()
    full = re.match(r"^```(?:python|py)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if full:
        return full.group(1).strip()
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return max(blocks, key=len).strip()
    return text


def validate_snippet(snippet: str) -> None:
    if "class ModelNew" not in snippet:
        raise ValueError("candidate must define class ModelNew")
    forbidden = ["def get_inputs", "def get_init_inputs", "TASK_ID =", "SUPPORTED_SCALES =", "class Model("]
    hits = [x for x in forbidden if x in snippet]
    if hits:
        raise ValueError(f"candidate changes protected task objects: {hits}")


def make_candidate_source(original_source: str, snippet: str, comment: Optional[str] = None) -> str:
    snippet = strip_code_fences(snippet)
    validate_snippet(snippet)
    return (
        original_source.rstrip()
        + "\n\n# ============================================================================\n"
        + f"# {comment or 'Our method candidate'}\n"
        + "# Appended by our_method. Baseline Model and input generator are unchanged.\n"
        + "# ============================================================================\n\n"
        + snippet.strip()
        + "\n"
    )


def task_slug(path: str | Path) -> str:
    raw = str(Path(path).with_suffix(""))
    raw = raw.replace("/", "__").replace("\\", "__")
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", raw).strip("_")


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
