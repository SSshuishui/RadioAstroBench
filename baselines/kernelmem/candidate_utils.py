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
        raise ValueError("Candidate snippet must define class ModelNew")
    # Keep the task harness stable. The candidate may define helper classes/functions,
    # but must not replace the input generator, metadata, or baseline model.
    forbidden_patterns = [
        r"\bdef\s+get_inputs\b",
        r"\bdef\s+get_init_inputs\b",
        r"\bTASK_ID\b\s*=",
        r"\bSUPPORTED_SCALES\b\s*=",
        r"\bclass\s+Model\s*\(",
    ]
    hits = [pat for pat in forbidden_patterns if re.search(pat, snippet)]
    if hits:
        raise ValueError(f"Candidate snippet changes forbidden task-level objects: {hits}")
    # Minimal RadioBench anti-delegation guard, matching the CudaForge baseline adaptation.
    delegation_patterns = [
        r"\bget_extension\s*\(",
        r"\bclass\s+ModelNew\s*\(\s*Model\s*\)",
        r"\bModel\s*\(",
        r"\bsuper\s*\(\s*\)\.forward\s*\(",
        r"\boriginal_model\b",
        r"\bbase_model\b",
        r"\bbaseline_model\b",
    ]
    hits = [pat for pat in delegation_patterns if re.search(pat, snippet)]
    if hits:
        raise ValueError(f"Candidate delegates to baseline instead of defining an optimized candidate: {hits}")


def make_candidate_source(original_source: str, snippet: str, metadata_comment: Optional[str] = None) -> str:
    snippet = strip_code_fences(snippet)
    validate_snippet(snippet)
    return (
        original_source.rstrip()
        + "\n\n# ============================================================================\n"
        + f"# {metadata_comment or 'KernelMem candidate'}\n"
        + "# Appended by baselines/kernelmem. Baseline Model and get_inputs stay unchanged.\n"
        + "# ============================================================================\n\n"
        + snippet.strip()
        + "\n"
    )


def task_slug(task_path: str | Path) -> str:
    raw = str(Path(task_path).with_suffix(""))
    raw = raw.replace("/", "__").replace("\\", "__")
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", raw).strip("_")


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
