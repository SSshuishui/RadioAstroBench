from __future__ import annotations

import difflib
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


def _extract_cuda_source_blocks(source: str) -> list[str]:
    blocks: list[str] = []
    for pat in [
        r"""(?:CUDA_SRC(?:_[A-Za-z0-9_]+)?|cuda_src)\s*=\s*r?\"\"\"(.*?)\"\"\"""",
        r"""(?:CUDA_SRC(?:_[A-Za-z0-9_]+)?|cuda_src)\s*=\s*r?\'\'\'(.*?)\'\'\'""",
    ]:
        blocks.extend(re.findall(pat, source, flags=re.DOTALL))
    return blocks


def _normalize_cuda_for_clone_check(src: str) -> str:
    src = re.sub(r"//.*?$", "", src, flags=re.MULTILINE)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"\b(opt|optimized|candidate|new|fast)\b", "", src, flags=re.IGNORECASE)
    src = re.sub(r"(_opt|_optimized|_candidate|_new|_fast)\b", "", src, flags=re.IGNORECASE)
    src = re.sub(r"\s+", "", src)
    return src.lower()


def _max_cuda_clone_ratio(snippet: str, original_source: str) -> float:
    cand_blocks = [b for b in _extract_cuda_source_blocks(snippet) if len(b) >= 800]
    orig_blocks = [b for b in _extract_cuda_source_blocks(original_source) if len(b) >= 800]
    best = 0.0
    for cb in cand_blocks:
        nc = _normalize_cuda_for_clone_check(cb)
        for ob in orig_blocks:
            no = _normalize_cuda_for_clone_check(ob)
            if nc and no:
                best = max(best, difflib.SequenceMatcher(None, nc, no).ratio())
    return best


def validate_snippet(snippet: str, original_source: str | None = None) -> None:
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
    banned = [
        r"\bget_extension\s*\(",
        r"\bclass\s+ModelNew\s*\(\s*Model\s*\)",
        r"\bModel\s*\(",
        r"\bsuper\s*\(\s*\)\.forward\s*\(",
    ]
    for pat in banned:
        if re.search(pat, snippet):
            raise ValueError(f"Candidate delegates to baseline or wraps identity pattern: {pat}")
    if original_source is not None:
        for fn in sorted(set(re.findall(r"\bext\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", original_source))):
            if re.search(rf"\bext\.{re.escape(fn)}\s*\(", snippet):
                raise ValueError(f"Candidate calls original extension function: ext.{fn}")
        clone_ratio = _max_cuda_clone_ratio(snippet, original_source)
        if clone_ratio >= 0.92:
            raise ValueError(f"Candidate appears to be a renamed copy of baseline CUDA (cuda_clone_ratio={clone_ratio:.4f})")


def make_candidate_source(original_source: str, snippet: str, metadata_comment: Optional[str] = None) -> str:
    snippet = strip_code_fences(snippet)
    validate_snippet(snippet, original_source)
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
