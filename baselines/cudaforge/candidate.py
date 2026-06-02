from __future__ import annotations
import re
from pathlib import Path
from .io_utils import extract_code_block


def validate_candidate_snippet(snippet: str, original_source: str | None = None) -> None:
    if "class ModelNew" not in snippet:
        raise ValueError("LLM output must define class ModelNew")
    forbidden_patterns = [
        r"\bdef\s+get_inputs\b",
        r"\bclass\s+Model\s*\(",
        r"\bTASK_ID\b\s*=",
        r"\bSUPPORTED_SCALES\b\s*=",
    ]
    for pat in forbidden_patterns:
        if re.search(pat, snippet):
            raise ValueError(f"candidate modifies forbidden task object: {pat}")
    banned_delegation_patterns = [
        r"\bget_extension\s*\(",
        r"\bclass\s+ModelNew\s*\(\s*Model\s*\)",
        r"\bModel\s*\(",
        r"\bsuper\s*\(\s*\)\.forward\s*\(",
        r"\boriginal_model\b",
        r"\bbase_model\b",
        r"\bbaseline_model\b",
    ]
    for pat in banned_delegation_patterns:
        if re.search(pat, snippet):
            raise ValueError(f"degenerate candidate delegates to baseline or identity wrapper: {pat}")


def normalize_candidate(raw: str, original_source: str | None = None) -> str:
    code = extract_code_block(raw).strip()
    validate_candidate_snippet(code, original_source)
    return code + "\n"


def make_candidate_source(original_source: str, candidate_code: str, comment: str = "CudaForge candidate") -> str:
    validate_candidate_snippet(candidate_code, original_source)
    return (
        original_source.rstrip()
        + "\n\n# ============================================================================\n"
        + f"# {comment}\n"
        + "# Appended by baselines/cudaforge. Baseline Model/get_inputs remain unchanged.\n"
        + "# ============================================================================\n\n"
        + candidate_code.strip()
        + "\n"
    )


def last_n_lines(text: str, n: int = 150) -> str:
    lines = str(text).splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else str(text)


def build_history_block(code_dir: Path, keep_last: int = 10) -> str:
    files = sorted(list(code_dir.glob("*.py")), key=lambda p: p.stat().st_mtime)[-keep_last:]
    if not files:
        return "## Existing kernels\n(None yet)\n"
    out = []
    for i, p in enumerate(files, 1):
        out.append(f"### Kernel {i} · {p.name}\n```python\n{p.read_text(encoding='utf-8', errors='ignore')[-8000:]}\n```")
    return "## Existing kernels\n" + "\n\n".join(out)
