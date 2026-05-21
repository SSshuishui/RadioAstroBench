from __future__ import annotations
import re
from pathlib import Path
from .io_utils import extract_code_block


def normalize_candidate(raw: str) -> str:
    code = extract_code_block(raw)
    if "class ModelNew" not in code:
        raise ValueError("LLM output must define class ModelNew")
    return code.strip() + "\n"


def make_candidate_source(original_source: str, candidate_code: str, comment: str = "CudaForge candidate") -> str:
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
