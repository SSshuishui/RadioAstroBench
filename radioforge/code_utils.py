from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def strip_markdown_fences(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip() + "\n"
    return text.strip() + "\n"


def extract_class_model_source(src: str) -> str | None:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    lines = src.splitlines()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            start = node.lineno - 1
            end = getattr(node, "end_lineno", None)
            if end is None:
                return None
            return "\n".join(lines[start:end]) + "\n"
    return None


def make_reference_copy_solution(ref_src: str) -> str:
    """Safe fallback: keep imports and class Model, drop benchmark-only helpers if possible."""
    try:
        tree = ast.parse(ref_src)
    except SyntaxError:
        return ref_src
    keep: list[str] = []
    lines = ref_src.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
            if isinstance(node, ast.ClassDef) and node.name != "Model":
                continue
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            keep.append("\n".join(lines[start:end]))
    out = "\n\n".join(keep).strip()
    if "class Model" not in out:
        out = ref_src
    return out + "\n"


def canonicalize_code_for_similarity(src: str) -> str:
    src = re.sub(r"#.*", "", src)
    src = re.sub(r"\s+", " ", src)
    src = re.sub(r"\b[a-zA-Z_]\w*\b", lambda m: m.group(0) if m.group(0) in {"class", "def", "return", "import", "from", "torch", "nn", "Model", "forward"} else "ID", src)
    return src.strip()


def too_similar(a: str, b: str, threshold: float = 0.96) -> bool:
    import difflib
    return difflib.SequenceMatcher(None, canonicalize_code_for_similarity(a), canonicalize_code_for_similarity(b)).ratio() >= threshold
