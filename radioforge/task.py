from __future__ import annotations

import ast
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from .code_utils import read_text

TORCH_OP_RE = re.compile(r"torch\.([A-Za-z_][A-Za-z0-9_]*)")


@dataclass(slots=True)
class TaskSignature:
    path: str
    stem: str
    torch_ops: list[str]
    has_cuda_extension: bool
    has_triton: bool
    patterns: list[str]
    class_model: bool
    forward_args: list[str]
    source_excerpt: str

    def to_dict(self) -> dict:
        return asdict(self)


def _forward_args(tree: ast.AST) -> list[str]:
    args: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            args = [a.arg for a in node.args.args if a.arg != "self"]
            break
    return args


def analyze_task(path: Path) -> TaskSignature:
    src = read_text(path)
    ops = sorted(set(TORCH_OP_RE.findall(src)))
    lower = src.lower()
    patterns: list[str] = []
    checks = {
        "elementwise": ["+", "-", "*", "/", "torch.sin", "torch.cos", "torch.exp", "torch.sqrt", "torch.abs", "relu", "sigmoid", "silu"],
        "reduction": ["sum(", "mean(", "amax", "amin", "max(", "min(", "norm(", "cumsum", "argmax", "argmin"],
        "matmul": ["matmul", "@", "mm(", "bmm(", "einsum", "linear"],
        "conv": ["conv1d", "conv2d", "conv3d"],
        "scatter_gather": ["gather", "scatter", "index_select", "where", "nonzero"],
        "complex": ["complex", "real", "imag", "fft", "view_as_real", "view_as_complex"],
        "shape_transform": ["reshape", "view(", "permute", "transpose", "contiguous", "unsqueeze", "squeeze"],
        "masking": ["mask", "masked", "where", ">", "<"],
        "sort_topk": ["sort", "topk", "argsort"],
    }
    for name, needles in checks.items():
        if any(n in lower for n in needles):
            patterns.append(name)
    try:
        tree = ast.parse(src)
        class_model = any(isinstance(n, ast.ClassDef) and n.name == "Model" for n in tree.body)
        fargs = _forward_args(tree)
    except SyntaxError:
        class_model = "class Model" in src
        fargs = []
    excerpt = src[:6000]
    return TaskSignature(
        path=str(path),
        stem=path.stem,
        torch_ops=ops,
        has_cuda_extension="load_inline" in src or "cpp_extension" in src,
        has_triton="triton" in lower,
        patterns=patterns,
        class_model=class_model,
        forward_args=fargs,
        source_excerpt=excerpt,
    )
