from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .code_utils import sha256_text
from .task import TaskSignature


@dataclass(slots=True)
class MemoryItem:
    task: str
    signature_key: str
    patterns: list[str]
    code_hash: str
    correct: bool
    compiled: bool
    speedup: float | None
    runtime_ms: float | None
    ref_runtime_ms: float | None
    note: str
    code_path: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def signature_key(sig: TaskSignature) -> str:
    ops = ",".join(sig.torch_ops[:32])
    pats = ",".join(sig.patterns)
    return sha256_text(f"ops={ops}|patterns={pats}|args={sig.forward_args}")[:16]


class KernelMemory:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items: list[dict[str, Any]] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    self.items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    def retrieve(self, sig: TaskSignature, limit: int = 6) -> list[dict[str, Any]]:
        pats = set(sig.patterns)
        ops = set(sig.torch_ops)
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self.items:
            item_pats = set(item.get("patterns", []))
            score = len(pats & item_pats) * 2.0
            if item.get("signature_key") == signature_key(sig):
                score += 5.0
            if item.get("correct"):
                score += 1.0
            speed = item.get("speedup") or 0
            score += min(float(speed), 10.0) * 0.1
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:limit]]

    def append(self, item: MemoryItem) -> None:
        self.items.append(asdict(item))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(item.to_json() + "\n")
