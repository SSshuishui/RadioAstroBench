from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any


class KernelIndividual:
    _next_id = 0

    def __init__(self, code: str):
        self.id: int = KernelIndividual._next_id
        KernelIndividual._next_id += 1
        self.code: str = code
        self.metrics: Optional[Dict[str, Any]] = None
        self.score: Optional[float] = None
        self.feedback: Optional[str] = None
        self.code_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "score": self.score, "code_path": str(self.code_path) if self.code_path else ""}

    def save_code(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"kernel_{self.id:04d}.py"
        path.write_text(self.code, encoding="utf-8")
        self.code_path = path
        return path

    def save_metrics(self, out_dir: Path) -> Path:
        if self.metrics is None:
            raise ValueError("metrics not set")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"eval_{self.id:04d}.json"
        path.write_text(json.dumps(self.metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
