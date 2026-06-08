from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunConfig:
    tasks: str
    out_dir: Path
    rounds: int = 3
    arch: str = "8.9"
    model: str = "deepseek-coder"
    temperature: float = 0.2
    timeout_s: int = 180
    num_correct_trials: int = 5
    num_perf_trials: int = 50
    warmup: int = 3
    evaluator: str = "auto"
    precision: str = "float32"
    no_llm: bool = False
    seed_reference: bool = False
    verbose: bool = False

    def validate(self) -> None:
        if self.arch != "8.9":
            raise ValueError("RadioForge intentionally supports only RTX 4090 / sm_89. Use --arch 8.9.")
        if self.rounds < 1:
            raise ValueError("--rounds must be >= 1")
        self.out_dir.mkdir(parents=True, exist_ok=True)
