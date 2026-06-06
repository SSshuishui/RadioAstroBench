from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def read_text_safe(path: Path, max_chars: int = 20000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[:max_chars]


class MemoryBank:
    """Long/short memory manager inspired by KernelMem.

    Long-term memory is static optimization knowledge copied from KernelMem's
    memorybank/prompts.  Short-term memory is summarized from previous local runs.
    """

    def __init__(self, root: Path):
        self.root = root
        self.memorybank_dir = root / "memorybank"
        self.prompts_dir = root / "prompts"

    def long_term_block(self, max_chars: int = 24000) -> str:
        chunks: List[str] = []
        for p in sorted(self.memorybank_dir.glob("*")):
            if p.is_file():
                text = read_text_safe(p, max_chars=8000)
                if text:
                    chunks.append(f"## memorybank/{p.name}\n{text}")
        # Prompt files are included as optional policy memory, but capped tightly.
        for name in ["generate_custom_cuda_memory.py", "optimization_memory_latest.py", "error_memory.py"]:
            p = self.prompts_dir / name
            text = read_text_safe(p, max_chars=6000)
            if text:
                chunks.append(f"## prompts/{name}\n{text}")
        block = "\n\n".join(chunks)
        if len(block) > max_chars:
            block = block[:max_chars] + "\n...[truncated long-term memory]"
        return block

    def short_term_block(self, runs_dir: Path, bench: str, max_tasks: int = 8, max_chars: int = 12000) -> str:
        if not runs_dir.exists():
            return "No previous short-term memory."
        summaries: List[Dict] = []
        for p in sorted(runs_dir.glob("*/summary.json"), reverse=True):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(obj, list):
                summaries.extend(obj)
            elif isinstance(obj, dict):
                if isinstance(obj.get("tasks"), list):
                    summaries.extend(obj["tasks"])
                else:
                    summaries.append(obj)
            if len(summaries) >= max_tasks:
                break
        lines = []
        for s in summaries[:max_tasks]:
            task = s.get("task") or s.get("task_path") or "unknown"
            best = s.get("best")
            attempts = s.get("attempts", [])
            ok_attempts = [a for a in attempts if a.get("ok")]
            lines.append(f"- task={task}, best={bool(best)}, ok_attempts={len(ok_attempts)}, attempts={len(attempts)}")
            if best and isinstance(best, dict):
                bench_result = best.get("bench_result") or {}
                speed = best.get("score") or bench_result.get("candidate_speedup") or bench_result.get("identity_speedup")
                lines.append(f"  speedup={speed}")
        text = "\n".join(lines) if lines else "No useful previous summaries."
        return text[:max_chars]

    def task_structure_block(self, task_source: str, max_chars: int = 10000) -> str:
        # Compact task source to the sections the model usually needs.
        keep_lines = []
        for line in task_source.splitlines():
            if any(key in line for key in ["TASK_ID", "SUPPORTED_SCALES", "CPP_SRC", "CUDA_SRC", "class Model", "class ModelNew", "def get_inputs", "def get_init_inputs", "load_inline", "__global__", "template<"]):
                keep_lines.append(line)
        compact = "\n".join(keep_lines)
        if len(compact) < 1000:
            compact = task_source[:max_chars]
        return compact[:max_chars]
