from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def read_text_safe(path: Path, max_chars: int = 12000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


class DomainMemory:
    def __init__(self, root: Path):
        self.root = root
        self.memory_dir = root / "memorybank"
        self.prompt_dir = root / "kernelmem_prompts"
        self.runs_dir = root / "runs"

    def long_term_memory(self, max_chars: int = 24000) -> str:
        chunks: List[str] = []
        for p in sorted(self.memory_dir.glob("*")):
            if p.is_file():
                text = read_text_safe(p, max_chars=8000)
                if text:
                    chunks.append(f"## memorybank/{p.name}\n{text}")
        for p in sorted((self.prompt_dir).glob("*.py"))[:6]:
            text = read_text_safe(p, max_chars=5000)
            if text:
                chunks.append(f"## kernelmem_prompts/{p.name}\n{text}")
        block = "\n\n".join(chunks)
        return block[:max_chars] + ("\n...[truncated]" if len(block) > max_chars else "")

    def short_term_memory(self, max_chars: int = 10000) -> str:
        if not self.runs_dir.exists():
            return "No previous short-term memory."
        lines: List[str] = []
        for p in sorted(self.runs_dir.glob("*/summary.json"), reverse=True)[:8]:
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            items = obj if isinstance(obj, list) else [obj]
            for item in items[:5]:
                task = item.get("task", "unknown")
                best = item.get("best")
                attempts = item.get("attempts", [])
                lines.append(f"- task={task}; best={bool(best)}; attempts={len(attempts)}")
                if best and isinstance(best, dict):
                    bench = best.get("bench_result", {})
                    speed = bench.get("candidate_speedup") or bench.get("identity_speedup")
                    lines.append(f"  speedup={speed}")
        return ("\n".join(lines) or "No useful previous memory.")[:max_chars]

    def record_success(self, task: str, analysis: Dict, snippet: str, result: Dict) -> None:
        path = self.root / "learned_memory.jsonl"
        row = {"task": task, "analysis": analysis, "snippet_head": snippet[:2000], "result": result}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
