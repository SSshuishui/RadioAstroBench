from __future__ import annotations

from pathlib import Path
import json

from .task import TaskSignature

ROOT = Path(__file__).resolve().parent
PROMPT_DIR = ROOT / "prompts"
KNOWLEDGE_DIR = ROOT / "knowledge"


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def load_knowledge() -> str:
    chunks = []
    for p in sorted(KNOWLEDGE_DIR.glob("*.md")):
        chunks.append(f"# {p.name}\n" + p.read_text(encoding="utf-8"))
    return "\n\n".join(chunks)


def build_messages(sig: TaskSignature, memory_items: list[dict], previous_feedback: str | None = None) -> list[dict[str, str]]:
    system = load_prompt("system.md")
    user_template = load_prompt("generate.md" if previous_feedback is None else "repair.md")
    user = user_template.format(
        signature=json.dumps(sig.to_dict(), ensure_ascii=False, indent=2),
        memory=json.dumps(memory_items, ensure_ascii=False, indent=2)[:12000],
        knowledge=load_knowledge()[:16000],
        feedback=previous_feedback or "",
        source=sig.source_excerpt,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
