from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

from .code_utils import make_reference_copy_solution, read_text, sha256_text, strip_markdown_fences, too_similar, write_text
from .config import RunConfig
from .evaluator import EvalResult, evaluate_auto
from .llm import LLMClient, LLMConfig
from .memory import KernelMemory, MemoryItem, signature_key
from .prompting import build_messages
from .task import analyze_task


class RadioForgeRunner:
    def __init__(self, cfg: RunConfig):
        cfg.validate()
        self.cfg = cfg
        self.repo_root = Path.cwd()
        self.memory = KernelMemory(cfg.out_dir / "memory" / "kernel_memory.jsonl")
        self.llm = LLMClient(LLMConfig(model=cfg.model, temperature=cfg.temperature, timeout_s=cfg.timeout_s))
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

    def _candidate_from_llm(self, sig, memory_items, feedback: str | None) -> str:
        if self.cfg.no_llm or not self.llm.available:
            return make_reference_copy_solution(read_text(Path(sig.path)))
        messages = build_messages(sig, memory_items, feedback)
        raw = self.llm.chat(messages)
        return strip_markdown_fences(raw)

    def _evaluate(self, task_path: Path, cand_path: Path) -> EvalResult:
        return evaluate_auto(
            ref_path=task_path,
            solution_path=cand_path,
            repo_root=self.repo_root,
            timeout_s=self.cfg.timeout_s,
            precision=self.cfg.precision,
            num_correct_trials=self.cfg.num_correct_trials,
            num_perf_trials=self.cfg.num_perf_trials,
            warmup=self.cfg.warmup,
        )

    def run_task(self, task_path: Path) -> dict:
        sig = analyze_task(task_path)
        task_dir = self.cfg.out_dir / "candidates" / sig.stem
        log_dir = self.cfg.out_dir / "logs" / sig.stem
        task_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        mem_items = self.memory.retrieve(sig)
        best: dict | None = None
        feedback: str | None = None
        prior_codes: list[str] = []
        round_summaries: list[dict] = []

        def evaluate_code(round_id: str, code: str, source: str) -> EvalResult:
            nonlocal best, feedback
            prior_codes.append(code)
            cand_path = task_dir / f"{round_id}.py"
            write_text(cand_path, code)
            result = self._evaluate(task_path, cand_path)
            log = {
                "task": str(task_path),
                "round": round_id,
                "source": source,
                "candidate": str(cand_path),
                "signature": sig.to_dict(),
                "result": result.to_dict(),
                "code_sha256": sha256_text(code),
                "time": time.time(),
            }
            write_text(log_dir / f"{round_id}.json", json.dumps(log, ensure_ascii=False, indent=2))
            round_summary = {
                "round": round_id,
                "source": source,
                "candidate": str(cand_path),
                "compiled": result.compiled,
                "correct": result.correct,
                "runtime_ms": result.runtime_ms,
                "ref_runtime_ms": result.ref_runtime_ms,
                "speedup": result.speedup,
                "stderr_tail": (result.stderr or "")[-1000:],
            }
            round_summaries.append(round_summary)
            if self.cfg.verbose:
                print(json.dumps(round_summary, ensure_ascii=False), flush=True)
            note = "ok" if result.correct else result.feedback_text(max_chars=1500)
            self.memory.append(MemoryItem(
                task=str(task_path),
                signature_key=signature_key(sig),
                patterns=sig.patterns,
                code_hash=sha256_text(code),
                correct=result.correct,
                compiled=result.compiled,
                speedup=result.speedup,
                runtime_ms=result.runtime_ms,
                ref_runtime_ms=result.ref_runtime_ms,
                note=note[:2000],
                code_path=str(cand_path),
            ))
            if result.correct and (best is None or (result.speedup or 0) > (best["result"].speedup or 0)):
                best = {"round": round_id, "path": str(cand_path), "result": result}
            feedback = result.feedback_text()
            return result

        # Always seed with a safe reference-copy candidate. This prevents a single bad LLM
        # response from producing best_result=null and gives a measured correctness baseline.
        if self.cfg.seed_reference:
            seed_code = make_reference_copy_solution(read_text(Path(sig.path)))
            evaluate_code("seed_ref", seed_code, "reference_copy_seed")

        for r in range(self.cfg.rounds):
            code = self._candidate_from_llm(sig, mem_items, feedback)
            if any(too_similar(code, old) for old in prior_codes) and feedback is not None and self.llm.available and not self.cfg.no_llm:
                feedback = (feedback or "") + "\nThe last candidate was too similar to a previous one. Change the implementation strategy."
                code = self._candidate_from_llm(sig, mem_items, feedback)
            evaluate_code(f"round_{r}", code, "llm" if self.llm.available and not self.cfg.no_llm else "reference_copy_no_llm")

        promoted = None
        if best is not None:
            promoted_path = task_dir / "best.py"
            write_text(promoted_path, read_text(Path(best["path"])))
            promoted = str(promoted_path)
        return {
            "task": str(task_path),
            "signature_key": signature_key(sig),
            "best_candidate": promoted,
            "best_round": None if best is None else best["round"],
            "best_result": None if best is None else best["result"].to_dict(),
            "rounds": round_summaries,
        }

    def run(self) -> dict:
        tasks = [Path(p) for p in sorted(glob.glob(self.cfg.tasks, recursive=True))]
        summary = {"method": "RadioForge-SEMS", "arch": "8.9", "tasks": []}
        for task in tasks:
            if task.is_file():
                item = self.run_task(task)
                summary["tasks"].append(item)
                if self.cfg.verbose:
                    print(f"===== {task} =====", flush=True)
                    print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)
        write_text(self.cfg.out_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
