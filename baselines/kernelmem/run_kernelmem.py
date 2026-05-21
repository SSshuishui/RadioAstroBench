from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from .candidate_utils import make_candidate_source, short_hash, strip_code_fences, task_slug
from .evaluators import evaluate_kernelbench, evaluate_radio
from .llm_client import LLMConfig, OpenAICompatibleChatClient
from .memory import MemoryBank
from .ncu_profile import maybe_profile_radio_task
from .prompt_builder import SYSTEM_PROMPT, build_generation_prompt, build_repair_prompt


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def discover_tasks(repo_root: Path, bench: str, level: Optional[str]) -> List[Path]:
    if bench == "radio":
        base = repo_root / "radio_bench"
    elif bench == "kernelbench":
        base = repo_root / "kernelbench"
    else:
        raise ValueError(f"unknown bench: {bench}")
    if level:
        return sorted((base / level).glob("*.py"))
    return sorted(base.glob("level*/*.py"))


def mock_snippet() -> str:
    return "class ModelNew(Model):\n    pass\n"


def evaluate(
    *,
    bench: str,
    repo_root: Path,
    candidate_task: Path,
    args,
    attempt_dir: Path,
):
    if bench == "radio":
        return evaluate_radio(
            repo_root=repo_root,
            candidate_task=candidate_task,
            scale=args.scale,
            segment_profile=args.segment_profile,
            warmup=args.warmup,
            repeat=args.repeat,
            timeout_s=args.timeout_s,
            cuda_visible_devices=args.cuda_visible_devices,
            json_path=attempt_dir / "bench_result.json",
        )
    if bench == "kernelbench":
        return evaluate_kernelbench(
            repo_root=repo_root,
            candidate_task=candidate_task,
            warmup=args.warmup,
            repeat=args.repeat,
            timeout_s=args.timeout_s,
            cuda_visible_devices=args.cuda_visible_devices,
            json_path=attempt_dir / "bench_result.json",
            atol=args.atol,
            rtol=args.rtol,
        )
    raise ValueError(bench)


def run_one_task(
    *,
    repo_root: Path,
    task_path: Path,
    bench: str,
    args,
    run_root: Path,
    client: Optional[OpenAICompatibleChatClient],
    memory_bank: MemoryBank,
) -> Dict:
    original = read_text(task_path)
    rel = task_path.relative_to(repo_root)
    slug = task_slug(rel)
    task_dir = run_root / slug
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "original_task.py").write_text(original, encoding="utf-8")

    long_mem = memory_bank.long_term_block(max_chars=args.long_memory_chars)
    short_mem = memory_bank.short_term_block(repo_root / "baselines/kernelmem/runs", bench=bench, max_chars=args.short_memory_chars)
    task_struct = memory_bank.task_structure_block(original)

    summary: Dict = {
        "bench": bench,
        "task": str(rel),
        "scale": args.scale if bench == "radio" else "kernelbench_default_inputs",
        "attempts": [],
        "best": None,
    }
    previous_snippet = ""
    previous_failure = ""
    ncu_text = ""

    for attempt in range(args.rounds):
        attempt_dir = task_dir / f"round_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        if args.mock:
            raw = mock_snippet()
        else:
            if attempt == 0:
                user_prompt = build_generation_prompt(
                    bench=bench,
                    task_path=str(rel),
                    task_source=original,
                    task_structure=task_struct,
                    long_memory=long_mem,
                    short_memory=short_mem,
                    scale=args.scale,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
            else:
                failure = previous_failure
                if ncu_text:
                    failure += "\n\nNCU feedback:\n" + ncu_text
                user_prompt = build_repair_prompt(
                    bench=bench,
                    task_path=str(rel),
                    task_source=original,
                    previous_snippet=previous_snippet,
                    failure_summary=failure,
                    long_memory=long_mem,
                    short_memory=short_mem,
                    scale=args.scale,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
            (attempt_dir / "prompt.txt").write_text(SYSTEM_PROMPT + "\n\n" + user_prompt, encoding="utf-8")
            raw = client.complete_text([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ])  # type: ignore[union-attr]

        snippet = strip_code_fences(raw)
        (attempt_dir / "llm_raw.txt").write_text(raw, encoding="utf-8")
        (attempt_dir / "candidate_snippet.py").write_text(snippet, encoding="utf-8")
        try:
            candidate_src = make_candidate_source(original, snippet, metadata_comment=f"KernelMem round={attempt} hash={short_hash(snippet)}")
        except Exception as e:  # noqa: BLE001
            failure = f"candidate validation failed: {e}"
            record = {"round": attempt, "ok": False, "stage": "candidate_validation", "failure": failure}
            previous_snippet = snippet
            previous_failure = failure
            summary["attempts"].append(record)
            write_json(attempt_dir / "result.json", record)
            continue

        candidate_task = attempt_dir / "candidate_task.py"
        candidate_task.write_text(candidate_src, encoding="utf-8")
        result = evaluate(bench=bench, repo_root=repo_root, candidate_task=candidate_task, args=args, attempt_dir=attempt_dir)
        (attempt_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8", errors="replace")
        (attempt_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8", errors="replace")
        record = {"round": attempt, "ok": result.ok, "returncode": result.returncode, "bench_result": result.result_json}
        summary["attempts"].append(record)
        write_json(attempt_dir / "result.json", record)
        previous_snippet = snippet
        previous_failure = result.failure_text()

        if result.ok and args.enable_ncu and bench == "radio":
            ncu_text = maybe_profile_radio_task(
                repo_root=repo_root,
                candidate_task=candidate_task,
                scale=args.scale,
                segment_profile=args.segment_profile,
                cuda_visible_devices=args.cuda_visible_devices,
                out_dir=attempt_dir / "ncu",
                timeout_s=args.ncu_timeout_s,
            ) or ""
            (attempt_dir / "ncu_feedback.txt").write_text(ncu_text, encoding="utf-8", errors="replace")

        if result.ok:
            summary["best"] = {
                "round": attempt,
                "candidate_task": str(candidate_task.relative_to(repo_root)),
                "candidate_snippet": str((attempt_dir / "candidate_snippet.py").relative_to(repo_root)),
                "bench_result": result.result_json,
            }
            if not args.continue_after_success:
                break

    write_json(task_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="KernelMem-style baseline for radio_bench and KernelBench.")
    parser.add_argument("--bench", choices=["radio", "kernelbench"], default="radio")
    parser.add_argument("--task", default="", help="Task path. If omitted, use --level.")
    parser.add_argument("--level", default="", choices=["", "level1", "level2", "level3", "level4"])
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--scale", default="smoke", choices=["smoke", "nside512_full", "nside4096_full", "nside16384_full"])
    parser.add_argument("--segment-profile", default="all10")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--continue-after-success", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--model", default="")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--long-memory-chars", type=int, default=24000)
    parser.add_argument("--short-memory-chars", type=int, default=12000)
    parser.add_argument("--enable-ncu", action="store_true")
    parser.add_argument("--ncu-timeout-s", type=int, default=600)
    parser.add_argument("--out-dir", default="baselines/kernelmem/runs")
    parser.add_argument("--mock", action="store_true", help="Append identity ModelNew without calling LLM; useful to test harness.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if args.task:
        p = Path(args.task)
        if not p.is_absolute():
            p = repo_root / p
        tasks = [p]
    else:
        tasks = discover_tasks(repo_root, args.bench, args.level or None)
        if args.num_tasks > 0:
            tasks = tasks[: args.num_tasks]
    if not tasks:
        raise RuntimeError("No tasks selected")

    run_root = repo_root / args.out_dir / now_tag()
    run_root.mkdir(parents=True, exist_ok=True)
    if args.mock:
        client = None
        llm_public = {"mock": True}
    else:
        cfg = LLMConfig.from_env(model=args.model or None, api_base=args.api_base or None, temperature=args.temperature, max_tokens=args.max_tokens)
        client = OpenAICompatibleChatClient(cfg)
        llm_public = {"api_base": cfg.api_base, "model": cfg.model, "temperature": cfg.temperature, "max_tokens": cfg.max_tokens}

    meta = {
        "baseline": "kernelmem",
        "bench": args.bench,
        "scale": args.scale,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "enable_ncu": args.enable_ncu,
        "llm": llm_public,
        "tasks": [str(t.relative_to(repo_root)) for t in tasks],
    }
    write_json(run_root / "run_meta.json", meta)
    memory_bank = MemoryBank(repo_root / "baselines/kernelmem")

    summaries = []
    for task in tasks:
        print(f"\n=== KernelMem bench={args.bench} task={task.relative_to(repo_root)} ===", flush=True)
        summaries.append(run_one_task(repo_root=repo_root, task_path=task, bench=args.bench, args=args, run_root=run_root, client=client, memory_bank=memory_bank))
    write_json(run_root / "summary.json", summaries)
    print(f"\nWrote run dir: {run_root}")


if __name__ == "__main__":
    main()
