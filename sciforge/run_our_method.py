from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from .benchmark_eval import EvalResult, evaluate_kernelbench, evaluate_radio
from .candidate_utils import make_candidate_source, short_hash, strip_code_fences, task_slug
from .llm_client import LLMConfig, OpenAICompatibleChatClient
from .memory_manager import DomainMemory
from .ncu_feedback import maybe_ncu_profile_radio
from .prompt_builder import SYSTEM_PROMPT, build_prompt
from .skill_library import route_skills
from .task_analyzer import analyze_task


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def discover_tasks(repo_root: Path, bench: str, level: Optional[str]) -> List[Path]:
    base = repo_root / ("radio_bench" if bench == "radio" else "kernelbench")
    if level:
        return sorted((base / level).glob("*.py"))
    return sorted(base.glob("level*/*.py"))


def mock_snippet() -> str:
    return "class ModelNew(Model):\n    pass\n"


def eval_candidate(bench: str, repo_root: Path, candidate_task: Path, args, out_dir: Path, scale: Optional[str] = None) -> EvalResult:
    if bench == "radio":
        return evaluate_radio(repo_root, candidate_task, scale or args.scale, args.segment_profile, args.warmup, args.repeat, args.timeout_s, args.cuda_visible_devices, out_dir / f"bench_result_{scale or args.scale}.json")
    return evaluate_kernelbench(repo_root, candidate_task, args.warmup, args.repeat, args.timeout_s, args.cuda_visible_devices, out_dir / "bench_result_kernelbench.json", args.atol, args.rtol)


def score_result(result: Optional[dict]) -> float:
    if not isinstance(result, dict): return 0.0
    speed = result.get("candidate_speedup") or result.get("identity_speedup") or 0.0
    try: return float(speed or 0.0)
    except Exception: return 0.0


def run_one_task(repo_root: Path, task_path: Path, bench: str, args, run_root: Path, client: Optional[OpenAICompatibleChatClient], memory: DomainMemory) -> Dict:
    source = read_text(task_path)
    rel = task_path.relative_to(repo_root)
    slug = task_slug(rel)
    task_dir = run_root / slug
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "original_task.py").write_text(source, encoding="utf-8")

    analysis = analyze_task(task_path, source, bench)
    skills = route_skills(analysis)
    long_mem = memory.long_term_memory(args.long_memory_chars)
    short_mem = memory.short_term_memory(args.short_memory_chars)
    write_json(task_dir / "analysis.json", analysis.to_dict())
    (task_dir / "skills.txt").write_text("\n".join(skills), encoding="utf-8")

    summary: Dict = {"method": "domain_aware_constrained_agent", "bench": bench, "task": str(rel), "analysis": analysis.to_dict(), "skills": skills, "attempts": [], "best": None}
    previous_snippet = ""; feedback = ""; ncu_text = ""; best_score = -1.0

    for round_idx in range(args.rounds):
        round_dir = task_dir / f"round_{round_idx:02d}"; round_dir.mkdir(parents=True, exist_ok=True)
        if args.mock:
            raw = mock_snippet()
        else:
            prompt = build_prompt(task_path=str(rel), bench=bench, source=source, analysis=analysis, skills=skills, long_memory=long_mem, short_memory=short_mem, scale=args.scale, warmup=args.warmup, repeat=args.repeat, previous_snippet=previous_snippet, feedback=feedback, ncu_feedback=ncu_text)
            (round_dir / "prompt.txt").write_text(SYSTEM_PROMPT + "\n\n" + prompt, encoding="utf-8")
            raw = client.complete_text([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])  # type: ignore[union-attr]
        snippet = strip_code_fences(raw)
        (round_dir / "llm_raw.txt").write_text(raw, encoding="utf-8")
        (round_dir / "candidate_snippet.py").write_text(snippet, encoding="utf-8")
        try:
            cand_src = make_candidate_source(source, snippet, comment=f"our_method round={round_idx} hash={short_hash(snippet)}")
        except Exception as e:  # noqa: BLE001
            feedback = f"candidate validation failed: {e}"
            record = {"round": round_idx, "ok": False, "stage": "candidate_validation", "failure": feedback}
            summary["attempts"].append(record); write_json(round_dir / "result.json", record); previous_snippet = snippet; continue
        candidate_task = round_dir / "candidate_task.py"; candidate_task.write_text(cand_src, encoding="utf-8")

        primary = eval_candidate(bench, repo_root, candidate_task, args, round_dir, scale=args.scale if bench == "radio" else None)
        (round_dir / "stdout.txt").write_text(primary.stdout, encoding="utf-8", errors="replace")
        (round_dir / "stderr.txt").write_text(primary.stderr, encoding="utf-8", errors="replace")
        validations = []
        if primary.ok and bench == "radio" and args.validate_scales:
            for sc in [x.strip() for x in args.validate_scales.split(",") if x.strip() and x.strip() != args.scale]:
                validations.append({"scale": sc, "result": eval_candidate(bench, repo_root, candidate_task, args, round_dir, scale=sc).__dict__})
        all_validation_ok = primary.ok and all(v["result"].get("ok") for v in validations)
        if primary.ok and args.enable_ncu and bench == "radio":
            ncu_text = maybe_ncu_profile_radio(repo_root, candidate_task, args.scale, args.segment_profile, args.cuda_visible_devices, round_dir / "ncu", args.ncu_timeout_s) or ""
            (round_dir / "ncu_feedback.txt").write_text(ncu_text, encoding="utf-8", errors="replace")
        record = {"round": round_idx, "ok": primary.ok, "all_validation_ok": all_validation_ok, "primary_result": primary.result_json, "validations": validations, "returncode": primary.returncode}
        summary["attempts"].append(record); write_json(round_dir / "result.json", record)
        previous_snippet = snippet; feedback = primary.failure_text()
        current_score = score_result(primary.result_json) if primary.ok else 0.0
        if all_validation_ok and current_score > best_score:
            best_score = current_score
            summary["best"] = {"round": round_idx, "score": current_score, "candidate_task": str(candidate_task.relative_to(repo_root)), "candidate_snippet": str((round_dir / "candidate_snippet.py").relative_to(repo_root)), "bench_result": primary.result_json, "validations": validations}
            memory.record_success(str(rel), analysis.to_dict(), snippet, primary.result_json or {})
            if not args.continue_after_success:
                break
    write_json(task_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Domain-aware constrained CUDA optimization agent.")
    parser.add_argument("--bench", choices=["radio", "kernelbench"], default="radio")
    parser.add_argument("--task", default="")
    parser.add_argument("--level", default="", choices=["", "level1", "level2", "level3", "level4"])
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--scale", default="smoke", choices=["smoke", "nside512_full", "nside4096_full", "nside16384_full"])
    parser.add_argument("--validate-scales", default="", help="Comma-separated extra radio scales, e.g. nside512_full")
    parser.add_argument("--segment-profile", default="all10")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=4)
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
    parser.add_argument("--out-dir", default="our_method/runs")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.task:
        t = Path(args.task); tasks = [t if t.is_absolute() else repo_root / t]
    else:
        tasks = discover_tasks(repo_root, args.bench, args.level or None)
        if args.num_tasks > 0: tasks = tasks[: args.num_tasks]
    if not tasks: raise RuntimeError("No tasks selected")
    run_root = repo_root / args.out_dir / now_tag(); run_root.mkdir(parents=True, exist_ok=True)
    client = None
    llm_public = {"mock": True}
    if not args.mock:
        cfg = LLMConfig.from_env(model=args.model or None, api_base=args.api_base or None, temperature=args.temperature, max_tokens=args.max_tokens)
        client = OpenAICompatibleChatClient(cfg)
        llm_public = {"api_base": cfg.api_base, "model": cfg.model, "temperature": cfg.temperature, "max_tokens": cfg.max_tokens}
    write_json(run_root / "run_meta.json", {"method": "domain_aware_constrained_agent", "bench": args.bench, "tasks": [str(t.relative_to(repo_root)) for t in tasks], "scale": args.scale, "validate_scales": args.validate_scales, "rounds": args.rounds, "llm": llm_public})
    memory = DomainMemory(repo_root / "our_method")
    summaries = []
    for task in tasks:
        print(f"\n=== OUR_METHOD bench={args.bench} task={task.relative_to(repo_root)} ===", flush=True)
        summaries.append(run_one_task(repo_root, task, args.bench, args, run_root, client, memory))
    write_json(run_root / "summary.json", summaries)
    print(f"\nWrote run dir: {run_root}")


if __name__ == "__main__":
    main()
