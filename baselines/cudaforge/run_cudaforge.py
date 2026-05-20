from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from baselines.llm_direct.llm_client import LLMConfig, OpenAICompatibleChatClient

from .candidate_utils import make_candidate_source, short_hash, strip_code_fences, task_slug
from .evaluators import evaluate_kernelbench_candidate, evaluate_radio_candidate
from .prompt_templates import (
    JUDGE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_coder_prompt,
    build_judge_prompt,
    build_repair_prompt,
)


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def build_mock_snippet() -> str:
    return "class ModelNew(Model):\n    pass\n"


def discover_tasks(repo_root: Path, dataset: str, level: str | None) -> List[Path]:
    if dataset == "radio":
        base = repo_root / "radio_bench"
        if level:
            return sorted((base / level).glob("*.py"))
        return sorted(base.glob("level*/*.py"))
    if dataset == "kernelbench":
        base = repo_root / "kernelbench"
        if level:
            return sorted((base / level).glob("*.py"))
        return sorted(base.glob("level*/*.py"))
    raise ValueError(f"unknown dataset={dataset}")


def relative_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def build_history_block(attempts: List[Dict], max_items: int = 3) -> str:
    if not attempts:
        return "(none)"
    lines = []
    for item in attempts[-max_items:]:
        lines.append(json.dumps({
            "attempt": item.get("attempt"),
            "ok": item.get("ok"),
            "stage": item.get("stage"),
            "speedup": item.get("speedup"),
            "candidate": item.get("candidate_task"),
        }, ensure_ascii=False))
    return "\n".join(lines)


def extract_speedup(dataset: str, result_json) -> Optional[float]:
    if not isinstance(result_json, dict):
        return None
    if dataset == "radio":
        return result_json.get("identity_speedup") or result_json.get("candidate_speedup")
    return result_json.get("speedup")


def evaluate(
    *,
    dataset: str,
    repo_root: Path,
    reference_task: Path,
    candidate_task: Path,
    scale: str,
    segment_profile: str,
    warmup: int,
    repeat: int,
    timeout_s: int,
    cuda_visible_devices: str,
    json_path: Path,
    tol: float,
):
    if dataset == "radio":
        return evaluate_radio_candidate(
            repo_root=repo_root,
            candidate_task=candidate_task,
            scale=scale,
            segment_profile=segment_profile,
            warmup=warmup,
            repeat=repeat,
            timeout_s=timeout_s,
            cuda_visible_devices=cuda_visible_devices,
            json_path=json_path,
        )
    return evaluate_kernelbench_candidate(
        repo_root=repo_root,
        reference_task=reference_task,
        candidate_task=candidate_task,
        warmup=warmup,
        repeat=repeat,
        timeout_s=timeout_s,
        cuda_visible_devices=cuda_visible_devices,
        json_path=json_path,
        tol=tol,
    )


def run_one_task(args, repo_root: Path, task_path: Path, run_root: Path, client: OpenAICompatibleChatClient | None) -> Dict:
    original_source = read_text(task_path)
    rel_task = relative_or_abs(task_path, repo_root)
    slug = task_slug(rel_task)
    task_run_dir = run_root / slug
    task_run_dir.mkdir(parents=True, exist_ok=True)
    (task_run_dir / "original_task.py").write_text(original_source, encoding="utf-8")

    profile_text = ""
    if args.profile_text_file:
        profile_path = Path(args.profile_text_file)
        if profile_path.exists():
            profile_text = profile_path.read_text(encoding="utf-8", errors="ignore")

    summary: Dict = {
        "baseline": "cudaforge",
        "dataset": args.dataset,
        "task": rel_task,
        "scale": args.scale,
        "segment_profile": args.segment_profile,
        "attempts": [],
        "best": None,
    }

    previous_snippet = ""
    previous_failure = ""
    judge_plan = ""

    for attempt in range(args.max_iters):
        attempt_dir = task_run_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        if args.mock:
            raw = build_mock_snippet()
        else:
            if attempt > 0 and previous_failure:
                prompt = build_repair_prompt(
                    dataset=args.dataset,
                    task_path=rel_task,
                    task_source=original_source,
                    previous_snippet=previous_snippet,
                    failure_text=previous_failure,
                    scale=args.scale,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
            else:
                prompt = build_coder_prompt(
                    dataset=args.dataset,
                    task_path=rel_task,
                    task_source=original_source,
                    scale=args.scale,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    history_block=build_history_block(summary["attempts"]),
                    judge_plan=judge_plan,
                )
            (attempt_dir / "coder_prompt.txt").write_text(prompt, encoding="utf-8")
            raw = client.complete_text([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])  # type: ignore[union-attr]

        snippet = strip_code_fences(raw)
        (attempt_dir / "llm_raw.txt").write_text(raw, encoding="utf-8")
        (attempt_dir / "candidate_snippet.py").write_text(snippet, encoding="utf-8")

        try:
            candidate_source = make_candidate_source(original_source, snippet, label=f"attempt={attempt} hash={short_hash(snippet)}")
        except Exception as e:  # noqa: BLE001
            failure = f"Candidate snippet validation failed: {e}"
            previous_snippet = snippet
            previous_failure = failure
            record = {"attempt": attempt, "ok": False, "stage": "snippet_validation", "failure": failure}
            summary["attempts"].append(record)
            write_json(attempt_dir / "result.json", record)
            continue

        candidate_task = attempt_dir / "candidate_task.py"
        candidate_task.write_text(candidate_source, encoding="utf-8")

        eval_result = evaluate(
            dataset=args.dataset,
            repo_root=repo_root,
            reference_task=task_path,
            candidate_task=candidate_task,
            scale=args.scale,
            segment_profile=args.segment_profile,
            warmup=args.warmup,
            repeat=args.repeat,
            timeout_s=args.timeout_s,
            cuda_visible_devices=args.cuda_visible_devices,
            json_path=attempt_dir / "bench_result.json",
            tol=args.tol,
        )
        (attempt_dir / "stdout.txt").write_text(eval_result.stdout, encoding="utf-8")
        (attempt_dir / "stderr.txt").write_text(eval_result.stderr, encoding="utf-8")

        speedup = extract_speedup(args.dataset, eval_result.result_json)
        record = {
            "attempt": attempt,
            "ok": eval_result.ok,
            "stage": eval_result.stage,
            "returncode": eval_result.returncode,
            "speedup": speedup,
            "candidate_task": relative_or_abs(candidate_task, repo_root),
            "bench_result": eval_result.result_json,
        }
        summary["attempts"].append(record)
        write_json(attempt_dir / "result.json", record)

        previous_snippet = snippet
        previous_failure = eval_result.failure_text()

        if eval_result.ok:
            best_speedup = summary["best"].get("speedup") if isinstance(summary.get("best"), dict) else None
            if summary["best"] is None or (speedup is not None and (best_speedup is None or speedup > best_speedup)):
                summary["best"] = {
                    "attempt": attempt,
                    "speedup": speedup,
                    "candidate_task": relative_or_abs(candidate_task, repo_root),
                    "candidate_snippet": relative_or_abs(attempt_dir / "candidate_snippet.py", repo_root),
                    "bench_result": eval_result.result_json,
                }
            if args.judge_after_success and not args.mock and attempt + 1 < args.max_iters:
                judge_prompt = build_judge_prompt(
                    dataset=args.dataset,
                    task_path=rel_task,
                    task_source=original_source,
                    result_json=json.dumps(eval_result.result_json, indent=2, ensure_ascii=False) if eval_result.result_json else "{}",
                    stdout_tail=eval_result.stdout,
                    stderr_tail=eval_result.stderr,
                    ncu_text=profile_text,
                )
                (attempt_dir / "judge_prompt.txt").write_text(judge_prompt, encoding="utf-8")
                judge_plan = client.complete_text([{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": judge_prompt}])  # type: ignore[union-attr]
                (attempt_dir / "judge_plan.txt").write_text(judge_plan, encoding="utf-8")
                previous_failure = ""
                if not args.continue_after_success:
                    # one success + judge record is enough unless user requests more
                    break
            elif not args.continue_after_success:
                break

    write_json(task_run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="CudaForge-style baseline for radio_bench and KernelBench tasks.")
    parser.add_argument("--dataset", default="radio", choices=["radio", "kernelbench"])
    parser.add_argument("--task", default="", help="Path to one task file. If omitted, use --level.")
    parser.add_argument("--level", default="", help="Run all tasks in a level, e.g. level1/level2/level3/level4.")
    parser.add_argument("--first-n", type=int, default=0, help="Limit discovered tasks to the first N.")
    parser.add_argument("--scale", default="smoke", choices=["smoke", "nside512_full", "nside4096_full", "nside16384_full"])
    parser.add_argument("--segment-profile", default="all10")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-4, help="KernelBench tolerance.")
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--continue-after-success", action="store_true")
    parser.add_argument("--judge-after-success", action="store_true", help="Ask a judge for the next optimization plan after a successful candidate.")
    parser.add_argument("--profile-text-file", default="", help="Optional NCU/profile text included in judge prompt.")
    parser.add_argument("--mock", action="store_true", help="Use identity ModelNew instead of calling an LLM.")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--run-root", default="baselines/cudaforge/runs")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    if args.task:
        task_paths = [Path(args.task)]
        task_paths = [p if p.is_absolute() else repo_root / p for p in task_paths]
    else:
        task_paths = discover_tasks(repo_root, args.dataset, args.level or None)
        if args.first_n > 0:
            task_paths = task_paths[: args.first_n]
    if not task_paths:
        raise RuntimeError("No tasks selected")

    client = None
    if not args.mock:
        config = LLMConfig.from_env(
            model=args.model or None,
            api_base=args.api_base or None,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
        )
        client = OpenAICompatibleChatClient(config)

    run_root = repo_root / args.run_root / f"{now_tag()}_{args.dataset}"
    run_root.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for task_path in task_paths:
        print(f"\n=== CudaForge baseline: dataset={args.dataset} task={relative_or_abs(task_path, repo_root)} ===", flush=True)
        summary = run_one_task(args, repo_root, task_path, run_root, client)
        all_summaries.append(summary)
        print(json.dumps(summary.get("best"), indent=2, ensure_ascii=False), flush=True)
    write_json(run_root / "summary_all.json", all_summaries)
    print(f"\nWrote {run_root}")


if __name__ == "__main__":
    main()
