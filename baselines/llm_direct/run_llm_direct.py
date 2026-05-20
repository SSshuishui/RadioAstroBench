from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

from .candidate_utils import make_candidate_source, short_hash, strip_code_fences, task_slug
from .evaluator import evaluate_candidate
from .llm_client import LLMConfig, OpenAICompatibleChatClient
from .prompt_templates import SYSTEM_PROMPT, build_initial_prompt, build_repair_prompt


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def build_mock_snippet() -> str:
    return """
class ModelNew(Model):
    pass
""".strip()


def run_one_task(args, repo_root: Path, task_path: Path, run_root: Path, client: OpenAICompatibleChatClient | None) -> Dict:
    original_source = read_text(task_path)
    slug = task_slug(task_path.relative_to(repo_root))
    task_run_dir = run_root / slug
    task_run_dir.mkdir(parents=True, exist_ok=True)
    (task_run_dir / "original_task.py").write_text(original_source, encoding="utf-8")

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_initial_prompt(
                str(task_path.relative_to(repo_root)),
                original_source,
                args.scale,
                args.warmup,
                args.repeat,
            ),
        },
    ]
    summary = {
        "task": str(task_path.relative_to(repo_root)),
        "scale": args.scale,
        "segment_profile": args.segment_profile,
        "attempts": [],
        "best": None,
    }

    previous_snippet = ""
    previous_failure = ""
    for attempt in range(args.max_iters):
        attempt_dir = task_run_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        if args.mock:
            raw = build_mock_snippet()
        else:
            if attempt > 0:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_repair_prompt(
                            str(task_path.relative_to(repo_root)),
                            original_source,
                            previous_snippet,
                            previous_failure,
                            args.scale,
                            args.warmup,
                            args.repeat,
                        ),
                    },
                ]
            raw = client.complete_text(messages)  # type: ignore[union-attr]
        snippet = strip_code_fences(raw)
        (attempt_dir / "llm_raw.txt").write_text(raw, encoding="utf-8")
        (attempt_dir / "candidate_snippet.py").write_text(snippet, encoding="utf-8")
        try:
            candidate_source = make_candidate_source(
                original_source,
                snippet,
                metadata_comment=f"LLM direct attempt={attempt} hash={short_hash(snippet)}",
            )
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
        eval_result = evaluate_candidate(
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
        (attempt_dir / "stdout.txt").write_text(eval_result.stdout, encoding="utf-8")
        (attempt_dir / "stderr.txt").write_text(eval_result.stderr, encoding="utf-8")
        record = {
            "attempt": attempt,
            "ok": eval_result.ok,
            "returncode": eval_result.returncode,
            "bench_result": eval_result.result_json,
        }
        summary["attempts"].append(record)
        write_json(attempt_dir / "result.json", record)
        previous_snippet = snippet
        previous_failure = eval_result.failure_text()
        if eval_result.ok:
            summary["best"] = {
                "attempt": attempt,
                "candidate_task": str(candidate_task.relative_to(repo_root)),
                "candidate_snippet": str((attempt_dir / "candidate_snippet.py").relative_to(repo_root)),
                "bench_result": eval_result.result_json,
            }
            if not args.continue_after_success:
                break
    write_json(task_run_dir / "summary.json", summary)
    return summary


def discover_tasks(repo_root: Path, level: str | None) -> List[Path]:
    base = repo_root / "radio_bench"
    if level:
        return sorted((base / level).glob("*.py"))
    return sorted(base.glob("level*/*.py"))


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-only direct CUDA optimization baseline for radio_bench tasks.")
    parser.add_argument("--task", default="", help="Path to one task file, or omit with --level to run a group.")
    parser.add_argument("--level", default="", choices=["", "level1", "level2", "level3"], help="Run all tasks in a level if --task is omitted.")
    parser.add_argument("--scale", default="smoke", choices=["smoke", "nside512_full", "nside4096_full", "nside16384_full"])
    parser.add_argument("--segment-profile", default="all10")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-iters", type=int, default=3, help="LLM attempts including repair attempts.")
    parser.add_argument("--continue-after-success", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--model", default="", help="Override LLM_MODEL.")
    parser.add_argument("--api-base", default="", help="Override LLM_API_BASE.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--out-dir", default="baselines/llm_direct/runs")
    parser.add_argument("--mock", action="store_true", help="Do not call an API; append identity ModelNew for harness testing.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_root = repo_root / args.out_dir / now_tag()
    run_root.mkdir(parents=True, exist_ok=True)

    if args.task:
        task = Path(args.task)
        if not task.is_absolute():
            task = repo_root / task
        tasks = [task]
    else:
        tasks = discover_tasks(repo_root, args.level or None)
    if not tasks:
        raise RuntimeError("No tasks selected.")

    client = None
    if not args.mock:
        cfg = LLMConfig.from_env(
            model=args.model or None,
            api_base=args.api_base or None,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        client = OpenAICompatibleChatClient(cfg)
        llm_config_public = {
            "api_base": cfg.api_base,
            "model": cfg.model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
    else:
        llm_config_public = {"mock": True}

    run_meta = {
        "baseline": "llm_direct",
        "scale": args.scale,
        "segment_profile": args.segment_profile,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "max_iters": args.max_iters,
        "cuda_visible_devices": args.cuda_visible_devices,
        "llm": llm_config_public,
        "tasks": [str(t.relative_to(repo_root)) for t in tasks],
        "run_root": str(run_root.relative_to(repo_root)),
    }
    write_json(run_root / "run_meta.json", run_meta)

    summaries = []
    for task_path in tasks:
        print(f"\n=== LLM_DIRECT task={task_path.relative_to(repo_root)} scale={args.scale} ===", flush=True)
        summaries.append(run_one_task(args, repo_root, task_path, run_root, client))
    write_json(run_root / "summary.json", summaries)
    print(f"\nWrote run directory: {run_root}")


if __name__ == "__main__":
    main()
