from __future__ import annotations
import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .candidate import build_history_block, last_n_lines, make_candidate_source, normalize_candidate
from .evaluators import evaluate_kernelbench, evaluate_radio, outcome_failure_text
from .individual import KernelIndividual
from .io_utils import extract_cuda_kernel_names, extract_json, task_slug
from .llm import query_server
from .ncu_profiler import (
    load_csv_text,
    metrics_to_prompt,
    profile_script,
    write_kernelbench_ncu_driver,
    write_radio_ncu_driver,
)
from .prompts import (
    DEFAULT_SYSTEM_PROMPT,
    build_correctness_prompts,
    build_error_prompt,
    build_judger_optimization_prompts,
    build_optimization_prompt,
    build_seed_prompt,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("CudaForge-style iterative CUDA optimization inside radio_astronomy_agent")
    p.add_argument("--dataset", choices=["radio", "kernelbench"], default="radio")
    p.add_argument("--task", required=True, help="Path to one task file or directory")
    p.add_argument("--gpu", default="RTX 4090")
    p.add_argument("--server_type", default="deepseek")
    p.add_argument("--server_address", default="localhost")
    p.add_argument("--server_port", type=int, default=8000)
    p.add_argument("--model_name", default="deepseek-coder")
    p.add_argument("--round", "-G", type=int, default=5)
    p.add_argument("--max-iters", dest="max_iters", type=int, default=None, help="Alias for --round; kept for radio smoke/real scripts")
    p.add_argument("--work_dir", type=Path, default=Path("runs/cudaforge"))
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--scale", default="smoke", help="radio_bench only")
    p.add_argument("--segment-profile", default="all10", help="radio_bench only")
    p.add_argument("--fixture", default=None, help="radio_bench only: explicit fixture .pt/.json path")
    p.add_argument("--fixture-profile", default=None, help="radio_bench only: real fixture profile, e.g. nside512_day1_10m_ring")
    p.add_argument("--require-fixture", action="store_true", help="radio_bench only: skip/fail tasks without mapped real fixtures")
    p.add_argument("--timeout-s", type=int, default=600)
    p.add_argument("--first_n", type=int, default=0)
    p.add_argument("--num_tasks", type=int, default=1)
    p.add_argument("--shuffle_seed", type=int, default=0)
    p.add_argument("--subproc_id", type=int, default=0)
    p.add_argument("--mock", action="store_true", help="Do not call an LLM; use identity ModelNew")
    p.add_argument("--no-ncu", action="store_true", help="Disable NCU profiling in optimization rounds")
    p.add_argument("--ncu-repeat", type=int, default=20)
    return p


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _collect_tasks(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted([p for p in path.rglob("*.py") if p.is_file()])
    raise FileNotFoundError(path)


def _select_tasks(tasks: List[Path], first_n: int, num_tasks: int, seed: int) -> List[Path]:
    if first_n and first_n > 0:
        return tasks[: max(1, min(first_n, len(tasks)))]
    k = max(1, min(num_tasks, len(tasks)))
    rng = random.Random(seed or int(time.time()))
    return rng.sample(tasks, k)


def _call_llm(args, prompt: str, system_prompt: str, log_path: Path, call_type: str, round_idx: int) -> str:
    return query_server(
        prompt,
        system_prompt,
        server_type=args.server_type,
        model_name=args.model_name,
        server_address=args.server_address,
        server_port=args.server_port,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        log_path=log_path,
        call_type=call_type,
        round_idx=round_idx,
    )


def _mock_code() -> str:
    return "class ModelNew(Model):\n    pass\n"


def _evaluate(args, repo: Path, dataset: str, task: Path, candidate: Path, attempt_dir: Path):
    if dataset == "radio":
        return evaluate_radio(
            repo_root=repo,
            candidate_task=candidate,
            original_task_path=task,
            scale=args.scale,
            segment_profile=args.segment_profile,
            fixture=args.fixture,
            fixture_profile=args.fixture_profile,
            require_fixture=args.require_fixture,
            warmup=args.warmup,
            repeat=args.repeat,
            device=args.device,
            timeout_s=args.timeout_s,
            json_path=attempt_dir / "bench_result.json",
        )
    return evaluate_kernelbench(
        ref_task=task,
        candidate_task=candidate,
        warmup=args.warmup,
        repeat=args.repeat,
        device=args.device,
        tol=args.tol,
    )


def _profile(args, repo: Path, dataset: str, task: Path, candidate: Path, attempt_dir: Path) -> str:
    if args.no_ncu:
        return "NCU disabled by --no-ncu."
    kernels = extract_cuda_kernel_names(candidate)
    driver = attempt_dir / "ncu_driver.py"
    csv_path = attempt_dir / "ncu_metrics.csv"
    if dataset == "radio":
        write_radio_ncu_driver(driver, candidate, args.scale, args.segment_profile, args.warmup, args.repeat)
    else:
        write_kernelbench_ncu_driver(driver, task, candidate, args.warmup, args.repeat, args.device, args.tol)
    try:
        profile_script(driver, csv_path, kernel_names=kernels, repeat=args.ncu_repeat, cwd=repo)
        return metrics_to_prompt(load_csv_text(csv_path))
    except Exception as e:
        return "NCU profiling failed or unavailable. Fallback to benchmark metrics.\n" + str(e)[-4000:]


def _append_usage_totals(log_path: Path) -> Dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if not log_path.exists():
        return totals
    with log_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in totals:
            try:
                totals[k] += int(r.get(k, 0) or 0)
            except Exception:
                pass
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "round_idx", "call_type", "input_tokens", "output_tokens", "total_tokens"])
        writer.writerow({"timestamp": "Total", "round_idx": "", "call_type": "sum", **totals})
    return totals


def _save_curve(path: Path, scores: List[float], error_flags: List[bool], title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = list(range(len(scores)))
        plt.figure()
        plt.plot(xs, scores, marker="o")
        for x, y, bad in zip(xs, scores, error_flags):
            if bad:
                plt.scatter([x], [y], marker="x")
        plt.xlabel("Round")
        plt.ylabel("Speedup")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.5)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def _run_one(args, repo: Path, task: Path, batch_dir: Path) -> Dict[str, Any]:
    task_rel = task.relative_to(repo) if task.is_relative_to(repo) else task
    original = task.read_text(encoding="utf-8", errors="ignore")
    task_dir = batch_dir / task_slug(task_rel)
    code_dir = task_dir / "code"
    eval_dir = task_dir / "evaluation"
    io_dir = eval_dir / "llm_io"
    for d in (code_dir, eval_dir, io_dir):
        d.mkdir(parents=True, exist_ok=True)
    (task_dir / "original_task.py").write_text(original, encoding="utf-8")
    log_path = task_dir / "usage.csv"

    current: Optional[KernelIndividual] = None
    best: Optional[KernelIndividual] = None
    best_score = float("-inf")
    scores: List[float] = []
    error_flags: List[bool] = []
    last_score = 0.0

    for round_idx in range(args.round):
        print(f"[{task_rel}] Round {round_idx}")
        attempt_dir = eval_dir / f"attempt_{round_idx:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        if args.mock:
            raw = _mock_code()
            call_type = "mock"
        elif round_idx == 0:
            prompt = build_seed_prompt(task, args.gpu, args.dataset, scale=args.scale, fixture_profile=args.fixture_profile)
            (io_dir / f"round{round_idx:03d}_seed_prompt.txt").write_text(prompt, encoding="utf-8")
            raw = _call_llm(args, prompt, DEFAULT_SYSTEM_PROMPT, log_path, "seed", round_idx)
            call_type = "seed"
        else:
            runnable = bool(current and current.metrics and current.metrics.get("runnable"))
            if not runnable:
                err = last_n_lines(current.metrics.get("message", "") if current and current.metrics else "", 160)
                sys_p, prob_prompt = build_correctness_prompts(err, task, current.code if current else "")
                (io_dir / f"round{round_idx:03d}_problem_identify_prompt.txt").write_text(prob_prompt, encoding="utf-8")
                raw_prob = _call_llm(args, prob_prompt, sys_p, log_path, "problem_identify", round_idx)
                (io_dir / f"round{round_idx:03d}_problem_identify_reply.txt").write_text(raw_prob, encoding="utf-8")
                problem = extract_json(raw_prob)
                prompt = build_error_prompt(current.code if current else "", err, problem, args.gpu)
                (io_dir / f"round{round_idx:03d}_repair_prompt.txt").write_text(prompt, encoding="utf-8")
                raw = _call_llm(args, prompt, DEFAULT_SYSTEM_PROMPT, log_path, "repair", round_idx)
                call_type = "repair"
            else:
                metrics_block = _profile(args, repo, args.dataset, task, current.code_path, attempt_dir)  # type: ignore[arg-type]
                sys_j, judge_prompt = build_judger_optimization_prompts(task, args.gpu, metrics_block, current.code)  # type: ignore[union-attr]
                (io_dir / f"round{round_idx:03d}_judge_optimization_prompt.txt").write_text(judge_prompt, encoding="utf-8")
                raw_judge = _call_llm(args, judge_prompt, sys_j, log_path, "judge_optimization", round_idx)
                (io_dir / f"round{round_idx:03d}_optimization_strategy_reply.txt").write_text(raw_judge, encoding="utf-8")
                strategy = extract_json(raw_judge)
                prompt = build_optimization_prompt(current.code_path, args.gpu, strategy, history_block=build_history_block(code_dir, keep_last=5))  # type: ignore[arg-type]
                (io_dir / f"round{round_idx:03d}_opt_prompt.txt").write_text(prompt, encoding="utf-8")
                raw = _call_llm(args, prompt, DEFAULT_SYSTEM_PROMPT, log_path, "optimization", round_idx)
                call_type = "optimization"

        (io_dir / f"round{round_idx:03d}_{call_type}_raw_reply.txt").write_text(raw, encoding="utf-8")
        try:
            candidate_code = normalize_candidate(raw, original_source=original)
            candidate_source = make_candidate_source(original, candidate_code, comment=f"CudaForge {call_type} round={round_idx}")
            ind = KernelIndividual(candidate_source)
            ind.save_code(code_dir)
            out = _evaluate(args, repo, args.dataset, task, ind.code_path, attempt_dir)  # type: ignore[arg-type]
            ind.metrics = out.metrics
            ind.score = out.score
            if not out.ok:
                ind.metrics["message"] = outcome_failure_text(out)
            ind.save_metrics(eval_dir)
        except Exception as e:
            ind = KernelIndividual(raw)
            ind.metrics = {"runnable": False, "error_type": e.__class__.__name__, "message": str(e)[-8000:]}
            ind.score = float("-inf")
            ind.save_metrics(eval_dir)

        current = ind
        runnable = bool(ind.metrics and ind.metrics.get("runnable"))
        if runnable and ind.score is not None:
            last_score = float(ind.score)
            scores.append(last_score)
            error_flags.append(False)
            if last_score > best_score:
                best_score = last_score
                best = ind
                (task_dir / "best_candidate.py").write_text(best.code, encoding="utf-8")
        else:
            scores.append(last_score)
            error_flags.append(True)
        print(f"[round {round_idx}] runnable={runnable} score={ind.score}")

    fig = task_dir / "figures" / f"{task.stem}_score.png"
    _save_curve(fig, scores, error_flags, f"{task.stem} best={best_score:.4f}")
    usage = _append_usage_totals(log_path)
    return {
        "task": str(task_rel),
        "best_score": float(best_score) if best_score != float("-inf") else 0.0,
        "best_runnable": bool(best and best.metrics and best.metrics.get("runnable")),
        "task_dir": str(task_dir),
        "figure": str(fig),
        **usage,
    }


def main():
    args = _parser().parse_args()
    if getattr(args, "max_iters", None) is not None:
        args.round = args.max_iters
    repo = _repo_root()
    task_path = Path(args.task)
    if not task_path.is_absolute():
        task_path = repo / task_path
    tasks = _collect_tasks(task_path)
    picked = tasks if task_path.is_file() else _select_tasks(tasks, args.first_n, args.num_tasks, args.shuffle_seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch = args.work_dir / f"{stamp}_{args.dataset}_{args.model_name.replace('/', '_')}"
    batch.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, t in enumerate(picked, 1):
        print(f"===== [{i}/{len(picked)}] {t} =====")
        summary.append(_run_one(args, repo, t, batch))
    avg = sum(x["best_score"] for x in summary) / max(1, len(summary))
    acc = sum(1 for x in summary if x["best_runnable"]) / max(1, len(summary))
    out = {
        "avg_speedup": avg,
        "accuracy": acc,
        "num_tasks": len(summary),
        "dataset": args.dataset,
        "scale": args.scale,
        "segment_profile": args.segment_profile,
        "fixture": args.fixture,
        "fixture_profile": args.fixture_profile,
        "tasks": summary,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (batch / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    with (batch / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "best_score", "best_runnable", "task_dir", "figure"])
        for s in summary:
            w.writerow([s["task"], f'{s["best_score"]:.6f}', int(s["best_runnable"]), s["task_dir"], s["figure"]])
        w.writerow([]); w.writerow(["avg_speedup", f"{avg:.6f}"]); w.writerow(["accuracy", f"{acc:.6f}"])
    print(f"[GLOBAL] Saved {batch/'summary.json'}")
    print(f"[GLOBAL] avg_speedup={avg:.4f} accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
