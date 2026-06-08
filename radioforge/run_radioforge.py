#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from radioforge.config import RunConfig
from radioforge.runner import RadioForgeRunner


def _default_out_dir(scale: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("radioforge") / "runs" / f"{stamp}_{scale}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RadioForge-SEMS on radio_bench tasks")

    # KernelMem/CUDAForge-compatible interface used by the top-level bash files.
    p.add_argument("--bench", default="radio", choices=["radio"], help="Benchmark name; only radio is supported")
    p.add_argument("--task", help="Single task file, e.g. radio_bench/level1/05-ws-build-nm1.py")
    p.add_argument("--scale", default="smoke", help="radio_bench scale/profile label")
    p.add_argument("--fixture-profile", default=None, help="radio_bench real-data fixture profile label")
    p.add_argument("--warmup", type=int, default=3, help="Number of warmup trials passed to evaluator-compatible config")
    p.add_argument("--repeat", type=int, default=5, help="Number of perf repeat trials")
    p.add_argument("--max-iters", type=int, default=None, help="Search iterations; alias of --rounds")
    p.add_argument("--continue-after-success", action="store_true", help="Accepted for baseline script compatibility")
    p.add_argument("--api-base", default=None, help="OpenAI-compatible API base URL")
    p.add_argument("--enable-ncu", action="store_true", help="Accepted for real-script compatibility")

    # Native RadioForge interface retained for direct debugging.
    p.add_argument("--tasks", help="Glob for task files, e.g. 'radio_bench/level1/*.py'")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--arch", default="8.9", help="Only 8.9 is supported")
    p.add_argument("--model", default="deepseek-v4-pro")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--timeout-s", type=int, default=180)
    p.add_argument("--num-correct-trials", type=int, default=5)
    p.add_argument("--num-perf-trials", type=int, default=None)
    p.add_argument("--precision", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--no-llm", action="store_true", help="Debug mode: copy reference as candidate")
    p.add_argument("--seed-reference", action="store_true", help="Debug only: first evaluate a reference-copy seed candidate")
    p.add_argument("--verbose", action="store_true", help="Print per-round JSON records and full task summaries to stdout")
    return p.parse_args()


def main() -> None:
    a = parse_args()

    task_pattern = a.tasks or a.task
    if not task_pattern:
        raise SystemExit("[ERROR] either --task or --tasks is required")

    if a.api_base:
        os.environ["LLM_API_BASE"] = a.api_base
        os.environ.setdefault("OPENAI_BASE_URL", a.api_base)
        os.environ.setdefault("DEEPSEEK_BASE_URL", a.api_base)
    if os.environ.get("LLM_API_KEY"):
        os.environ.setdefault("OPENAI_API_KEY", os.environ["LLM_API_KEY"])
        os.environ.setdefault("DEEPSEEK_API_KEY", os.environ["LLM_API_KEY"])

    rounds = a.rounds if a.rounds is not None else (a.max_iters if a.max_iters is not None else 3)
    out_dir = a.out_dir or _default_out_dir(a.scale)
    num_perf_trials = a.num_perf_trials if a.num_perf_trials is not None else a.repeat

    cfg = RunConfig(
        tasks=task_pattern,
        out_dir=out_dir,
        rounds=rounds,
        arch=a.arch,
        model=a.model,
        temperature=a.temperature,
        timeout_s=a.timeout_s,
        num_correct_trials=a.num_correct_trials,
        num_perf_trials=num_perf_trials,
        warmup=a.warmup,
        precision=a.precision,
        no_llm=a.no_llm,
        seed_reference=a.seed_reference,
        verbose=a.verbose,
    )
    RadioForgeRunner(cfg).run()


if __name__ == "__main__":
    main()
