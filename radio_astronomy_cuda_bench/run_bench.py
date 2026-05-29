from __future__ import annotations

import argparse
import inspect
import json
import os
import traceback
from pathlib import Path
from typing import Any, Optional

# Force a known-good default before task code triggers torch extension compilation.
os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("RKB_CUDA_ARCH_LIST", "8.9")
os.environ.setdefault("CC", "/usr/bin/gcc")
os.environ.setdefault("CXX", "/usr/bin/g++")
os.environ.setdefault("CUDAHOSTCXX", "/usr/bin/g++")
os.environ.setdefault("NVCC_APPEND_FLAGS", "-allow-unsupported-compiler")

import torch

from radio_astronomy_cuda_bench.common import (
    compare_outputs,
    cuda_time_ms,
    estimate_tensor_bytes,
    flatten_outputs,
    human_bytes,
    load_task,
    require_cuda,
    tensor_summary,
)
from radio_astronomy_cuda_bench.configs.scales import get_scale, scale_to_dict
from radio_astronomy_cuda_bench.configs.real_fixtures import resolve_fixture, configured_task_ids


def discover_tasks(root: Path) -> list[Path]:
    return sorted((root / "radio_bench").glob("level*/*.py"))


def resolve_task_paths(root: Path, task: str) -> list[Path]:
    if task == "all":
        return discover_tasks(root)
    p = Path(task)
    if not p.is_absolute():
        p = root / p
    return [p]


def _accepts_kwargs(sig: inspect.Signature) -> bool:
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def call_get_inputs(mod: Any, scale: str, segment_profile: str, fixture: str | None):
    fn = mod.get_inputs
    sig = inspect.signature(fn)
    kwargs = {}
    has_varkw = _accepts_kwargs(sig)
    if "scale" in sig.parameters or has_varkw:
        kwargs["scale"] = scale
    if "segment_profile" in sig.parameters or has_varkw:
        kwargs["segment_profile"] = segment_profile
    if "fixture" in sig.parameters or has_varkw:
        kwargs["fixture"] = fixture
    return fn(**kwargs)


def task_supports_scale(mod: Any, scale: str, *, fixture: str | None) -> bool:
    if fixture:
        # Real fixtures carry scale-specific arrays; allow even if task did not
        # declare SUPPORTED_SCALES in older versions.
        return True
    supported = getattr(mod, "SUPPORTED_SCALES", None)
    if supported is None:
        sig = inspect.signature(mod.get_inputs)
        return scale == "smoke" or "scale" in sig.parameters or _accepts_kwargs(sig)
    return scale in supported


def run_one(task_path: Path, *, scale: str, segment_profile: str, fixture: str | None,
            fixture_profile: str | None, warmup: int, repeat: int, run_identity: bool,
            require_fixture: bool, repo_root: Path) -> dict:
    mod = load_task(task_path)
    task_id = getattr(mod, "TASK_ID", f"{task_path.parent.name}/{task_path.stem}")

    auto_fixture = None
    if fixture is None and fixture_profile:
        resolved = resolve_fixture(task_path, profile=fixture_profile)
        auto_fixture = str(resolved) if resolved is not None else None
        fixture = auto_fixture

    if require_fixture and not fixture:
        return {
            "task_id": task_id,
            "task_path": str(task_path),
            "scale": scale,
            "fixture_profile": fixture_profile or "",
            "fixture": "",
            "input_mode": "skipped_no_fixture",
            "skipped": True,
            "reason": "no real fixture configured for this task",
        }

    if fixture and not Path(fixture).exists():
        if require_fixture:
            return {
                "task_id": task_id,
                "task_path": str(task_path),
                "scale": scale,
                "fixture_profile": fixture_profile or "",
                "fixture": fixture,
                "input_mode": "skipped_missing_fixture",
                "skipped": True,
                "reason": "configured real fixture file does not exist",
            }
        # Auto mode fallback: mapping exists but file missing, so run synthetic scale input.
        fixture = None

    if not task_supports_scale(mod, scale, fixture=fixture):
        return {
            "task_id": task_id,
            "task_path": str(task_path),
            "scale": scale,
            "fixture_profile": fixture_profile or "",
            "fixture": fixture or "",
            "input_mode": "skipped_unsupported_scale",
            "skipped": True,
            "reason": f"task does not support scale={scale}",
            "supported_scales": getattr(mod, "SUPPORTED_SCALES", None),
        }

    cfg = get_scale(scale)
    if ("visibility" in task_id.lower()) and not cfg.run_visibility:
        return {"task_id": task_id, "scale": scale, "skipped": True, "reason": "scale disables visibility tasks"}
    if ("recon" in task_id.lower()) and not cfg.run_reconstruction:
        return {"task_id": task_id, "scale": scale, "skipped": True, "reason": "scale disables reconstruction tasks"}

    inputs = call_get_inputs(mod, scale=scale, segment_profile=segment_profile, fixture=fixture)
    input_bytes = estimate_tensor_bytes([x for x in inputs if isinstance(x, torch.Tensor)])

    model = mod.Model().cuda().eval()
    with torch.no_grad():
        out = model(*inputs)
    summaries = [tensor_summary(t) for t in flatten_outputs(out)]
    output_bytes = estimate_tensor_bytes(flatten_outputs(out))
    base_ms = cuda_time_ms(model, inputs, warmup=warmup, repeat=repeat)

    result = {
        "task_id": task_id,
        "task_path": str(task_path),
        "scale": scale,
        "scale_config": scale_to_dict(cfg),
        "segment_profile": segment_profile,
        "fixture_profile": fixture_profile or "",
        "fixture": fixture or "",
        "input_mode": "real_fixture" if fixture else "synthetic_scale",
        "input_bytes": input_bytes,
        "input_memory": human_bytes(input_bytes),
        "output_bytes": output_bytes,
        "output_memory": human_bytes(output_bytes),
        "baseline_ms": base_ms,
        "outputs": summaries,
    }

    if hasattr(mod, "describe_inputs"):
        try:
            result["input_description"] = mod.describe_inputs(scale=scale, segment_profile=segment_profile, fixture=fixture)
        except Exception as e:
            result["input_description_error"] = str(e)

    if run_identity and hasattr(mod, "ModelNew"):
        cand = mod.ModelNew().cuda().eval()
        with torch.no_grad():
            out_new = cand(*inputs)
        cand_ms = cuda_time_ms(cand, inputs, warmup=warmup, repeat=repeat)
        cand_speedup = base_ms / cand_ms if cand_ms > 0 else None
        cand_correctness = compare_outputs(out, out_new)
        result["candidate_ms"] = cand_ms
        result["candidate_speedup"] = cand_speedup
        result["candidate_correctness"] = cand_correctness
        result["identity_candidate_ms"] = cand_ms
        result["identity_speedup"] = cand_speedup
        result["identity_correctness"] = cand_correctness
    return result


def main():
    parser = argparse.ArgumentParser(description="Run extracted existing-CUDA RadioKernelBench tasks.")
    parser.add_argument("--task", default="all", help="all or path to one task file")
    parser.add_argument("--scale", default="smoke", choices=["smoke", "nside512_full", "nside4096_full", "nside16384_full"])
    parser.add_argument("--segment-profile", default="all10", help="all10 or seg<id>, e.g. seg5")
    parser.add_argument("--fixture", default="", help="explicit fixture .pt/.json path for this task")
    parser.add_argument("--fixture-profile", default="", help="profile name, e.g. nside512_day1_10m_ring; mapped tasks use real data, others use scale-aware synthetic input")
    parser.add_argument("--require-fixture", action="store_true", help="skip tasks that do not have a real fixture; default is auto fallback to synthetic")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--no-identity", action="store_true", help="only run Model, not ModelNew candidate")
    parser.add_argument("--json", default="", help="optional path to write JSON result")
    args = parser.parse_args()

    require_cuda()
    root = Path(__file__).resolve().parents[1]
    task_paths = resolve_task_paths(root, args.task)

    print(f"[run_bench] TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')}", flush=True)
    print(f"[run_bench] CC={os.environ.get('CC')} CXX={os.environ.get('CXX')} CUDAHOSTCXX={os.environ.get('CUDAHOSTCXX')}", flush=True)
    if args.fixture_profile:
        print(f"[run_bench] fixture_profile={args.fixture_profile} configured real fixture tasks: {configured_task_ids()}", flush=True)
        print(f"[run_bench] fixture policy={'require' if args.require_fixture else 'auto: mapped tasks use real data, unmapped tasks use synthetic scale'}", flush=True)

    results = []
    for p in task_paths:
        rel = p.relative_to(root) if p.is_relative_to(root) else p
        print(f"\n=== Running {rel} scale={args.scale} segment={args.segment_profile} fixture_profile={args.fixture_profile} ===", flush=True)
        try:
            r = run_one(
                p,
                scale=args.scale,
                segment_profile=args.segment_profile,
                fixture=args.fixture or None,
                fixture_profile=args.fixture_profile or None,
                warmup=args.warmup,
                repeat=args.repeat,
                run_identity=not args.no_identity,
                require_fixture=args.require_fixture,
                repo_root=root,
            )
            results.append(r)
            print(json.dumps(r, indent=2), flush=True)
        except Exception as e:
            err = {"task_path": str(p), "scale": args.scale, "error": str(e), "traceback": traceback.format_exc()}
            results.append(err)
            print(json.dumps(err, indent=2), flush=True)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
