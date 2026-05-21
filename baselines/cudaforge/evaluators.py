from __future__ import annotations
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class EvalOutcome:
    ok: bool
    score: float
    metrics: Dict[str, Any]
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


def _failure_text(out: EvalOutcome) -> str:
    parts = [f"returncode={out.returncode}", json.dumps(out.metrics, indent=2, ensure_ascii=False)[-6000:]]
    if out.stdout:
        parts.append("STDOUT:\n" + out.stdout[-6000:])
    if out.stderr:
        parts.append("STDERR:\n" + out.stderr[-6000:])
    return "\n\n".join(parts)


def evaluate_radio(
    *,
    repo_root: Path,
    candidate_task: Path,
    scale: str,
    segment_profile: str,
    warmup: int,
    repeat: int,
    device: int,
    timeout_s: int,
    json_path: Path,
) -> EvalOutcome:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    cmd = [
        sys.executable, "-m", "radio_astronomy_cuda_bench.run_smoke",
        "--task", str(candidate_task),
        "--scale", scale,
        "--segment-profile", segment_profile,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--json", str(json_path),
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
    metrics: Dict[str, Any] = {"runnable": False, "message": "no result json"}
    ok = False
    score = float("-inf")
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            item = data[0] if isinstance(data, list) and data else data
            metrics = dict(item)
            corr = metrics.get("identity_correctness") or metrics.get("correctness") or {}
            ok = bool(corr.get("passed", False)) and "error" not in metrics and not metrics.get("skipped")
            metrics["runnable"] = ok
            if ok:
                base = float(metrics.get("baseline_ms", 0.0) or 0.0)
                cand = float(metrics.get("identity_candidate_ms", 0.0) or 0.0)
                score = base / max(1e-9, cand)
                metrics["score"] = score
        except Exception as e:
            metrics = {"runnable": False, "error_type": "ResultParseError", "message": str(e)}
    if proc.returncode != 0 and ok is False:
        metrics.setdefault("error_type", "ProcessError")
        metrics.setdefault("message", (proc.stderr or proc.stdout)[-6000:])
    return EvalOutcome(ok=ok, score=score, metrics=metrics, stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)


def evaluate_kernelbench(
    *,
    ref_task: Path,
    candidate_task: Path,
    warmup: int,
    repeat: int,
    device: int,
    tol: float,
) -> EvalOutcome:
    from .compile_and_run import compare_and_bench
    try:
        res = compare_and_bench(ref_py=ref_task, test_py=candidate_task, device_idx=device, warmup=warmup, repeat=repeat, tol=tol)
        score = float(res["ref_latency_ms"]["avg"]) / max(1e-9, float(res["test_latency_ms"]["avg"]))
        res["runnable"] = True
        res["score"] = score
        return EvalOutcome(ok=True, score=score, metrics=res)
    except Exception as e:
        return EvalOutcome(ok=False, score=float("-inf"), metrics={"runnable": False, "error_type": e.__class__.__name__, "message": str(e)[-12000:]})


def outcome_failure_text(out: EvalOutcome) -> str:
    return _failure_text(out)
