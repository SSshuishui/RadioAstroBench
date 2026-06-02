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


def _try_resolve_fixture_for_original_task(
    *,
    original_task_path: Path | None,
    fixture_profile: str | None,
) -> str | None:
    """Resolve real-data fixtures with the original radio_bench task path.

    CudaForge writes candidates into baselines/cudaforge/runs/..., so resolving
    fixtures from the candidate path would miss the task-id mapping and silently
    fall back to synthetic inputs. This keeps fixture resolution tied to the original benchmark task.
    """
    if not original_task_path or not fixture_profile:
        return None
    try:
        from radio_astronomy_cuda_bench.configs.real_fixtures import resolve_fixture
        resolved = resolve_fixture(original_task_path, profile=fixture_profile)
    except Exception:
        return None
    return str(resolved) if resolved else None


def _env(repo_root: Path, device: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["TORCH_CUDA_ARCH_LIST"] = os.environ.get("RKB_CUDA_ARCH_LIST", "8.9")
    env.setdefault("CC", "/usr/bin/gcc")
    env.setdefault("CXX", "/usr/bin/g++")
    env.setdefault("CUDAHOSTCXX", "/usr/bin/g++")
    env.setdefault("NVCC_APPEND_FLAGS", "-allow-unsupported-compiler")
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    return env


def _score_from_radio_metrics(metrics: Dict[str, Any]) -> float:
    v = metrics.get("candidate_speedup")
    if isinstance(v, (int, float)):
        return float(v)
    base = metrics.get("baseline_ms")
    cand = metrics.get("candidate_ms")
    if isinstance(base, (int, float)) and isinstance(cand, (int, float)) and cand > 0:
        return float(base) / float(cand)
    # Backward compatibility for older harness outputs only.
    old = metrics.get("identity_speedup")
    if isinstance(old, (int, float)):
        return float(old)
    base = metrics.get("baseline_ms")
    cand = metrics.get("identity_candidate_ms")
    if isinstance(base, (int, float)) and isinstance(cand, (int, float)) and cand > 0:
        return float(base) / float(cand)
    return float("-inf")


def evaluate_radio(
    *,
    repo_root: Path,
    candidate_task: Path,
    original_task_path: Path | None = None,
    scale: str,
    segment_profile: str,
    fixture: str | None = None,
    fixture_profile: str | None = None,
    require_fixture: bool = False,
    warmup: int,
    repeat: int,
    device: int,
    timeout_s: int,
    json_path: Path,
) -> EvalOutcome:
    effective_fixture = fixture
    auto_resolved_fixture = None
    if not effective_fixture and fixture_profile and original_task_path is not None:
        auto_resolved_fixture = _try_resolve_fixture_for_original_task(
            original_task_path=original_task_path,
            fixture_profile=fixture_profile,
        )
        if auto_resolved_fixture:
            effective_fixture = auto_resolved_fixture

    cmd = [
        sys.executable, "-m", "radio_astronomy_cuda_bench.run_bench",
        "--task", str(candidate_task),
        "--scale", scale,
        "--segment-profile", segment_profile,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--json", str(json_path),
    ]
    if effective_fixture:
        cmd.extend(["--fixture", effective_fixture])
    if fixture_profile:
        cmd.extend(["--fixture-profile", fixture_profile])
    if require_fixture:
        cmd.append("--require-fixture")

    proc = subprocess.run(cmd, cwd=str(repo_root), env=_env(repo_root, device), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
    metrics: Dict[str, Any] = {"runnable": False, "message": "no result json"}
    ok = False
    score = float("-inf")
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            item = data[0] if isinstance(data, list) and data else data
            metrics = dict(item)
            metrics["auto_resolved_fixture"] = auto_resolved_fixture
            metrics["effective_fixture"] = effective_fixture
            # Prefer the current candidate_* fields. identity_* is kept only for old JSONs.
            corr = metrics.get("candidate_correctness") or metrics.get("identity_correctness") or metrics.get("correctness") or {}
            ok = bool(corr.get("passed", False)) and "error" not in metrics and not metrics.get("skipped")
            metrics["runnable"] = ok
            if ok:
                score = _score_from_radio_metrics(metrics)
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
