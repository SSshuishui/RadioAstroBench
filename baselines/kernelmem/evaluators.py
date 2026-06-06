from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import torch


def _load_module(path: Path):
    name = "km_task_" + re.sub(r"[^0-9A-Za-z_]", "_", path.stem)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flatten_outputs(x: Any) -> List[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (list, tuple)):
        out: List[torch.Tensor] = []
        for y in x:
            out.extend(_flatten_outputs(y))
        return out
    if isinstance(x, dict):
        out: List[torch.Tensor] = []
        for key in sorted(x.keys()):
            out.extend(_flatten_outputs(x[key]))
        return out
    raise TypeError(f"Unsupported output type: {type(x)}")


def _compare_outputs(ref: Any, cand: Any, atol: float = 1e-3, rtol: float = 1e-3) -> dict:
    a = _flatten_outputs(ref)
    b = _flatten_outputs(cand)
    if len(a) != len(b):
        return {"passed": False, "reason": f"output count mismatch {len(a)} vs {len(b)}"}
    details = []
    ok = True
    for i, (x, y) in enumerate(zip(a, b)):
        if x.shape != y.shape or x.dtype != y.dtype:
            details.append({"index": i, "passed": False, "reason": f"shape/dtype mismatch {x.shape}/{x.dtype} vs {y.shape}/{y.dtype}"})
            ok = False
            continue
        if x.dtype.is_floating_point or x.dtype.is_complex:
            xf = x.detach().float()
            yf = y.detach().float()
            diff = (xf - yf).abs()
            max_abs = float(diff.max().item()) if diff.numel() else 0.0
            denom = xf.abs().clamp_min(1e-12)
            max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
            passed = bool(torch.allclose(xf, yf, atol=atol, rtol=rtol, equal_nan=True))
            details.append({"index": i, "passed": passed, "max_abs": max_abs, "max_rel": max_rel})
            ok = ok and passed
        else:
            ne = int((x != y).sum().item())
            details.append({"index": i, "passed": ne == 0, "num_mismatch": ne})
            ok = ok and (ne == 0)
    return {"passed": ok, "details": details}


def _tensor_summary(t: torch.Tensor) -> dict:
    d = {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)}
    if t.numel() == 0:
        return d
    if t.dtype.is_floating_point or t.dtype.is_complex:
        tf = t.detach().float()
        d.update({
            "min": float(torch.nan_to_num(tf, nan=0.0, posinf=0.0, neginf=0.0).min().item()),
            "max": float(torch.nan_to_num(tf, nan=0.0, posinf=0.0, neginf=0.0).max().item()),
            "mean": float(torch.nan_to_num(tf, nan=0.0, posinf=0.0, neginf=0.0).mean().item()),
            "has_nan": bool(torch.isnan(tf).any().item()),
            "has_inf": bool(torch.isinf(tf).any().item()),
        })
    else:
        # Some PyTorch builds do not implement min/max for uint32 on CUDA.
        tc = t.detach().to(torch.int64)
        d.update({"min": int(tc.min().item()), "max": int(tc.max().item()), "mean": float(tc.float().mean().item())})
    return d


def _cuda_time_ms(model, inputs: Sequence[Any], warmup: int, repeat: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            _ = model(*inputs)
        end.record()
        torch.cuda.synchronize()
    return float(start.elapsed_time(end) / max(1, repeat))


@dataclass
class EvalResult:
    ok: bool
    result_json: Optional[dict]
    stdout: str
    stderr: str
    returncode: int

    def failure_text(self) -> str:
        parts = [f"returncode={self.returncode}"]
        if self.result_json is not None:
            parts.append("RESULT_JSON:\n" + json.dumps(self.result_json, indent=2)[-12000:])
        if self.stdout:
            parts.append("STDOUT:\n" + self.stdout[-12000:])
        if self.stderr:
            parts.append("STDERR:\n" + self.stderr[-12000:])
        return "\n\n".join(parts)


def _try_resolve_fixture_for_original_task(
    *,
    original_task_path: Path | None,
    fixture_profile: str | None,
) -> str | None:
    """Resolve real-data fixtures using the original radio_bench task path.

    KernelMem writes candidates into baselines/kernelmem/runs/..., so resolving
    fixtures from the generated candidate path can miss the task-id mapping.
    """
    if not original_task_path or not fixture_profile:
        return None
    try:
        from radio_astronomy_cuda_bench.configs.real_fixtures import resolve_fixture
        resolved = resolve_fixture(original_task_path, profile=fixture_profile)
    except Exception:
        return None
    return str(resolved) if resolved else None


def _radio_env(repo_root: Path, cuda_visible_devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["TORCH_CUDA_ARCH_LIST"] = os.environ.get("RKB_CUDA_ARCH_LIST", os.environ.get("TORCH_CUDA_ARCH_LIST", "8.9"))
    env.setdefault("CC", "/usr/bin/gcc")
    env.setdefault("CXX", "/usr/bin/g++")
    env.setdefault("CUDAHOSTCXX", "/usr/bin/g++")
    env.setdefault("NVCC_APPEND_FLAGS", "-allow-unsupported-compiler")
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    return env


def _radio_score(result: dict | None) -> float | None:
    if not isinstance(result, dict):
        return None
    v = result.get("candidate_speedup")
    if isinstance(v, (int, float)):
        return float(v)
    base = result.get("baseline_ms")
    cand = result.get("candidate_ms")
    if isinstance(base, (int, float)) and isinstance(cand, (int, float)) and cand > 0:
        return float(base) / float(cand)
    # Backward compatibility for old harness JSONs.
    v = result.get("identity_speedup")
    if isinstance(v, (int, float)):
        return float(v)
    cand = result.get("identity_candidate_ms")
    if isinstance(base, (int, float)) and isinstance(cand, (int, float)) and cand > 0:
        return float(base) / float(cand)
    return None


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
    timeout_s: int,
    cuda_visible_devices: str,
    json_path: Path,
) -> EvalResult:
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
        sys.executable,
        "-m",
        "radio_astronomy_cuda_bench.run_bench",
        "--task",
        str(candidate_task),
        "--scale",
        scale,
        "--segment-profile",
        segment_profile,
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--json",
        str(json_path),
    ]
    if effective_fixture:
        cmd.extend(["--fixture", effective_fixture])
    if fixture_profile:
        cmd.extend(["--fixture-profile", fixture_profile])
    if require_fixture:
        cmd.append("--require-fixture")

    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env=_radio_env(repo_root, cuda_visible_devices),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
    )
    result = None
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            result = data[0] if isinstance(data, list) and data else data
            if isinstance(result, dict):
                result = dict(result)
                result["auto_resolved_fixture"] = auto_resolved_fixture
                result["effective_fixture"] = effective_fixture
                score = _radio_score(result)
                if score is not None:
                    result["score"] = score
        except Exception:
            result = None
    ok = False
    if proc.returncode == 0 and isinstance(result, dict):
        corr = result.get("candidate_correctness") or result.get("identity_correctness") or result.get("correctness") or {}
        ok = bool(isinstance(corr, dict) and corr.get("passed")) and "error" not in result and not result.get("skipped")
        result["runnable"] = ok
    return EvalResult(ok, result, proc.stdout, proc.stderr, proc.returncode)

def evaluate_kernelbench_inprocess(
    *,
    task_path: Path,
    warmup: int,
    repeat: int,
    atol: float,
    rtol: float,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for KernelBench evaluation")
    mod = _load_module(task_path)
    init_inputs = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    inputs = mod.get_inputs()
    def to_cuda(x):
        if isinstance(x, torch.Tensor):
            return x.cuda()
        if isinstance(x, (list, tuple)):
            return type(x)(to_cuda(v) for v in x)
        if isinstance(x, dict):
            return {k: to_cuda(v) for k, v in x.items()}
        return x
    init_inputs = to_cuda(init_inputs)
    inputs = to_cuda(inputs)
    baseline = mod.Model(*init_inputs).cuda().eval()
    candidate_cls = getattr(mod, "ModelNew", None)
    if candidate_cls is None:
        raise RuntimeError("candidate task does not define ModelNew")
    candidate = candidate_cls(*init_inputs).cuda().eval()
    with torch.no_grad():
        ref = baseline(*inputs)
        out = candidate(*inputs)
    correctness = _compare_outputs(ref, out, atol=atol, rtol=rtol)
    baseline_ms = _cuda_time_ms(baseline, inputs, warmup, repeat)
    candidate_ms = _cuda_time_ms(candidate, inputs, warmup, repeat)
    return {
        "task_path": str(task_path),
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "candidate_speedup": baseline_ms / candidate_ms if candidate_ms > 0 else None,
        "correctness": correctness,
        "outputs": [_tensor_summary(t) for t in _flatten_outputs(out)],
    }


def evaluate_kernelbench(
    *,
    repo_root: Path,
    candidate_task: Path,
    warmup: int,
    repeat: int,
    timeout_s: int,
    cuda_visible_devices: str,
    json_path: Path,
    atol: float,
    rtol: float,
) -> EvalResult:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    code = (
        "import json; from pathlib import Path; "
        "from baselines.kernelmem.evaluators import evaluate_kernelbench_inprocess; "
        f"res=evaluate_kernelbench_inprocess(task_path=Path({str(candidate_task)!r}), warmup={warmup}, repeat={repeat}, atol={atol}, rtol={rtol}); "
        f"Path({str(json_path)!r}).write_text(json.dumps(res, indent=2), encoding='utf-8'); "
        "print(json.dumps(res, indent=2))"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=repo_root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
    result = None
    if json_path.exists():
        try:
            result = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            result = None
    ok = proc.returncode == 0 and isinstance(result, dict) and bool(result.get("correctness", {}).get("passed"))
    return EvalResult(ok, result, proc.stdout, proc.stderr, proc.returncode)
