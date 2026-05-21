from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import torch


def _load_module(path: Path):
    name = "our_eval_" + re.sub(r"[^0-9A-Za-z_]", "_", path.stem)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flatten_outputs(x: Any) -> List[torch.Tensor]:
    if isinstance(x, torch.Tensor): return [x]
    if isinstance(x, (list, tuple)):
        out: List[torch.Tensor] = []
        for y in x: out.extend(_flatten_outputs(y))
        return out
    if isinstance(x, dict):
        out: List[torch.Tensor] = []
        for k in sorted(x): out.extend(_flatten_outputs(x[k]))
        return out
    raise TypeError(f"Unsupported output type: {type(x)}")


def _compare(ref: Any, cand: Any, atol: float = 1e-3, rtol: float = 1e-3) -> dict:
    a, b = _flatten_outputs(ref), _flatten_outputs(cand)
    if len(a) != len(b): return {"passed": False, "reason": f"output count mismatch {len(a)} vs {len(b)}"}
    ok, details = True, []
    for i, (x, y) in enumerate(zip(a, b)):
        if x.shape != y.shape or x.dtype != y.dtype:
            details.append({"index": i, "passed": False, "reason": f"shape/dtype mismatch {x.shape}/{x.dtype} vs {y.shape}/{y.dtype}"}); ok = False; continue
        if x.dtype.is_floating_point or x.dtype.is_complex:
            xf, yf = x.detach().float(), y.detach().float()
            diff = (xf - yf).abs()
            max_abs = float(diff.max().item()) if diff.numel() else 0.0
            denom = xf.abs().clamp_min(1e-12)
            max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
            passed = bool(torch.allclose(xf, yf, atol=atol, rtol=rtol, equal_nan=True))
            details.append({"index": i, "passed": passed, "max_abs": max_abs, "max_rel": max_rel}); ok = ok and passed
        else:
            ne = int((x != y).sum().item())
            details.append({"index": i, "passed": ne == 0, "num_mismatch": ne}); ok = ok and ne == 0
    return {"passed": ok, "details": details}


def _cuda_time_ms(model, inputs: Sequence[Any], warmup: int, repeat: int) -> float:
    with torch.no_grad():
        for _ in range(warmup): _ = model(*inputs)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(repeat): _ = model(*inputs)
        e.record(); torch.cuda.synchronize()
    return float(s.elapsed_time(e) / max(1, repeat))


@dataclass
class EvalResult:
    ok: bool
    result_json: Optional[dict]
    stdout: str
    stderr: str
    returncode: int
    def failure_text(self) -> str:
        parts = [f"returncode={self.returncode}"]
        if self.result_json is not None: parts.append("RESULT_JSON:\n" + json.dumps(self.result_json, indent=2)[-12000:])
        if self.stdout: parts.append("STDOUT:\n" + self.stdout[-12000:])
        if self.stderr: parts.append("STDERR:\n" + self.stderr[-12000:])
        return "\n\n".join(parts)


def evaluate_radio(repo_root: Path, candidate_task: Path, scale: str, segment_profile: str, warmup: int, repeat: int, timeout_s: int, cuda_visible_devices: str, json_path: Path) -> EvalResult:
    env = os.environ.copy(); env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", ""); env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9"); env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    cmd = [sys.executable, "-m", "radio_astronomy_cuda_bench.run_smoke", "--task", str(candidate_task), "--scale", scale, "--segment-profile", segment_profile, "--warmup", str(warmup), "--repeat", str(repeat), "--json", str(json_path)]
    proc = subprocess.run(cmd, cwd=repo_root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
    result = None
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8")); result = data[0] if isinstance(data, list) and data else data
        except Exception: result = None
    corr = result.get("identity_correctness") if isinstance(result, dict) else None
    ok = proc.returncode == 0 and isinstance(corr, dict) and bool(corr.get("passed"))
    return EvalResult(ok, result, proc.stdout, proc.stderr, proc.returncode)


def _evaluate_kernelbench_inprocess(task_path: Path, warmup: int, repeat: int, atol: float, rtol: float) -> dict:
    mod = _load_module(task_path)
    init_inputs = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    inputs = mod.get_inputs()
    def to_cuda(x):
        if isinstance(x, torch.Tensor): return x.cuda()
        if isinstance(x, (list, tuple)): return type(x)(to_cuda(v) for v in x)
        if isinstance(x, dict): return {k: to_cuda(v) for k, v in x.items()}
        return x
    init_inputs, inputs = to_cuda(init_inputs), to_cuda(inputs)
    base, cand = mod.Model(*init_inputs).cuda().eval(), mod.ModelNew(*init_inputs).cuda().eval()
    with torch.no_grad(): ref, out = base(*inputs), cand(*inputs)
    corr = _compare(ref, out, atol=atol, rtol=rtol)
    base_ms, cand_ms = _cuda_time_ms(base, inputs, warmup, repeat), _cuda_time_ms(cand, inputs, warmup, repeat)
    return {"task_path": str(task_path), "baseline_ms": base_ms, "candidate_ms": cand_ms, "candidate_speedup": base_ms / cand_ms if cand_ms > 0 else None, "correctness": corr}


def evaluate_kernelbench(repo_root: Path, candidate_task: Path, warmup: int, repeat: int, timeout_s: int, cuda_visible_devices: str, json_path: Path, atol: float, rtol: float) -> EvalResult:
    env = os.environ.copy(); env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", ""); env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9"); env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    code = "from pathlib import Path; import json; from our_method.benchmark_eval import _evaluate_kernelbench_inprocess; res=_evaluate_kernelbench_inprocess(Path(%r), %d, %d, %g, %g); Path(%r).write_text(json.dumps(res, indent=2), encoding='utf-8'); print(json.dumps(res, indent=2))" % (str(candidate_task), warmup, repeat, atol, rtol, str(json_path))
    proc = subprocess.run([sys.executable, "-c", code], cwd=repo_root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
    result = None
    if json_path.exists():
        try: result = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception: result = None
    ok = proc.returncode == 0 and isinstance(result, dict) and bool(result.get("correctness", {}).get("passed"))
    return EvalResult(ok, result, proc.stdout, proc.stderr, proc.returncode)
