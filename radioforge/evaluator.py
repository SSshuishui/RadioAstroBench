from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(slots=True)
class EvalResult:
    compiled: bool
    correct: bool
    runtime_ms: float | None
    ref_runtime_ms: float | None
    speedup: float | None
    stdout: str
    stderr: str
    command: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def feedback_text(self, max_chars: int = 6000) -> str:
        d = self.to_dict()
        text = json.dumps({k: v for k, v in d.items() if k not in {"stdout", "stderr"}}, ensure_ascii=False, indent=2)
        trace = (self.stdout + "\n" + self.stderr)[-max_chars:]
        return text + "\n\nTRACE:\n" + trace


def _parse_structured_output(stdout: str, stderr: str, cmd: list[str]) -> EvalResult:
    def b(name: str) -> bool:
        m = re.search(rf"{name}:\s*(True|False)", stdout)
        return bool(m and m.group(1) == "True")

    def f(name: str) -> float | None:
        m = re.search(rf"{name}:\s*([0-9.eE+-]+)", stdout)
        return float(m.group(1)) if m else None

    return EvalResult(
        compiled=b("COMPILED"),
        correct=b("CORRECT"),
        runtime_ms=f("RUNTIME"),
        ref_runtime_ms=f("REF_RUNTIME"),
        speedup=f("SPEEDUP"),
        stdout=stdout,
        stderr=stderr,
        command=cmd,
    )


def _parse_radio_json(stdout: str, stderr: str, cmd: list[str]) -> EvalResult:
    marker = "RADIOFORGE_EVAL_JSON="
    payload = None
    for line in reversed(stdout.splitlines()):
        if line.startswith(marker):
            payload = line[len(marker):]
            break
    if not payload:
        return EvalResult(False, False, None, None, None, stdout, stderr or "Missing RADIOFORGE_EVAL_JSON marker", cmd)
    try:
        d = json.loads(payload)
    except Exception as e:
        return EvalResult(False, False, None, None, None, stdout, stderr + f"\nBad eval JSON: {e}", cmd)
    return EvalResult(
        compiled=bool(d.get("compiled")),
        correct=bool(d.get("correct")),
        runtime_ms=d.get("runtime_ms"),
        ref_runtime_ms=d.get("ref_runtime_ms"),
        speedup=d.get("speedup"),
        stdout=stdout,
        stderr=stderr + ("\n" + d.get("error", "") if d.get("error") else ""),
        command=cmd,
    )


def find_kernelbench(repo_root: Path) -> Path | None:
    candidates = [
        repo_root / "bench" / "kernelbench" / "bench.py",
        repo_root / "AKO4ALL-main" / "bench" / "kernelbench" / "bench.py",
        repo_root / "bench.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    for c in repo_root.rglob("bench/kernelbench/bench.py"):
        return c
    return None


def evaluate_with_kernelbench(
    ref_path: Path,
    solution_path: Path,
    repo_root: Path,
    timeout_s: int,
    precision: str = "float32",
    num_correct_trials: int = 5,
    num_perf_trials: int = 50,
    **_: object,
) -> EvalResult:
    bench = find_kernelbench(repo_root)
    if bench is None:
        return EvalResult(False, False, None, None, None, "", "Cannot find bench/kernelbench/bench.py", [])
    cmd = [
        sys.executable,
        str(bench),
        "--ref",
        str(ref_path),
        "--solution",
        str(solution_path),
        "--precision",
        precision,
        "--num-correct-trials",
        str(num_correct_trials),
        "--num-perf-trials",
        str(num_perf_trials),
        "--verbose",
    ]
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = "8.9"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["OUR_METHOD_ARCH"] = "sm_89"
    try:
        p = subprocess.run(cmd, cwd=str(repo_root), env=env, text=True, capture_output=True, timeout=timeout_s)
        result = _parse_structured_output(p.stdout, p.stderr, cmd)
        if p.returncode != 0 and result.correct:
            result.correct = False
        return result
    except subprocess.TimeoutExpired as e:
        return EvalResult(False, False, None, None, None, e.stdout or "", e.stderr or f"Timeout after {timeout_s}s", cmd)


_RADIO_EVAL_SCRIPT = r'''
import importlib.util, json, math, os, sys, time, traceback
from pathlib import Path

ref_path = Path(sys.argv[1]).resolve()
sol_path = Path(sys.argv[2]).resolve()
correct_trials = int(sys.argv[3])
perf_trials = int(sys.argv[4])
warmup = int(sys.argv[5])

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

try:
    import torch
except Exception as e:
    print("RADIOFORGE_EVAL_JSON=" + json.dumps({"compiled": False, "correct": False, "error": "import torch failed: " + repr(e)}))
    raise SystemExit(0)


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def tree_to_cuda(x):
    if torch.is_tensor(x):
        return x.cuda() if torch.cuda.is_available() else x
    if isinstance(x, (list, tuple)):
        return type(x)(tree_to_cuda(v) for v in x)
    if isinstance(x, dict):
        return {k: tree_to_cuda(v) for k, v in x.items()}
    return x


def clone_tree(x):
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, (list, tuple)):
        return type(x)(clone_tree(v) for v in x)
    if isinstance(x, dict):
        return {k: clone_tree(v) for k, v in x.items()}
    return x


def as_tuple(x):
    if x is None:
        return tuple()
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


def make_model(mod):
    init = as_tuple(mod.get_init_inputs()) if hasattr(mod, "get_init_inputs") else tuple()
    model = mod.Model(*init)
    if hasattr(model, "cuda") and torch.cuda.is_available():
        model = model.cuda()
    if hasattr(model, "eval"):
        model.eval()
    return model


def make_inputs(mod):
    if not hasattr(mod, "get_inputs"):
        raise RuntimeError("task module has no get_inputs(); cannot evaluate radio_bench-style task")
    return as_tuple(tree_to_cuda(mod.get_inputs()))


def close(a, b, atol=1e-3, rtol=1e-3):
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.dtype != b.dtype:
            b = b.to(a.dtype)
        return bool(torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, atol, rtol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], atol, rtol) for k in a.keys())
    try:
        return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))
    except Exception:
        return a == b


def run_model(model, inputs):
    with torch.no_grad():
        return model(*inputs)


def bench(model, mod, n, warmup):
    # Regenerate inputs each run to match KernelBench/radio_bench style while keeping timing simple.
    times = []
    for i in range(warmup + n):
        inputs = make_inputs(mod)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_model(model, inputs)
            end.record()
            torch.cuda.synchronize()
            ms = float(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            run_model(model, inputs)
            ms = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(ms)
    return sum(times) / max(1, len(times))

try:
    ref = load_mod(ref_path, "radioforge_ref_mod")
    sol = load_mod(sol_path, "radioforge_sol_mod")
    ref_model = make_model(ref)
    sol_model = make_model(sol)
    compiled = True

    correct = True
    err = ""
    for _ in range(correct_trials):
        inputs = make_inputs(ref)
        inputs2 = clone_tree(inputs)
        out_ref = run_model(ref_model, inputs)
        out_sol = run_model(sol_model, inputs2)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if not close(out_ref, out_sol):
            correct = False
            err = "output mismatch"
            break

    ref_ms = sol_ms = speedup = None
    if correct:
        ref_ms = bench(ref_model, ref, perf_trials, warmup)
        sol_ms = bench(sol_model, ref, perf_trials, warmup)
        speedup = (ref_ms / sol_ms) if sol_ms and sol_ms > 0 else None
    print("RADIOFORGE_EVAL_JSON=" + json.dumps({
        "compiled": compiled,
        "correct": correct,
        "runtime_ms": sol_ms,
        "ref_runtime_ms": ref_ms,
        "speedup": speedup,
        "error": err,
    }))
except Exception:
    print("RADIOFORGE_EVAL_JSON=" + json.dumps({
        "compiled": False,
        "correct": False,
        "runtime_ms": None,
        "ref_runtime_ms": None,
        "speedup": None,
        "error": traceback.format_exc()[-6000:],
    }))
'''


def evaluate_with_radio_bench(
    ref_path: Path,
    solution_path: Path,
    repo_root: Path,
    timeout_s: int,
    num_correct_trials: int = 5,
    num_perf_trials: int = 5,
    warmup: int = 3,
    **_: object,
) -> EvalResult:
    with tempfile.NamedTemporaryFile("w", suffix="_radioforge_eval.py", delete=False, encoding="utf-8") as f:
        f.write(_RADIO_EVAL_SCRIPT)
        script = f.name
    cmd = [sys.executable, script, str(ref_path), str(solution_path), str(num_correct_trials), str(num_perf_trials), str(warmup)]
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = "8.9"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["OUR_METHOD_ARCH"] = "sm_89"
    try:
        p = subprocess.run(cmd, cwd=str(repo_root), env=env, text=True, capture_output=True, timeout=timeout_s)
        return _parse_radio_json(p.stdout, p.stderr, cmd)
    except subprocess.TimeoutExpired as e:
        return EvalResult(False, False, None, None, None, e.stdout or "", e.stderr or f"Timeout after {timeout_s}s", cmd)
    finally:
        try:
            os.remove(script)
        except OSError:
            pass


def evaluate_auto(ref_path: Path, solution_path: Path, repo_root: Path, **kwargs) -> EvalResult:
    # radio_bench tasks are KernelBench-style Python modules in this project, but the external
    # bench/kernelbench parser is not always present. Use the self-contained radio evaluator first.
    radio_result = evaluate_with_radio_bench(ref_path, solution_path, repo_root, **kwargs)
    if radio_result.compiled or "radio_bench" in str(ref_path):
        return radio_result
    kb = evaluate_with_kernelbench(ref_path, solution_path, repo_root, **kwargs)
    if kb.compiled or kb.correct:
        return kb
    # Return the radio result because it usually has the more informative traceback.
    return radio_result
