from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional



def _try_resolve_fixture_for_original_task(
    *,
    original_task_path: Path | None,
    fixture_profile: str | None,
) -> str | None:
    """Resolve real fixture using the original radio_bench task path.

    LLM candidates are saved under baselines/llm_direct/runs/.../candidate_task.py.
    If run_bench resolves fixtures from that candidate path, the configured real-data
    task mapping will not match and the benchmark silently falls back to synthetic
    input. This helper resolves the fixture before launching run_bench, using the
    original radio_bench task path, and passes it as an explicit --fixture.
    """
    if not original_task_path or not fixture_profile:
        return None
    try:
        from radio_astronomy_cuda_bench.configs.real_fixtures import resolve_fixture
    except Exception:
        return None
    try:
        resolved = resolve_fixture(original_task_path, profile=fixture_profile)
    except TypeError:
        # Older resolver signatures should still accept only task_path + profile.
        try:
            resolved = resolve_fixture(str(original_task_path), profile=fixture_profile)
        except Exception:
            return None
    except Exception:
        return None
    if not resolved:
        return None
    try:
        return str(Path(resolved))
    except Exception:
        return str(resolved)


def _env(repo_root: Path, cuda_visible_devices: Optional[str] = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["TORCH_CUDA_ARCH_LIST"] = os.environ.get("RKB_CUDA_ARCH_LIST", "8.9")
    env.setdefault("CC", "/usr/bin/gcc")
    env.setdefault("CXX", "/usr/bin/g++")
    env.setdefault("CUDAHOSTCXX", "/usr/bin/g++")
    env.setdefault("NVCC_APPEND_FLAGS", "-allow-unsupported-compiler")
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    return env


def evaluate_candidate(
    *,
    repo_root: Path,
    task_path: Path,
    original_task_path: Path | None = None,
    scale: str,
    segment_profile: str = "all10",
    fixture: str | None = None,
    fixture_profile: str | None = None,
    warmup: int = 3,
    repeat: int = 5,
    timeout_s: int = 600,
    cuda_visible_devices: str | None = None,
    json_path: Path | None = None,
) -> dict[str, Any]:
    if json_path is None:
        json_path = task_path.parent / "bench_result.json"

    # Important: for LLM candidates, task_path points to candidate_task.py under
    # baselines/llm_direct/runs, not the original radio_bench task. Resolve the
    # real fixture using original_task_path and pass it explicitly.
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
        str(task_path),
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

    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=_env(repo_root, cuda_visible_devices),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
    )

    result_json = None
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(payload, list) and payload:
                result_json = payload[0]
            else:
                result_json = payload
        except Exception:
            result_json = None

    ok = proc.returncode == 0
    if isinstance(result_json, dict):
        # LLM-direct must validate the actual appended candidate, not identity_* harness fields.
        correctness = result_json.get("candidate_correctness")
        if correctness is not None:
            ok = ok and bool(correctness.get("passed"))
        elif "error" in result_json:
            ok = False
        else:
            ok = False

    return {
        "ok": ok,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "bench_result": result_json,
        "json_path": str(json_path),
        "cmd": cmd,
        "auto_resolved_fixture": auto_resolved_fixture,
        "effective_fixture": effective_fixture,
    }
