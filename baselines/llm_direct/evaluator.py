from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class EvalResult:
    ok: bool
    result_json: Optional[Dict[str, Any]]
    stdout: str
    stderr: str
    returncode: int

    def failure_text(self) -> str:
        parts = [f"returncode={self.returncode}"]
        if self.stdout:
            parts.append("STDOUT:\n" + self.stdout[-8000:])
        if self.stderr:
            parts.append("STDERR:\n" + self.stderr[-8000:])
        if self.result_json is not None:
            parts.append("RESULT_JSON:\n" + json.dumps(self.result_json, indent=2)[-8000:])
        return "\n\n".join(parts)


def evaluate_candidate(
    *,
    repo_root: Path,
    candidate_task: Path,
    scale: str,
    segment_profile: str,
    warmup: int,
    repeat: int,
    timeout_s: int,
    cuda_visible_devices: str,
    json_path: Path,
) -> EvalResult:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    cmd = [
        sys.executable,
        "-m",
        "radio_astronomy_cuda_bench.run_smoke",
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
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
    )
    result_json: Optional[Dict[str, Any]] = None
    ok = False
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                result_json = data[0]
            elif isinstance(data, dict):
                result_json = data
        except Exception:  # noqa: BLE001
            result_json = None
    if proc.returncode == 0 and result_json is not None:
        correctness = result_json.get("identity_correctness") or result_json.get("candidate_correctness")
        if isinstance(correctness, dict):
            ok = bool(correctness.get("passed"))
        else:
            ok = "error" not in result_json and not result_json.get("skipped", False)
    return EvalResult(
        ok=ok,
        result_json=result_json,
        stdout=proc.stdout,
        stderr=proc.stderr,
        returncode=proc.returncode,
    )
