from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def maybe_profile_radio_task(
    *,
    repo_root: Path,
    candidate_task: Path,
    scale: str,
    segment_profile: str,
    fixture: str | None = None,
    fixture_profile: str | None = None,
    require_fixture: bool = False,
    cuda_visible_devices: str,
    out_dir: Path,
    timeout_s: int = 600,
) -> Optional[str]:
    ncu = shutil.which("ncu")
    if not ncu:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "ncu_stdout.txt"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    cmd = [
        ncu,
        "--set",
        "default",
        "--target-processes",
        "all",
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
        "1",
        "--repeat",
        "1",
    ]
    if fixture:
        cmd.extend(["--fixture", fixture])
    if fixture_profile:
        cmd.extend(["--fixture-profile", fixture_profile])
    if require_fixture:
        cmd.append("--require-fixture")
    try:
        proc = subprocess.run(cmd, cwd=repo_root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s)
        log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
        return proc.stdout[-12000:]
    except Exception as e:  # noqa: BLE001
        text = f"NCU profiling failed: {e}"
        log_path.write_text(text, encoding="utf-8")
        return text
