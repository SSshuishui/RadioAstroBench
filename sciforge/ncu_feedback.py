from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def maybe_ncu_profile_radio(repo_root: Path, task: Path, scale: str, segment_profile: str, cuda_visible_devices: str, out_dir: Path, timeout_s: int = 600) -> Optional[str]:
    ncu = shutil.which("ncu")
    if not ncu:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", ""); env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    cmd = [ncu, "--set", "default", "--target-processes", "all", sys.executable, "-m", "radio_astronomy_cuda_bench.run_smoke", "--task", str(task), "--scale", scale, "--segment-profile", segment_profile, "--warmup", "1", "--repeat", "1", "--no-identity"]
    try:
        proc = subprocess.run(cmd, cwd=repo_root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s)
        text = proc.stdout
    except Exception as e:  # noqa: BLE001
        text = f"NCU profiling failed: {e}"
    (out_dir / "ncu_stdout.txt").write_text(text, encoding="utf-8", errors="replace")
    return text[-12000:]
