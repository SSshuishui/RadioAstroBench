from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

METRICS = ",".join([
    "sm__cycles_active.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__registers_per_thread",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
])


def profile_script(script: Path, out_csv: Path, kernel_names: Optional[List[str]] = None, repeat: int = 20, cwd: Optional[Path] = None) -> Path:
    ncu = shutil.which("ncu")
    if not ncu:
        raise RuntimeError("ncu not found in PATH")
    cmd = [
        ncu, "--csv", "--page=raw", "--kernel-name-base=demangled",
        "--target-processes=all", "--replay-mode=kernel", "--profile-from-start=on",
        f"--log-file={out_csv}", f"--metrics={METRICS}", "--launch-skip=0", "--launch-count=20",
    ]
    if kernel_names:
        names = sorted({x for x in kernel_names if x})
        if names:
            pattern = "|".join(re.escape(x) for x in names)
            cmd.append(f"--kernel-name=::regex:^({pattern})(\\(|$)")
    cmd += [sys.executable, str(script), "--repeat", str(repeat)]
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-8000:])
    return out_csv


def load_csv_text(csv_path: Path, max_chars: int = 12000) -> str:
    if not csv_path.exists():
        return "(NCU CSV not found)"
    txt = csv_path.read_text(encoding="utf-8", errors="ignore")
    return txt[-max_chars:]


def metrics_to_prompt(csv_text: str) -> str:
    return "Here are Nsight Compute CSV metrics for the candidate kernels:\n" + csv_text[-12000:]


def write_radio_ncu_driver(path: Path, candidate_task: Path, scale: str, segment_profile: str, warmup: int, repeat: int):
    path.write_text(f'''from __future__ import annotations\nimport sys\nfrom radio_astronomy_cuda_bench.run_smoke import main\n\nif __name__ == "__main__":\n    rep = "20"\n    if "--repeat" in sys.argv:\n        rep = sys.argv[sys.argv.index("--repeat") + 1]\n    sys.argv = ["run_smoke", "--task", r"{candidate_task}", "--scale", "{scale}", "--segment-profile", "{segment_profile}", "--warmup", "{warmup}", "--repeat", rep, "--no-identity"]\n    main()\n''', encoding="utf-8")


def write_kernelbench_ncu_driver(path: Path, ref_task: Path, candidate_task: Path, warmup: int, repeat: int, device: int, tol: float):
    path.write_text(f'''from __future__ import annotations\nimport sys\nfrom pathlib import Path\nfrom baselines.cudaforge.compile_and_run import compare_and_bench\n\nif __name__ == "__main__":\n    rep = {repeat}\n    if "--repeat" in sys.argv:\n        rep = int(sys.argv[sys.argv.index("--repeat") + 1])\n    compare_and_bench(ref_py=Path(r"{ref_task}"), test_py=Path(r"{candidate_task}"), device_idx={device}, warmup={warmup}, repeat=rep, tol={tol})\n''', encoding="utf-8")
