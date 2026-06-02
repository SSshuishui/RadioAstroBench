from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    # .../radio_astronomy_cuda_bench/configs/real_fixtures.py -> repo root
    return Path(__file__).resolve().parents[2]


def _data_root() -> Path:
    # Parent directory that contains profile directories, e.g.
    #   <repo>/radio_astro_data/nside512_day1_10m_ring
    return Path(os.environ.get("RKB_RADIO_ASTRO_DATA_ROOT", _repo_root() / "radio_astro_data")).resolve()


DEFAULT_PROFILE = "nside512_day1_10m_ring"

# Profile name -> profile directory.  RKB_REAL_FIXTURE_ROOT overrides the selected
# profile root directly, useful for quick testing on a non-standard directory.
def profile_root(profile: str = DEFAULT_PROFILE) -> Path:
    direct = os.environ.get("RKB_REAL_FIXTURE_ROOT")
    if direct:
        return Path(direct).resolve()
    return (_data_root() / profile).resolve()


# These tasks have real radio astronomy fixtures.  Tasks not listed here should
# still run with synthetic inputs scaled by --scale, e.g. nside512_full.
TASK_TO_FIXTURE = {
    # ------------------------- level2 / WS -------------------------
    "level2/02-ws-visibility-forward": "ws/visibility_seg00.pt",
    "level2/03-ws-recon-representative": "ws/recon_rep_seg00.pt",
    "level2/04-ws-recon-grid-average": "ws/recon_grid_average_seg00.pt",

    # ------------------------- level2 / 3D -------------------------
    "level2/05-3d-visibility-forward": "3d/visibility_seg00.pt",
    "level2/06-3d-recon-direct": "3d/recon_direct_seg00.pt",
    "level2/07-3d-phase-reduce": "3d/phase_reduce_seg00.pt",
    "level2/08-3d-pair-weight": "3d/pair_weight_seg00.pt",
    "level2/09-3d-task-flags": "3d/task_flags_seg00.pt",
    "level2/10-3d-recon-tasklist-vv": "3d/recon_tasklist_vv_seg00.pt",
    "level2/11-3d-recon-tasklist-mixed": "3d/recon_tasklist_mixed_seg00.pt",

    # ------------------------- level3 / WS -------------------------
    "level3/01-ws-day-visibility-forward": "manifests/ws_visibility_all10.json",
    "level3/02-ws-day-recon-representative": "manifests/ws_recon_rep_all10.json",

    # ------------------------- level3 / 3D -------------------------
    "level3/03-3d-day-visibility-forward": "manifests/3d_visibility_all10.json",
    "level3/04-3d-day-recon-direct": "manifests/3d_recon_direct_all10.json",
}


def task_id_from_path(task_path: str | Path) -> str:
    p = Path(task_path)
    # Works for radio_bench/level2/xx.py and copied candidate_task.py only if a
    # task_id is provided separately by the caller.  For normal task files, this
    # returns level/stem.
    if p.parent.name.startswith("level"):
        return f"{p.parent.name}/{p.stem}"
    # Candidate files may not live under radio_bench.  Try to recover from an
    # embedded original path if callers pass a task id; otherwise return stem.
    return p.stem


def resolve_fixture_for_task_id(task_id: str, profile: str = DEFAULT_PROFILE, must_exist: bool = False) -> Optional[str]:
    rel = TASK_TO_FIXTURE.get(task_id)
    if rel is None:
        return None
    path = profile_root(profile) / rel
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Fixture for {task_id} not found: {path}")
    return str(path)


def resolve_fixture(task_path: str | Path, profile: str = DEFAULT_PROFILE, must_exist: bool = False) -> Optional[str]:
    task_id = task_id_from_path(task_path)
    return resolve_fixture_for_task_id(task_id, profile=profile, must_exist=must_exist)


def configured_task_ids() -> list[str]:
    return sorted(TASK_TO_FIXTURE.keys())


def configured_task_paths(repo_root: str | Path | None = None) -> list[str]:
    root = Path(repo_root) if repo_root is not None else _repo_root()
    paths: list[str] = []
    for task_id in configured_task_ids():
        level, stem = task_id.split("/", 1)
        paths.append(str(root / "radio_bench" / level / f"{stem}.py"))
    return paths
