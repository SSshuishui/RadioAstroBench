from __future__ import annotations

from typing import Dict, List
from .task_analyzer import TaskAnalysis


DOMAIN_SKILLS: Dict[str, List[str]] = {
    "geometry_pixel_mapping": [
        "Fuse intermediate theta/phi calculation into direct l/m/n when the task already expects l/m/n.",
        "Avoid changing NEST/RING indexing. Keep 0-based/1-based conversions exactly as baseline.",
        "Use float output but preserve double precision for sensitive index-to-angle calculations if baseline does so.",
    ],
    "visibility_forward": [
        "Preserve all-visible/all-hidden/mixed blockage semantics.",
        "Stage sky tile arrays into shared memory only when it improves reuse.",
        "Use __sincosf for phase calculation when tolerated by correctness threshold.",
        "Do not drop mixed-path point visibility checks.",
    ],
    "reconstruction_adjoint": [
        "Cache u/v/w/Viss/pair weights in shared memory for baseline blocks.",
        "Separate all-visible and mixed branches when task flags are available.",
        "Tune TILE_BL/CHUNK_KEYS and block size, but keep output accumulation semantics identical.",
        "Prefer additive local accumulation before writing output.",
    ],
    "baseline_generation": [
        "Compute unique 28 baselines first, then signed counterpart if required.",
        "Pack endpoint data into float4 only if the task output expects it.",
        "Avoid repeated norm computations by reusing inverse norms per endpoint when possible.",
    ],
    "task_scheduling_load_balance": [
        "Keep task id encoding stable.",
        "Reduce divergent classification cost by early-exit all-hidden/all-visible checks.",
        "Do not change task ordering if downstream tasklist kernels depend on it.",
    ],
    "weighting_reduction": [
        "Clamp and finite-check weights exactly as baseline.",
        "Avoid unsupported CUDA uint32 PyTorch reductions in Python-level summaries.",
        "Fuse simple postprocessing where output equivalence is clear.",
    ],
    "array_helper": [
        "Use simple coalesced global reads/writes.",
        "Avoid overengineering kernels dominated by launch overhead.",
        "Use vectorized load/store only when alignment and dtype are guaranteed.",
    ],
    "generic_kernel": [
        "Make a conservative candidate first, then optimize after correctness passes.",
        "Tune block size and memory access pattern without changing semantics.",
    ],
}


def route_skills(analysis: TaskAnalysis, max_items: int = 10) -> List[str]:
    skills = list(DOMAIN_SKILLS.get(analysis.role, DOMAIN_SKILLS["generic_kernel"]))
    if analysis.data_pattern == "phase_trig_heavy":
        skills.append("For phase-heavy kernels, compare __sincosf vs sin/cos and preserve phase sign conventions.")
    if analysis.data_pattern == "shared_memory_tile":
        skills.append("For tile kernels, ensure shared-memory size is compatible with launch config and occupancy.")
    if analysis.algorithm == "3d":
        skills.append("For 3D path, preserve direct 3D phase u*l+v*m+w*n and endpoint-direction conventions.")
    if analysis.algorithm == "ws":
        skills.append("For WS path, preserve local-frame fa/fb projection and half-grid symmetry.")
    return skills[:max_items]
