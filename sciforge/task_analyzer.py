from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


@dataclass
class TaskAnalysis:
    task_path: str
    bench: str
    level: str
    algorithm: str
    role: str
    data_pattern: str
    risk_level: str
    protected_semantics: List[str]
    optimization_knobs: List[str]
    source_signals: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


def _contains(src: str, *keys: str) -> bool:
    low = src.lower()
    return any(k.lower() in low for k in keys)


def analyze_task(task_path: Path, source: str, bench: str) -> TaskAnalysis:
    name = task_path.name.lower()
    level = task_path.parent.name
    algorithm = "3d" if "-3d-" in name or "3d" in name else ("ws" if "-ws-" in name or "ws" in name else "kernelbench")
    signals: List[str] = []

    if _contains(source, "pix2lmn", "pix2ang", "healpix") or any(k in name for k in ["pix2", "healpix", "lmn"]):
        role = "geometry_pixel_mapping"; signals.append("pixel_mapping")
    elif _contains(source, "visibility", "viss", "forward_phase") or "visibility" in name:
        role = "visibility_forward"; signals.append("visibility")
    elif _contains(source, "recon", "adjoint_phase", "cacc") or "recon" in name:
        role = "reconstruction_adjoint"; signals.append("reconstruction")
    elif _contains(source, "task_flags", "tasklist", "mixed", "vv_flags") or "task" in name:
        role = "task_scheduling_load_balance"; signals.append("tasklist")
    elif _contains(source, "pair_weight", "dcf") or "weight" in name:
        role = "weighting_reduction"; signals.append("weighting")
    elif _contains(source, "baseline", "uvw") or "uvw" in name or "baseline" in name:
        role = "baseline_generation"; signals.append("baseline")
    elif _contains(source, "norm", "normalize", "scale", "add_inplace", "compute_avg", "gather"):
        role = "array_helper"; signals.append("array_helper")
    else:
        role = "generic_kernel"; signals.append("generic")

    if _contains(source, "sincos", "cos", "sin", "phase"):
        data_pattern = "phase_trig_heavy"; signals.append("trig")
    elif _contains(source, "shared", "__shared__"):
        data_pattern = "shared_memory_tile"; signals.append("shared_memory")
    elif _contains(source, "float4", "float2"):
        data_pattern = "vector_packed"; signals.append("vector_packed")
    elif _contains(source, "atomic", "reduction", "sum"):
        data_pattern = "reduction"; signals.append("reduction")
    else:
        data_pattern = "elementwise_or_regular"

    protected = []
    if role == "geometry_pixel_mapping":
        protected += ["HEALPix NEST/RING indexing convention", "MATLAB phi/theta convention", "l/m/n direction-cosine definitions"]
    if role == "visibility_forward":
        protected += ["visibility phase formula", "blockage/occultation checks", "half-baseline symmetry"]
    if role == "reconstruction_adjoint":
        protected += ["adjoint phase formula", "visibility weighting", "blockage checks", "output accumulation semantics"]
    if role == "baseline_generation":
        protected += ["endpoint order", "28/56 baseline symmetry", "uvw scaling by wavelength"]
    protected += ["input/output shape", "input/output dtype", "deterministic correctness"]

    knobs = []
    if data_pattern == "phase_trig_heavy": knobs += ["use __sincosf/sincos_fast", "phase reuse", "fast math with correctness check"]
    if data_pattern == "shared_memory_tile": knobs += ["TILE size", "shared-memory staging", "coalesced tile loads"]
    if role == "reconstruction_adjoint": knobs += ["CHUNK_KEYS or TILE_BL", "block size", "split all-visible/mixed branch", "cache u/v/w/Viss"]
    if role == "visibility_forward": knobs += ["TILE_PIX", "all-visible/all-hidden/mixed fast paths", "read-only cached B/l/m/n loads"]
    if role == "array_helper": knobs += ["vectorized loads", "fused elementwise operations", "avoid dtype unsupported ops"]
    if role == "task_scheduling_load_balance": knobs += ["task granularity", "prefix/flag compaction", "branch classification costs"]
    knobs += ["launch block size"]

    risk = "high" if role in ["visibility_forward", "reconstruction_adjoint", "geometry_pixel_mapping"] else "medium"
    if bench == "kernelbench": risk = "medium"
    return TaskAnalysis(str(task_path), bench, level, algorithm, role, data_pattern, risk, protected, knobs, signals)
