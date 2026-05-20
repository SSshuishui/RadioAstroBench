"""Scale presets for Radio Existing-CUDA Bench.

The benchmark deliberately separates the *search scale* from the
*final-validation scale*:

- nside512_full: full-sky workload for agent search.
- nside4096_full: full-sky workload for final validation.
- nside16384_full: auxiliary / streaming-compatible tasks only; visibility and
  reconstruction tasks should not run this scale in the agent loop.

The values below are safe defaults.  Domain-specific segment metadata such as
nuniq_halfgrid should come from day1_10m_segments.py or exported fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ScaleConfig:
    name: str
    nside: int | None
    npix: int
    role: str
    run_visibility: bool
    run_reconstruction: bool
    description: str


SCALES: Dict[str, ScaleConfig] = {
    "smoke": ScaleConfig(
        name="smoke",
        nside=None,
        npix=8192,
        role="debug",
        run_visibility=True,
        run_reconstruction=True,
        description="Tiny synthetic workload for compile/interface/correctness smoke tests.",
    ),
    "nside512_full": ScaleConfig(
        name="nside512_full",
        nside=512,
        npix=512 * 512 * 12,
        role="agent_search",
        run_visibility=True,
        run_reconstruction=True,
        description="Full-sky nside=512 workload for multi-round agent search.",
    ),
    "nside4096_full": ScaleConfig(
        name="nside4096_full",
        nside=4096,
        npix=4096 * 4096 * 12,
        role="final_validation",
        run_visibility=True,
        run_reconstruction=True,
        description="Full-sky nside=4096 workload for final candidate validation.",
    ),
    "nside16384_full": ScaleConfig(
        name="nside16384_full",
        nside=16384,
        npix=16384 * 16384 * 12,
        role="large_scale_auxiliary",
        run_visibility=False,
        run_reconstruction=False,
        description="Auxiliary/streaming-compatible large-scale workload; no full visibility/reconstruction.",
    ),
}


def get_scale(name: str) -> ScaleConfig:
    try:
        return SCALES[name]
    except KeyError as e:
        valid = ", ".join(SCALES)
        raise KeyError(f"Unknown scale {name!r}. Valid scales: {valid}") from e


def scale_to_dict(cfg: ScaleConfig) -> dict:
    return {
        "name": cfg.name,
        "nside": cfg.nside,
        "npix": cfg.npix,
        "role": cfg.role,
        "run_visibility": cfg.run_visibility,
        "run_reconstruction": cfg.run_reconstruction,
        "description": cfg.description,
    }
