"""Day-1, 10 MHz segment presets.

The nside=4096 values are copied from the user's real 2x4090 day-1 log.
The nside=512 values are placeholders scaled from nside=4096 and MUST be
replaced by fixtures/logs dumped from the real nside=512 run before formal
experiments.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Dict, List

COSPHI_10M = -0.522349
LAMDA_10M = 30.0
THETA_10M = 1.02119
PHI_10M = 2.1204


@dataclass(frozen=True)
class SegmentConfig:
    seg_id: int
    t0: int
    segN: int
    segN_half: int
    fa: float
    fb: float
    nuniq_halfgrid: int
    RES: int
    u_min: float | None = None
    u_max: float | None = None
    v_min: float | None = None
    v_max: float | None = None
    w_min: float | None = None
    w_max: float | None = None


DAY1_10M_NSIDES_4096: List[SegmentConfig] = [
    SegmentConfig(1, 0,     390936, 195468, -0.00059748, 0.577127, 25358, 15397, -364.111, 364.111, -306.517, 306.517, -176.939, 176.939),
    SegmentConfig(2, 6981,  390936, 195468, -0.00100829, 0.577089, 27217, 15397, -384.693, 384.693, -302.307, 302.307, -174.610, 174.610),
    SegmentConfig(3, 13962, 390936, 195468, -0.00181886, 0.577126, 29919, 15397, -425.750, 425.750, -377.623, 377.623, -217.983, 217.983),
    SegmentConfig(4, 20943, 390936, 195468, -0.00302034, 0.577080, 33883, 15397, -487.286, 487.286, -368.689, 368.689, -212.018, 212.018),
    SegmentConfig(5, 27924, 390936, 195468, -0.00360893, 0.577239, 38268, 15399, -458.004, 458.004, -448.768, 448.768, -259.074, 259.074),
    SegmentConfig(6, 34905, 390936, 195468, -0.00426615, 0.577160, 38481, 15397, -548.926, 548.926, -484.334, 484.334, -279.608, 279.608),
    SegmentConfig(7, 41886, 390936, 195468, -0.00503643, 0.577091, 40311, 15397, -589.986, 589.986, -516.063, 516.063, -297.542, 297.542),
    SegmentConfig(8, 48867, 390936, 195468, -0.00597092, 0.577207, 46000, 15399, -613.254, 613.254, -555.467, 555.467, -320.675, 320.675),
    SegmentConfig(9, 55848, 390936, 195468, -0.00681376, 0.577207, 49205, 15399, -623.124, 623.124, -591.035, 591.035, -341.210, 341.210),
    SegmentConfig(10,62829, 390936, 195468, -0.00745090, 0.577045, 47525, 15397, -713.162, 713.162, -592.698, 592.698, -340.378, 340.378),
]


def _make_nside512_placeholders() -> List[SegmentConfig]:
    """Create temporary nside=512 segment presets.

    These are scaled only for development.  Replace them with real day-1 nside=512
    fixture metadata before reporting benchmark numbers.
    """
    out: List[SegmentConfig] = []
    # Pixel grid resolution is about 1/8 of nside4096. Occupied cells roughly
    # scale by area; clamp to avoid extremely tiny synthetic workloads.
    for s in DAY1_10M_NSIDES_4096:
        out.append(SegmentConfig(
            seg_id=s.seg_id,
            t0=s.t0,
            segN=s.segN,
            segN_half=s.segN_half,
            fa=s.fa,
            fb=s.fb,
            nuniq_halfgrid=max(256, int(round(s.nuniq_halfgrid / 64.0))),
            RES=max(256, int(round(s.RES / 8.0))),
            u_min=s.u_min, u_max=s.u_max,
            v_min=s.v_min, v_max=s.v_max,
            w_min=s.w_min, w_max=s.w_max,
        ))
    return out


DAY1_10M_NSIDES_512_PLACEHOLDER: List[SegmentConfig] = _make_nside512_placeholders()


SEGMENT_PRESETS: Dict[str, List[SegmentConfig]] = {
    "smoke": [SegmentConfig(1, 0, 390936, 195468, 0.08, -0.05, 256, 512)],
    "nside512_full": DAY1_10M_NSIDES_512_PLACEHOLDER,
    "nside4096_full": DAY1_10M_NSIDES_4096,
}


def get_segments(scale: str, profile: str = "all10") -> List[SegmentConfig]:
    if scale not in SEGMENT_PRESETS:
        raise KeyError(f"No day1-10M segment preset for scale={scale!r}")
    segs = SEGMENT_PRESETS[scale]
    if profile in ("all", "all10", "day1"):
        return list(segs)
    if profile.startswith("seg"):
        sid = int(profile.replace("seg", ""))
        return [s for s in segs if s.seg_id == sid]
    raise KeyError("segment profile must be all10 or seg<id>, e.g. seg5")


def segment_to_dict(s: SegmentConfig) -> dict:
    return asdict(s)
