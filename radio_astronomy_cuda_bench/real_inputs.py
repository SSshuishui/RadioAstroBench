from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import torch

from radio_astronomy_cuda_bench.common import make_tile_meta_torch, make_unit_vectors_cuda
from radio_astronomy_cuda_bench.configs.scales import get_scale

DEFAULT_COSPHI = 0.15


def _to_cuda(x: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    return x.to(device=device, non_blocking=False).contiguous()


def load_pt(path: str | Path) -> dict[str, Any]:
    obj = torch.load(Path(path), map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"fixture must be a dict saved by torch.save, got {type(obj)!r}: {path}")
    return obj


def fixture_root_from_path(path: str | Path) -> Path:
    p = Path(path).resolve()
    if p.suffix == ".json":
        # .../<profile>/manifests/name.json
        return p.parent.parent
    # .../<profile>/(ws|3d)/file.pt or .../<profile>/common/file.pt
    if p.parent.name in {"ws", "3d", "common"}:
        return p.parent.parent
    return p.parent


def common_b_path_for_fixture(path: str | Path) -> Path:
    return fixture_root_from_path(path) / "common" / "B_ring.pt"


def load_common_B(path: str | Path, *, n_pix: int, device: str = "cuda") -> torch.Tensor:
    p = common_b_path_for_fixture(path)
    if p.exists():
        obj = load_pt(p)
        B = obj.get("B")
        if B is None:
            raise KeyError(f"{p} does not contain key 'B'")
        B = B.reshape(-1).to(torch.float32)
        if B.numel() < n_pix:
            raise ValueError(f"{p}: B length {B.numel()} < required {n_pix}")
        return _to_cuda(B[:n_pix], device=device)
    # Fallback synthetic B if common/B_ring.pt is not present.
    idx = torch.arange(n_pix, device=device, dtype=torch.float32)
    return (1.0 + 0.05 * torch.sin(idx * 0.001)).contiguous()


@torch.no_grad()
def pix2lmn_ring_torch(nside: int, *, device: str = "cuda", tile_multiple: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate HEALPix RING direction cosines using the same convention as pix2lmn_ring_kernel.

    This is input preparation, not part of timed Model.forward().
    """
    npix = 12 * int(nside) * int(nside)
    if tile_multiple:
        npix = (npix // int(tile_multiple)) * int(tile_multiple)
    ns = int(nside)
    nl4 = 4 * ns
    ncap = 2 * ns * (ns - 1)
    limit_equ = (2 * ns) * (5 * ns + 1)

    ipix1_i = torch.arange(1, npix + 1, device=device, dtype=torch.int64)
    ipix1 = ipix1_i.to(torch.float64)
    z = torch.empty((npix,), device=device, dtype=torch.float64)
    phi = torch.empty((npix,), device=device, dtype=torch.float64)

    north = ipix1_i <= ncap
    equ = (ipix1_i > ncap) & (ipix1_i <= limit_equ)
    south = ~(north | equ)
    pi = math.pi

    if bool(north.any().item()):
        ip = ipix1[north]
        iring = torch.floor(0.5 * (1.0 + torch.sqrt(1.0 + 2.0 * ip))).to(torch.int64)
        iring_f = iring.to(torch.float64)
        iphi = ipix1_i[north] - 2 * iring * (iring - 1)
        z[north] = 1.0 - (iring_f * iring_f) / (3.0 * ns * ns)
        phi[north] = (iphi.to(torch.float64) - 0.5) * pi / (2.0 * iring_f)

    if bool(equ.any().item()):
        ip_i = ipix1_i[equ] - ncap - 1
        iring = (ip_i // nl4) + ns
        iphi = (ip_i % nl4) + 1
        fodd = 0.5 * (1.0 + ((iring + ns) & 1).to(torch.float64))
        z[equ] = ((2 * ns - iring).to(torch.float64)) * 2.0 / (3.0 * ns)
        phi[equ] = (iphi.to(torch.float64) - fodd) * pi / (2.0 * ns)

    if bool(south.any().item()):
        ip = (12 * ns * ns - ipix1_i[south] + 1).to(torch.float64)
        ip_i = (12 * ns * ns - ipix1_i[south] + 1)
        iring = torch.floor(0.5 * (1.0 + torch.sqrt(1.0 + 2.0 * ip))).to(torch.int64)
        iring_f = iring.to(torch.float64)
        iphi = 4 * iring + 1 - (ip_i - 2 * iring * (iring - 1))
        z[south] = -1.0 + (iring_f * iring_f) / (3.0 * ns * ns)
        phi[south] = (iphi.to(torch.float64) - 0.5) * pi / (2.0 * iring_f)

    sintheta = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    l = (sintheta * torch.cos(phi)).to(torch.float32).contiguous()
    m = (sintheta * torch.sin(phi)).to(torch.float32).contiguous()
    n = z.to(torch.float32).contiguous()
    return l, m, n


def lmn_for_scale(scale: str, *, nside: int | None = None, device: str = "cuda", tile_pix: int = 256, seed: int = 0):
    if nside is None:
        cfg = get_scale(scale)
        nside = cfg.nside
        n_pix = (cfg.npix // tile_pix) * tile_pix
    else:
        n_pix = (12 * int(nside) * int(nside) // tile_pix) * tile_pix
    if nside is not None:
        return pix2lmn_ring_torch(int(nside), device=device, tile_multiple=tile_pix)
    return make_unit_vectors_cuda(n_pix, device=device, seed=seed)


def _cosphi(meta: dict[str, Any]) -> float:
    for k in ("cosphi", "cos_phi", "COSPHI_10M"):
        if k in meta:
            try:
                return float(meta[k])
            except Exception:
                pass
    return DEFAULT_COSPHI


def _fa(meta: dict[str, Any], default: float = 0.08) -> float:
    return float(meta.get("fa", default))


def _fb(meta: dict[str, Any], default: float = -0.05) -> float:
    return float(meta.get("fb", default))


def _res_half_du(meta: dict[str, Any]) -> tuple[int, int, float]:
    RES = int(meta.get("RES", meta.get("res", 64)))
    half = int(meta.get("half", RES // 2))
    du = float(meta.get("du", 1.0 / max(1, RES)))
    return RES, half, du


def split_p4(p: torch.Tensor, *, need_invn: bool, device: str = "cuda"):
    p = _to_cuda(p.to(torch.float32), device=device)
    x, y, z = p[:, 0].contiguous(), p[:, 1].contiguous(), p[:, 2].contiguous()
    if need_invn:
        if p.shape[1] >= 4:
            invn = p[:, 3].contiguous()
        else:
            invn = torch.rsqrt(x * x + y * y + z * z + 1e-20).contiguous()
        return x, y, z, invn
    return x, y, z


def load_ws_visibility_fixture(fixture: str | Path, *, scale: str = "nside512_full", tile_pix: int = 256, device: str = "cuda"):
    data = load_pt(fixture)
    meta = dict(data.get("meta", {}))
    nside = int(meta.get("nside") or get_scale(scale).nside or 512)
    l, m, n = lmn_for_scale(scale, nside=nside, device=device, tile_pix=tile_pix, seed=401)
    nm1 = (n - 1.0).contiguous()
    B = load_common_B(fixture, n_pix=l.numel(), device=device)
    tile_meta = make_tile_meta_torch(l, m, n, tile_pix)
    u = _to_cuda(data["u"].to(torch.float32), device=device)
    v = _to_cuda(data["v"].to(torch.float32), device=device)
    w = _to_cuda(data["w"].to(torch.float32), device=device)
    x1, y1, z1, invn1 = split_p4(data["p1"], need_invn=True, device=device)
    x2, y2, z2, invn2 = split_p4(data["p2"], need_invn=True, device=device)
    return [B, l, m, n, nm1, *tile_meta, u, v, w, x1, y1, z1, invn1, x2, y2, z2, invn2, _cosphi(meta)]


def load_3d_visibility_fixture(fixture: str | Path, *, scale: str = "nside512_full", tile_pix: int = 512, device: str = "cuda"):
    data = load_pt(fixture)
    meta = dict(data.get("meta", {}))
    nside = int(meta.get("nside") or get_scale(scale).nside or 512)
    l, m, n = lmn_for_scale(scale, nside=nside, device=device, tile_pix=tile_pix, seed=501)
    nm1 = (n - 1.0).contiguous()
    B = load_common_B(fixture, n_pix=l.numel(), device=device)
    tile_meta = make_tile_meta_torch(l, m, n, tile_pix)
    u = _to_cuda(data["u"].to(torch.float32), device=device)
    v = _to_cuda(data["v"].to(torch.float32), device=device)
    w = _to_cuda(data["w"].to(torch.float32), device=device)
    x1, y1, z1 = split_p4(data["p1"], need_invn=False, device=device)
    x2, y2, z2 = split_p4(data["p2"], need_invn=False, device=device)
    return [B, l, m, n, nm1, *tile_meta, u, v, w, x1, y1, z1, x2, y2, z2, _cosphi(meta)]


def _sibling_visibility_for_recon(path: str | Path) -> Path | None:
    p = Path(path)
    m = re.search(r"recon_direct_seg(\d+)\.pt$", p.name)
    if not m:
        return None
    cand = p.with_name(f"visibility_seg{m.group(1)}.pt")
    return cand if cand.exists() else None


def load_3d_recon_direct_fixture(fixture: str | Path, *, scale: str = "nside512_full", tile_pix: int = 256, device: str = "cuda"):
    data = load_pt(fixture)
    meta = dict(data.get("meta", {}))
    nside = int(meta.get("nside") or get_scale(scale).nside or 512)
    l, m, n = lmn_for_scale(scale, nside=nside, device=device, tile_pix=tile_pix, seed=801)
    tile_meta = make_tile_meta_torch(l, m, n, tile_pix)
    u = _to_cuda(data["u"].to(torch.float32), device=device)
    v = _to_cuda(data["v"].to(torch.float32), device=device)
    w = _to_cuda(data["w"].to(torch.float32), device=device)
    if "p1" in data and "p2" in data:
        p1_src, p2_src = data["p1"], data["p2"]
    else:
        sib = _sibling_visibility_for_recon(fixture)
        if sib is None:
            raise KeyError(f"{fixture} has no p1/p2 and sibling visibility_segXX.pt was not found")
        vdata = load_pt(sib)
        p1_src, p2_src = vdata["p1"], vdata["p2"]
    x1, y1, z1 = split_p4(p1_src, need_invn=False, device=device)
    x2, y2, z2 = split_p4(p2_src, need_invn=False, device=device)
    pair_weight = _to_cuda(data.get("pair_weight_half", torch.ones_like(data["u"])).to(torch.float32), device=device)
    Viss_half = _to_cuda(data["Viss_half"].to(torch.float32), device=device)
    return [l, m, n, *tile_meta, u, v, w, x1, y1, z1, x2, y2, z2, pair_weight, Viss_half, _cosphi(meta)]


def load_ws_recon_rep_fixture(fixture: str | Path, *, scale: str = "nside512_full", tile_pix: int = 256, device: str = "cuda"):
    data = load_pt(fixture)
    meta = dict(data.get("meta", {}))
    nside = int(meta.get("nside") or get_scale(scale).nside or 512)
    l, m, n = lmn_for_scale(scale, nside=nside, device=device, tile_pix=tile_pix, seed=701)
    keys = _to_cuda(data["keys_unique"].to(torch.int32), device=device)
    ugu = _to_cuda(data["ugu_list"].to(torch.float32), device=device)
    vgu = _to_cuda(data["vgu_list"].to(torch.float32), device=device)
    repV = _to_cuda(data["repV"].to(torch.float32), device=device)
    repP1 = _to_cuda(data["repP1"].to(torch.float32), device=device)
    repP2 = _to_cuda(data["repP2"].to(torch.float32), device=device)
    return [l, m, n, keys, ugu, vgu, repV, repP1, repP2, _fa(meta), _fb(meta), _cosphi(meta)]


def load_ws_grid_average_fixture(fixture: str | Path, *, scale: str = "nside512_full", tile_pix: int = 256, device: str = "cuda"):
    data = load_pt(fixture)
    meta = dict(data.get("meta", {}))
    nside = int(meta.get("nside") or get_scale(scale).nside or 512)
    l, m, n = lmn_for_scale(scale, nside=nside, device=device, tile_pix=tile_pix, seed=702)
    keys = _to_cuda(data["keys_unique"].to(torch.int32), device=device)
    viss_avg = _to_cuda(data["repV"].to(torch.float32), device=device)
    RES, half, du = _res_half_du(meta)
    return [l, m, n, keys, viss_avg, RES, half, du, _fa(meta), _fb(meta)]


def manifest_segment_paths(manifest: str | Path, *, segment_profile: str = "all10") -> list[Path]:
    mp = Path(manifest).resolve()
    with mp.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    paths = [(mp.parent.parent / rel).resolve() for rel in obj.get("segments", [])]
    if segment_profile and segment_profile != "all10":
        m = re.fullmatch(r"seg(\d+)", segment_profile)
        if m:
            idx = int(m.group(1))
            return [paths[idx]]
    return paths


def load_day_ws_visibility_manifest(manifest: str | Path, *, scale: str = "nside512_full", segment_profile: str = "all10", tile_pix: int = 256, device: str = "cuda"):
    segs = manifest_segment_paths(manifest, segment_profile=segment_profile)
    if not segs:
        raise ValueError(f"no segments in manifest {manifest}")
    # Use first segment meta to construct common sky input.
    meta = dict(load_pt(segs[0]).get("meta", {}))
    nside = int(meta.get("nside") or get_scale(scale).nside or 512)
    l, m, n = lmn_for_scale(scale, nside=nside, device=device, tile_pix=tile_pix, seed=27)
    nm1 = (n - 1.0).contiguous()
    B = load_common_B(segs[0], n_pix=l.numel(), device=device)
    tile_meta = make_tile_meta_torch(l, m, n, tile_pix)
    lists: list[list[torch.Tensor]] = [[] for _ in range(11)]
    cosphi = _cosphi(meta)
    for seg in segs:
        data = load_pt(seg)
        md = dict(data.get("meta", {})); cosphi = _cosphi(md)
        u = _to_cuda(data["u"].to(torch.float32), device=device)
        v = _to_cuda(data["v"].to(torch.float32), device=device)
        w = _to_cuda(data["w"].to(torch.float32), device=device)
        x1, y1, z1, invn1 = split_p4(data["p1"], need_invn=True, device=device)
        x2, y2, z2, invn2 = split_p4(data["p2"], need_invn=True, device=device)
        vals = [u, v, w, x1, y1, z1, invn1, x2, y2, z2, invn2]
        for dst, val in zip(lists, vals):
            dst.append(val)
    return [B, l, m, n, nm1, *tile_meta, *lists, cosphi]


def load_day_3d_visibility_manifest(manifest: str | Path, *, scale: str = "nside512_full", segment_profile: str = "all10", tile_pix: int = 512, device: str = "cuda"):
    segs = manifest_segment_paths(manifest, segment_profile=segment_profile)
    if not segs:
        raise ValueError(f"no segments in manifest {manifest}")
    meta = dict(load_pt(segs[0]).get("meta", {}))
    nside = int(meta.get("nside") or get_scale(scale).nside or 512)
    l, m, n = lmn_for_scale(scale, nside=nside, device=device, tile_pix=tile_pix, seed=607)
    nm1 = (n - 1.0).contiguous()
    B = load_common_B(segs[0], n_pix=l.numel(), device=device)
    tile_meta = make_tile_meta_torch(l, m, n, tile_pix)
    lists: list[list[torch.Tensor]] = [[] for _ in range(9)]
    cosphi = _cosphi(meta)
    for seg in segs:
        data = load_pt(seg)
        md = dict(data.get("meta", {})); cosphi = _cosphi(md)
        u = _to_cuda(data["u"].to(torch.float32), device=device)
        v = _to_cuda(data["v"].to(torch.float32), device=device)
        w = _to_cuda(data["w"].to(torch.float32), device=device)
        x1, y1, z1 = split_p4(data["p1"], need_invn=False, device=device)
        x2, y2, z2 = split_p4(data["p2"], need_invn=False, device=device)
        vals = [u, v, w, x1, y1, z1, x2, y2, z2]
        for dst, val in zip(lists, vals):
            dst.append(val)
    return [B, l, m, n, nm1, *tile_meta, *lists, cosphi]


def load_day_ws_recon_rep_manifest(manifest: str | Path, *, scale: str = "nside512_full", segment_profile: str = "all10", tile_pix: int = 256, device: str = "cuda"):
    segs = manifest_segment_paths(manifest, segment_profile=segment_profile)
    if not segs:
        raise ValueError(f"no segments in manifest {manifest}")
    meta = dict(load_pt(segs[0]).get("meta", {}))
    nside = int(meta.get("nside") or get_scale(scale).nside or 512)
    l, m, n = lmn_for_scale(scale, nside=nside, device=device, tile_pix=tile_pix, seed=17)
    keys_list=[]; ugu_list=[]; vgu_list=[]; repV_list=[]; repP1_list=[]; repP2_list=[]; fa_list=[]; fb_list=[]; cosphi=_cosphi(meta)
    for seg in segs:
        data = load_pt(seg); md = dict(data.get("meta", {})); cosphi = _cosphi(md)
        keys_list.append(_to_cuda(data["keys_unique"].to(torch.int32), device=device))
        ugu_list.append(_to_cuda(data["ugu_list"].to(torch.float32), device=device))
        vgu_list.append(_to_cuda(data["vgu_list"].to(torch.float32), device=device))
        repV_list.append(_to_cuda(data["repV"].to(torch.float32), device=device))
        repP1_list.append(_to_cuda(data["repP1"].to(torch.float32), device=device))
        repP2_list.append(_to_cuda(data["repP2"].to(torch.float32), device=device))
        fa_list.append(_fa(md)); fb_list.append(_fb(md))
    return [l, m, n, keys_list, ugu_list, vgu_list, repV_list, repP1_list, repP2_list, fa_list, fb_list, cosphi]


def load_day_3d_recon_direct_manifest(manifest: str | Path, *, scale: str = "nside512_full", segment_profile: str = "all10", tile_pix: int = 256, device: str = "cuda"):
    segs = manifest_segment_paths(manifest, segment_profile=segment_profile)
    if not segs:
        raise ValueError(f"no segments in manifest {manifest}")
    meta = dict(load_pt(segs[0]).get("meta", {}))
    nside = int(meta.get("nside") or get_scale(scale).nside or 512)
    l, m, n = lmn_for_scale(scale, nside=nside, device=device, tile_pix=tile_pix, seed=907)
    tile_meta = make_tile_meta_torch(l, m, n, tile_pix)
    lists: list[list[torch.Tensor]] = [[] for _ in range(11)]
    cosphi = _cosphi(meta)
    for seg in segs:
        data = load_pt(seg); md = dict(data.get("meta", {})); cosphi = _cosphi(md)
        u = _to_cuda(data["u"].to(torch.float32), device=device)
        v = _to_cuda(data["v"].to(torch.float32), device=device)
        w = _to_cuda(data["w"].to(torch.float32), device=device)
        if "p1" in data and "p2" in data:
            p1_src, p2_src = data["p1"], data["p2"]
        else:
            sib = _sibling_visibility_for_recon(seg)
            if sib is None:
                raise KeyError(f"{seg} has no p1/p2 and sibling visibility_segXX.pt was not found")
            vdata = load_pt(sib); p1_src, p2_src = vdata["p1"], vdata["p2"]
        x1, y1, z1 = split_p4(p1_src, need_invn=False, device=device)
        x2, y2, z2 = split_p4(p2_src, need_invn=False, device=device)
        pair_weight = _to_cuda(data.get("pair_weight_half", torch.ones_like(data["u"])).to(torch.float32), device=device)
        Viss_half = _to_cuda(data["Viss_half"].to(torch.float32), device=device)
        vals = [u, v, w, x1, y1, z1, x2, y2, z2, pair_weight, Viss_half]
        for dst, val in zip(lists, vals):
            dst.append(val)
    return [l, m, n, *tile_meta, *lists, cosphi]
