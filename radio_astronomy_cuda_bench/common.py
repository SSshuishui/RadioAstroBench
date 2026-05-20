"""Common helpers for the existing-CUDA RadioKernelBench prototype.

This benchmark intentionally uses Torch only as:
1) a tensor container,
2) a CUDA extension build/call interface, and
3) a timing/checking harness.
The actual baseline kernels are CUDA kernels extracted from the original project.
"""
from __future__ import annotations

import importlib.util
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import torch


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this benchmark on a CUDA-enabled server.")


def load_task(task_path: str | Path):
    task_path = Path(task_path).resolve()
    # Task filenames intentionally use KernelBench-style readable prefixes such
    # as ``01-ws-...`` and ``02-3d-...``.  These are valid file names but not
    # valid Python identifiers, so we sanitize the dynamic module name here.
    module_name = "radio_task_" + re.sub(r"[^0-9a-zA-Z_]", "_", task_path.stem)
    spec = importlib.util.spec_from_file_location(module_name, str(task_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load task from {task_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flatten_outputs(out: Any) -> List[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        tensors: List[torch.Tensor] = []
        for item in out:
            tensors.extend(flatten_outputs(item))
        return tensors
    raise TypeError(f"Unsupported output type: {type(out)!r}")


def tensor_summary(t: torch.Tensor) -> dict:
    tt = t.detach()
    d = {"shape": list(tt.shape), "dtype": str(tt.dtype), "device": str(tt.device)}
    if tt.numel() == 0:
        d.update({"min": None, "max": None, "mean": None})
        return d
    if tt.dtype.is_floating_point or tt.dtype.is_complex:
        d.update({
            "min": float(torch.nan_to_num(tt.float()).min().item()),
            "max": float(torch.nan_to_num(tt.float()).max().item()),
            "mean": float(torch.nan_to_num(tt.float()).mean().item()),
            "has_nan": bool(torch.isnan(tt.float()).any().item()),
            "has_inf": bool(torch.isinf(tt.float()).any().item()),
        })
    else:
        # PyTorch CUDA does not implement reductions such as min/max for some
        # unsigned integer dtypes (notably torch.uint32).  Wseg in the radio
        # reconstruction tasks is uint32, so cast only for summary statistics.
        # This does not change the actual benchmark output tensor.
        if tt.dtype == torch.bool:
            work = tt.to(torch.int64)
        elif tt.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint32, torch.uint64):
            work = tt.to(torch.int64)
        else:
            work = tt.to(torch.float32)
        d.update({
            "min": int(work.min().item()),
            "max": int(work.max().item()),
            "mean": float(work.to(torch.float32).mean().item()),
            "has_nan": False,
            "has_inf": False,
        })
    return d


@torch.no_grad()
def cuda_time_ms(model: torch.nn.Module, inputs: Sequence[Any], warmup: int = 5, repeat: int = 20) -> float:
    require_cuda()
    for _ in range(warmup):
        _ = model(*inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        _ = model(*inputs)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / max(1, repeat))


def compare_outputs(ref: Any, cand: Any, rtol: float = 1e-4, atol: float = 1e-4) -> dict:
    refs = flatten_outputs(ref)
    cands = flatten_outputs(cand)
    if len(refs) != len(cands):
        return {"passed": False, "reason": f"output_count_mismatch {len(refs)} vs {len(cands)}"}
    details = []
    passed = True
    for i, (a, b) in enumerate(zip(refs, cands)):
        if a.shape != b.shape or a.dtype != b.dtype:
            details.append({"index": i, "passed": False, "reason": f"shape/dtype mismatch {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"})
            passed = False
            continue
        if a.dtype.is_floating_point:
            diff = (a - b).abs()
            max_abs = float(diff.max().item()) if diff.numel() else 0.0
            denom = a.abs().clamp_min(1e-12)
            max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
            ok = bool(torch.allclose(a, b, rtol=rtol, atol=atol))
            details.append({"index": i, "passed": ok, "max_abs": max_abs, "max_rel": max_rel})
            passed = passed and ok
        else:
            # Cast for comparison because some CUDA unsigned integer dtypes do
            # not support all reduction paths consistently across PyTorch builds.
            aa = a.to(torch.int64) if a.dtype in (torch.uint32, torch.uint64) else a
            bb = b.to(torch.int64) if b.dtype in (torch.uint32, torch.uint64) else b
            neq = int((aa != bb).sum().item())
            ok = (neq == 0)
            details.append({"index": i, "passed": ok, "num_mismatch": neq})
            passed = passed and ok
    return {"passed": bool(passed), "details": details}


def make_unit_vectors(n: int, device: str = "cuda", seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic directions on the sphere for smoke tests.

    This is not a real HEALPix sky; it is a synthetic fixture for isolated kernel tests.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    z = torch.linspace(-0.95, 0.95, n, dtype=torch.float32)
    phi = torch.linspace(-math.pi, math.pi, n, dtype=torch.float32)
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    l = r * torch.cos(phi)
    m = r * torch.sin(phi)
    return l.to(device), m.to(device), z.to(device)


def make_endpoint_vectors(n: int, device: str = "cuda", seed: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn(n, generator=gen, dtype=torch.float32)
    y = torch.randn(n, generator=gen, dtype=torch.float32)
    z = torch.randn(n, generator=gen, dtype=torch.float32)
    inv = torch.rsqrt(x * x + y * y + z * z + 1e-12)
    return x.to(device), y.to(device), z.to(device), inv.to(device)


def make_float4(n: int, device: str = "cuda", seed: int = 11) -> torch.Tensor:
    x, y, z, inv = make_endpoint_vectors(n, device=device, seed=seed)
    return torch.stack([x, y, z, inv], dim=1).contiguous()


def make_tile_meta_torch(l: torch.Tensor, m: torch.Tensor, n: torch.Tensor, tile_pix: int = 256):
    """Torch reference implementation for tile cone metadata generation."""
    require_cuda()
    n_chunk = l.numel()
    assert n_chunk % tile_pix == 0, "for this smoke fixture, n_chunk should be divisible by tile_pix"
    ntile = n_chunk // tile_pix
    L = l.view(ntile, tile_pix)
    M = m.view(ntile, tile_pix)
    N = n.view(ntile, tile_pix)
    sx = L.sum(dim=1)
    sy = M.sum(dim=1)
    sz = N.sum(dim=1)
    norm = torch.rsqrt(sx * sx + sy * sy + sz * sz + 1e-12)
    cx = sx * norm
    cy = sy * norm
    cz = sz * norm
    dots = L * cx[:, None] + M * cy[:, None] + N * cz[:, None]
    cosA = dots.min(dim=1).values.clamp(-1.0, 1.0)
    sinA = torch.sqrt(torch.clamp(1.0 - cosA * cosA, min=0.0))
    return cx.contiguous(), cy.contiguous(), cz.contiguous(), cosA.contiguous(), sinA.contiguous()

# ---- Scale-aware helpers added in v1.3 ----

def make_unit_vectors_cuda(n: int, device: str = "cuda", seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate deterministic full-sky-like direction cosines directly on GPU.

    This avoids creating very large CPU tensors for nside=4096 full workloads.
    It is still a synthetic direction generator, not a strict HEALPix NEST mapping.
    Use exported fixtures or the original pix2lmn kernel when exact HEALPix
    ordering matters.
    """
    idx = torch.arange(n, device=device, dtype=torch.float32)
    # z in [-0.95, 0.95], azimuth wraps many times for a full-sky-like spread.
    z = -0.95 + 1.90 * (idx + 0.5) / float(max(1, n))
    phi = (idx * 2.39996322972865332 + float(seed) * 0.17)  # golden-angle style
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    l = r * torch.cos(phi)
    m = r * torch.sin(phi)
    return l.contiguous(), m.contiguous(), z.contiguous()


def estimate_tensor_bytes(tensors: Sequence[torch.Tensor]) -> int:
    total = 0
    for t in tensors:
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
    return int(total)


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{n} B"
