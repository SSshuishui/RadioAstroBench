"""WS Level 3: day-level all-10-segment representative reconstruction bench.

Self-contained task file. It intentionally does not import any level2 task file,
because task filenames use readable KernelBench-style prefixes such as
``01-ws-*`` that are not valid Python module identifiers.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

from radio_astronomy_cuda_bench.common import make_float4, make_unit_vectors_cuda
from radio_astronomy_cuda_bench.configs.scales import get_scale
from radio_astronomy_cuda_bench.configs.day1_10m_segments import COSPHI_10M, get_segments, segment_to_dict

TASK_ID = "level3/02-ws-day-recon-representative"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full"]
CHUNK_KEYS = 64
BLOCK_SIZE = 128

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> ws_recon_blockage_half_rep_forward(
    torch::Tensor l, torch::Tensor m, torch::Tensor n,
    torch::Tensor keys_unique, torch::Tensor ugu_list, torch::Tensor vgu_list,
    torch::Tensor repV, torch::Tensor repP1, torch::Tensor repP2,
    double fa, double fb, double cosphi);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>
#include <stdint.h>

#ifndef SINCOS_FAST_DEFINED
#define SINCOS_FAST_DEFINED
__device__ __forceinline__ void sincos_fast(float x, float* s, float* c) { __sincosf(x, s, c); }
#endif

template<int CHUNK_KEYS>
__global__ void recon_seg_blockage_half_rep(
    long long n_chunk,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const int* __restrict__ keys_unique,
    const float* __restrict__ ugu_list,
    const float* __restrict__ vgu_list,
    const float2* __restrict__ repV,
    const float4* __restrict__ repP1,
    const float4* __restrict__ repP2,
    int nuniq,
    float fa, float fb,
    float cosphi,
    float* __restrict__ Cseg,
    uint32_t* __restrict__ Wseg)
{
  __shared__ int shK[64];
  __shared__ float shU[64];
  __shared__ float shVg[64];
  __shared__ float2 shVV[64];
  __shared__ float4 shP1[64];
  __shared__ float4 shP2[64];

  long long pix=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(pix>=n_chunk) return;

  float lp0=l[pix], mp0=m[pix], np0=n[pix];
  float lp=lp0 + fa*np0;
  float mp=mp0 + fb*np0;

  float acc=0.0f;
  uint32_t wacc=0;
  const float TWO_PI=6.2831853071795864769f;

  for(int base=0;base<nuniq;base+=CHUNK_KEYS){
    int t=threadIdx.x;
    if(t<CHUNK_KEYS){
      int q=base+t;
      if(q<nuniq){
        shK[t]=keys_unique[q];
        shU[t]=ugu_list[q];
        shVg[t]=vgu_list[q];
        shVV[t]=repV[q];
        shP1[t]=repP1[q];
        shP2[t]=repP2[q];
      } else {
        shK[t]=0; shU[t]=0.0f; shVg[t]=0.0f;
        shVV[t]=make_float2(0,0);
        shP1[t]=make_float4(0,0,0,0);
        shP2[t]=make_float4(0,0,0,0);
      }
    }
    __syncthreads();

    #pragma unroll
    for(int kk=0; kk<CHUNK_KEYS; kk++){
      int key=shK[kk];
      if(key==0) continue;
      float4 p1 = shP1[kk];
      float4 p2 = shP2[kk];
      float c1=(lp0*p1.x + mp0*p1.y + np0*p1.z) * p1.w;
      float c2=(lp0*p2.x + mp0*p2.y + np0*p2.z) * p2.w;
      if(c1>=cosphi && c2>=cosphi){
        float phase=TWO_PI*(shU[kk]*lp + shVg[kk]*mp);
        float s,c; sincos_fast(phase,&s,&c);
        float2 vv=shVV[kk];
        acc += vv.x*c - vv.y*s;
        wacc += 1;
      }
    }
    __syncthreads();
  }
  Cseg[pix]=acc;
  Wseg[pix]=wacc;
}

std::vector<torch::Tensor> ws_recon_blockage_half_rep_forward(
    torch::Tensor l, torch::Tensor m, torch::Tensor n,
    torch::Tensor keys_unique, torch::Tensor ugu_list, torch::Tensor vgu_list,
    torch::Tensor repV, torch::Tensor repP1, torch::Tensor repP2,
    double fa_d, double fb_d, double cosphi_d) {
  TORCH_CHECK(l.is_cuda() && keys_unique.is_cuda(), "inputs must be CUDA tensors");
  l=l.contiguous(); m=m.contiguous(); n=n.contiguous(); keys_unique=keys_unique.contiguous();
  ugu_list=ugu_list.contiguous(); vgu_list=vgu_list.contiguous(); repV=repV.contiguous(); repP1=repP1.contiguous(); repP2=repP2.contiguous();
  long long n_chunk = l.numel();
  int nuniq = (int)keys_unique.numel();
  auto Cseg = torch::empty({n_chunk}, l.options());
  auto Wseg = torch::empty({n_chunk}, torch::TensorOptions().dtype(torch::kUInt32).device(l.device()));
  const int block = 128;
  int grid = (int)((n_chunk + block - 1) / block);
  recon_seg_blockage_half_rep<64><<<grid, block>>>(
      n_chunk, l.data_ptr<float>(), m.data_ptr<float>(), n.data_ptr<float>(),
      keys_unique.data_ptr<int>(), ugu_list.data_ptr<float>(), vgu_list.data_ptr<float>(),
      reinterpret_cast<float2*>(repV.data_ptr<float>()), reinterpret_cast<float4*>(repP1.data_ptr<float>()), reinterpret_cast<float4*>(repP2.data_ptr<float>()),
      nuniq, (float)fa_d, (float)fb_d, (float)cosphi_d, Cseg.data_ptr<float>(), (uint32_t*)Wseg.data_ptr<unsigned int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {Cseg, Wseg};
}
"""

_EXT = None

def get_extension():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="rkb_ws_day_recon_blockage_half_rep_selfcontained",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["ws_recon_blockage_half_rep_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT


def _segment_synthetic_tensors(seg, device: str = "cuda"):
    nuniq = int(seg.nuniq_halfgrid)
    keys_unique = torch.arange(1, nuniq + 1, device=device, dtype=torch.int32)
    umin, umax = (seg.u_min, seg.u_max) if seg.u_min is not None and seg.u_max is not None else (-8.0, 8.0)
    vmin, vmax = (seg.v_min, seg.v_max) if seg.v_min is not None and seg.v_max is not None else (-7.0, 7.0)
    ugu_list = torch.linspace(float(umin), float(umax), nuniq, device=device, dtype=torch.float32)
    vgu_list = torch.linspace(float(vmin), float(vmax), nuniq, device=device, dtype=torch.float32)
    phase = torch.linspace(0, 6.283185307179586, nuniq, device=device, dtype=torch.float32) + 0.11 * float(seg.seg_id)
    repV = torch.stack([torch.cos(phase), torch.sin(phase)], dim=1).contiguous()
    repP1 = make_float4(nuniq, device=device, seed=100 + seg.seg_id)
    repP2 = make_float4(nuniq, device=device, seed=200 + seg.seg_id)
    return keys_unique, ugu_list, vgu_list, repV, repP1, repP2


def _try_load_segment_fixture(seg_dir: Path, device: str = "cuda"):
    names = ["keys_unique", "ugu_list", "vgu_list", "repV", "repP1", "repP2"]
    vals = []
    for name in names:
        pt = seg_dir / f"{name}.pt"
        if not pt.exists():
            return None
        vals.append(torch.load(pt, map_location=device))
    return tuple(v.contiguous() for v in vals)


class Model(nn.Module):
    def forward(self, l, m, n, keys_list, ugu_list, vgu_list, repV_list, repP1_list, repP2_list, fa_list, fb_list, cosphi: float):
        ext = get_extension()
        C_total = torch.zeros_like(l)
        W_total = torch.zeros_like(l, dtype=torch.int32)
        for i in range(len(keys_list)):
            Cseg, Wseg = ext.ws_recon_blockage_half_rep_forward(
                l, m, n,
                keys_list[i], ugu_list[i], vgu_list[i], repV_list[i], repP1_list[i], repP2_list[i],
                float(fa_list[i]), float(fb_list[i]), float(cosphi),
            )
            C_total = C_total + Cseg
            W_total = W_total + Wseg.to(torch.int32)
        return [C_total, W_total]


class ModelNew(Model):
    pass


def get_inputs(scale: str = "smoke", segment_profile: str = "all10", fixture: str | None = None):
    cfg = get_scale(scale)
    if not cfg.run_reconstruction:
        raise ValueError(f"Scale {scale} disables reconstruction tasks")
    device = "cuda"
    l, m, n = make_unit_vectors_cuda(cfg.npix, device=device, seed=17)
    segs = get_segments(scale, segment_profile)
    keys_list: List[torch.Tensor] = []
    ugu_list: List[torch.Tensor] = []
    vgu_list: List[torch.Tensor] = []
    repV_list: List[torch.Tensor] = []
    repP1_list: List[torch.Tensor] = []
    repP2_list: List[torch.Tensor] = []
    fa_list: List[float] = []
    fb_list: List[float] = []
    for seg in segs:
        loaded = None
        if fixture:
            loaded = _try_load_segment_fixture(Path(fixture) / f"seg{seg.seg_id:02d}", device=device)
        k, u, v, rv, p1, p2 = loaded if loaded is not None else _segment_synthetic_tensors(seg, device=device)
        keys_list.append(k); ugu_list.append(u); vgu_list.append(v); repV_list.append(rv); repP1_list.append(p1); repP2_list.append(p2)
        fa_list.append(float(seg.fa)); fb_list.append(float(seg.fb))
    return [l, m, n, keys_list, ugu_list, vgu_list, repV_list, repP1_list, repP2_list, fa_list, fb_list, float(COSPHI_10M)]


def describe_inputs(scale: str = "smoke", segment_profile: str = "all10", fixture: str | None = None):
    cfg = get_scale(scale); segs = get_segments(scale, segment_profile)
    return {"mode": "ws day-level all-segment reconstruction", "scale": scale, "nside": cfg.nside, "npix": cfg.npix, "num_segments": len(segs), "fixture_mode": "external" if fixture else "synthetic_segment_aware", "segments": [segment_to_dict(s) for s in segs]}


def get_init_inputs():
    return []
