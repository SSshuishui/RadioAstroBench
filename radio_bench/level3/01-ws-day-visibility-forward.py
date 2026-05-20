"""WS Level 3: day-level all-10-segment visibility tile-cone bench.

Self-contained task file. It does not import level2 task files.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from typing import List

from radio_astronomy_cuda_bench.common import make_endpoint_vectors, make_tile_meta_torch, make_unit_vectors_cuda
from radio_astronomy_cuda_bench.configs.scales import get_scale
from radio_astronomy_cuda_bench.configs.day1_10m_segments import COSPHI_10M, get_segments, segment_to_dict

TASK_ID = "level3/01-ws-day-visibility-forward"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full"]
TILE_PIX = 256
UNIQUE_BASELINES_PER_T = 28
SIGNED_BASELINES_PER_T = 56

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor visibility_tilecone_forward(
    torch::Tensor B, torch::Tensor l, torch::Tensor m, torch::Tensor n, torch::Tensor nm1,
    torch::Tensor tile_cx, torch::Tensor tile_cy, torch::Tensor tile_cz,
    torch::Tensor tile_cosA, torch::Tensor tile_sinA,
    torch::Tensor u, torch::Tensor v, torch::Tensor w,
    torch::Tensor x1, torch::Tensor y1, torch::Tensor z1, torch::Tensor invn1,
    torch::Tensor x2, torch::Tensor y2, torch::Tensor z2, torch::Tensor invn2,
    double cosphi);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef SINCOS_FAST_DEFINED
#define SINCOS_FAST_DEFINED
__device__ __forceinline__ void sincos_fast(float x, float* s, float* c) { __sincosf(x, s, c); }
#endif

static constexpr int SIGNED_BASELINES_PER_T = 56;
static constexpr int UNIQUE_BASELINES_PER_T = 28;

__device__ __forceinline__ void classify_tile_cone(float dotc, float cosA, float sinA, float cosphi, bool &all_vis, bool &all_hid){
  dotc = fminf(1.0f, fmaxf(-1.0f, dotc));
  float sind = sqrtf(fmaxf(0.0f, 1.0f - dotc*dotc));
  float lower = dotc * cosA - sind * sinA;
  float upper = dotc * cosA + sind * sinA;
  all_vis = (lower >= cosphi);
  all_hid = (upper <  cosphi);
}

template<int TILE_PIX, bool DO_BLOCKAGE>
__global__ void viss_partial_all_halfsym_tilecone(
    const float* __restrict__ B,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const float* __restrict__ nm1,
    long long n_chunk,
    const float* __restrict__ tile_cx,
    const float* __restrict__ tile_cy,
    const float* __restrict__ tile_cz,
    const float* __restrict__ tile_cosA,
    const float* __restrict__ tile_sinA,
    int ntile,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ x1,
    const float* __restrict__ y1,
    const float* __restrict__ z1,
    const float* __restrict__ invn1,
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    const float* __restrict__ invn2,
    int N_half,
    float cosphi,
    float2* __restrict__ Vpart)
{
  int ih = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = (ih < N_half);

  int i = 0;
  float u0 = 0.0f, v0 = 0.0f, w0 = 0.0f;
  float x1i = 0.0f, y1i = 0.0f, z1i = 0.0f;
  float x2i = 0.0f, y2i = 0.0f, z2i = 0.0f;
  float in1 = 0.0f, in2 = 0.0f;
  float ux1=0.0f, uy1=0.0f, uz1=0.0f, ux2=0.0f, uy2=0.0f, uz2=0.0f;

  if(active){
    int group = ih / UNIQUE_BASELINES_PER_T;
    int j     = ih - group * UNIQUE_BASELINES_PER_T;
    i         = group * SIGNED_BASELINES_PER_T + j;
    u0 = u[i]; v0 = v[i]; w0 = w[i];
    if constexpr (DO_BLOCKAGE){
      x1i = x1[i]; y1i = y1[i]; z1i = z1[i]; in1 = invn1[i];
      x2i = x2[i]; y2i = y2[i]; z2i = z2[i]; in2 = invn2[i];
      ux1 = x1i * in1; uy1 = y1i * in1; uz1 = z1i * in1;
      ux2 = x2i * in2; uy2 = y2i * in2; uz2 = z2i * in2;
    }
  }

  float acc_re = 0.0f, acc_im = 0.0f;
  const float k = -2.0f * (float)M_PI;

  float ku=0.0f, kv=0.0f, kw=0.0f;
  if(active){ ku = k * u0; kv = k * v0; kw = k * w0; }

  extern __shared__ float smem[];
  float* sB = smem;
  float* sL = sB + TILE_PIX;
  float* sM = sL + TILE_PIX;
  float* sN = sM + TILE_PIX;
  float* sNM1 = sN + TILE_PIX;

  for(int tid=0; tid<ntile; ++tid){
    long long p0 = (long long)tid * TILE_PIX;
    int tileN = (int)min((long long)TILE_PIX, n_chunk - p0);
    for(int lane = threadIdx.x; lane < tileN; lane += blockDim.x){
      long long p = p0 + lane;
      sB[lane] = B[p];
      sL[lane] = l[p];
      sM[lane] = m[p];
      sN[lane] = n[p];
      sNM1[lane] = nm1[p];
    }
    __syncthreads();

    bool all_visible = false;
    bool all_hidden  = false;
    if(active && DO_BLOCKAGE){
      float cx = tile_cx[tid], cy = tile_cy[tid], cz = tile_cz[tid];
      float cosA = tile_cosA[tid], sinA = tile_sinA[tid];
      bool vis1=false, hid1=false, vis2=false, hid2=false;
      classify_tile_cone(ux1*cx + uy1*cy + uz1*cz, cosA, sinA, cosphi, vis1, hid1);
      classify_tile_cone(ux2*cx + uy2*cy + uz2*cz, cosA, sinA, cosphi, vis2, hid2);
      all_hidden = hid1 || hid2;
      all_visible = vis1 && vis2;
    }

    if(active && !(DO_BLOCKAGE && all_hidden)) {
      if(!DO_BLOCKAGE || all_visible){
        #pragma unroll 4
        for(int k0=0; k0<tileN; ++k0){
          float lp = sL[k0], mp = sM[k0], nm1v = sNM1[k0];
          float ang = fmaf(kw, nm1v, fmaf(kv, mp, ku * lp));
          float s,c; sincos_fast(ang,&s,&c);
          float bp=sB[k0];
          acc_re += bp*c;
          acc_im += bp*s;
        }
      }else{
        #pragma unroll 4
        for(int k0=0; k0<tileN; ++k0){
          float lp = sL[k0], mp = sM[k0], npv = sN[k0], nm1v = sNM1[k0];
          float c1=(lp*x1i + mp*y1i + npv*z1i)*in1;
          float c2=(lp*x2i + mp*y2i + npv*z2i)*in2;
          if(c1<cosphi || c2<cosphi) continue;
          float ang = fmaf(kw, nm1v, fmaf(kv, mp, ku * lp));
          float s,c; sincos_fast(ang,&s,&c);
          float bp=sB[k0];
          acc_re += bp*c;
          acc_im += bp*s;
        }
      }
    }
    __syncthreads();
  }

  if(active) Vpart[ih] = make_float2(acc_re, acc_im);
}

torch::Tensor visibility_tilecone_forward(
    torch::Tensor B, torch::Tensor l, torch::Tensor m, torch::Tensor n, torch::Tensor nm1,
    torch::Tensor tile_cx, torch::Tensor tile_cy, torch::Tensor tile_cz,
    torch::Tensor tile_cosA, torch::Tensor tile_sinA,
    torch::Tensor u, torch::Tensor v, torch::Tensor w,
    torch::Tensor x1, torch::Tensor y1, torch::Tensor z1, torch::Tensor invn1,
    torch::Tensor x2, torch::Tensor y2, torch::Tensor z2, torch::Tensor invn2,
    double cosphi_d) {
  TORCH_CHECK(B.is_cuda(), "inputs must be CUDA tensors");
  B=B.contiguous(); l=l.contiguous(); m=m.contiguous(); n=n.contiguous(); nm1=nm1.contiguous();
  tile_cx=tile_cx.contiguous(); tile_cy=tile_cy.contiguous(); tile_cz=tile_cz.contiguous();
  tile_cosA=tile_cosA.contiguous(); tile_sinA=tile_sinA.contiguous();
  u=u.contiguous(); v=v.contiguous(); w=w.contiguous();
  x1=x1.contiguous(); y1=y1.contiguous(); z1=z1.contiguous(); invn1=invn1.contiguous();
  x2=x2.contiguous(); y2=y2.contiguous(); z2=z2.contiguous(); invn2=invn2.contiguous();
  long long n_chunk = B.numel();
  int ntile = (int)tile_cx.numel();
  int N_full = (int)u.numel();
  TORCH_CHECK(N_full % SIGNED_BASELINES_PER_T == 0, "u/v/w length must be multiple of 56");
  int N_half = (N_full / SIGNED_BASELINES_PER_T) * UNIQUE_BASELINES_PER_T;
  auto out = torch::empty({N_half, 2}, B.options());
  int block = 128;
  int grid = (N_half + block - 1) / block;
  size_t smem = 5 * 256 * sizeof(float);
  viss_partial_all_halfsym_tilecone<256, true><<<grid, block, smem>>>(
      B.data_ptr<float>(), l.data_ptr<float>(), m.data_ptr<float>(), n.data_ptr<float>(), nm1.data_ptr<float>(), n_chunk,
      tile_cx.data_ptr<float>(), tile_cy.data_ptr<float>(), tile_cz.data_ptr<float>(), tile_cosA.data_ptr<float>(), tile_sinA.data_ptr<float>(), ntile,
      u.data_ptr<float>(), v.data_ptr<float>(), w.data_ptr<float>(),
      x1.data_ptr<float>(), y1.data_ptr<float>(), z1.data_ptr<float>(), invn1.data_ptr<float>(),
      x2.data_ptr<float>(), y2.data_ptr<float>(), z2.data_ptr<float>(), invn2.data_ptr<float>(),
      N_half, (float)cosphi_d, reinterpret_cast<float2*>(out.data_ptr<float>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""


_EXT = None

def get_extension():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="rkb_ws_day_visibility_tilecone_selfcontained",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["visibility_tilecone_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT


def _make_B(n: int, device: str = "cuda") -> torch.Tensor:
    idx = torch.arange(n, device=device, dtype=torch.float32)
    return (1.0 + 0.10 * torch.sin(idx * 0.00013) + 0.03 * torch.cos(idx * 0.0017)).contiguous()


def _segment_baseline_tensors(seg, device: str = "cuda"):
    N_full = int(seg.segN)
    groups = N_full // SIGNED_BASELINES_PER_T
    assert N_full % SIGNED_BASELINES_PER_T == 0
    first = groups * UNIQUE_BASELINES_PER_T
    u_unique = torch.linspace(float(seg.u_min or -100.0), float(seg.u_max or 100.0), first, device=device, dtype=torch.float32)
    v_unique = torch.linspace(float(seg.v_min or -100.0), float(seg.v_max or 100.0), first, device=device, dtype=torch.float32)
    w_unique = torch.linspace(float(seg.w_min or -50.0), float(seg.w_max or 50.0), first, device=device, dtype=torch.float32)
    u = torch.empty(N_full, device=device, dtype=torch.float32)
    v = torch.empty_like(u); w = torch.empty_like(u)
    uu = u.view(groups, SIGNED_BASELINES_PER_T); vv = v.view(groups, SIGNED_BASELINES_PER_T); ww = w.view(groups, SIGNED_BASELINES_PER_T)
    uu[:, :UNIQUE_BASELINES_PER_T] = u_unique.view(groups, UNIQUE_BASELINES_PER_T)
    vv[:, :UNIQUE_BASELINES_PER_T] = v_unique.view(groups, UNIQUE_BASELINES_PER_T)
    ww[:, :UNIQUE_BASELINES_PER_T] = w_unique.view(groups, UNIQUE_BASELINES_PER_T)
    uu[:, UNIQUE_BASELINES_PER_T:] = -uu[:, :UNIQUE_BASELINES_PER_T]
    vv[:, UNIQUE_BASELINES_PER_T:] = -vv[:, :UNIQUE_BASELINES_PER_T]
    ww[:, UNIQUE_BASELINES_PER_T:] = -ww[:, :UNIQUE_BASELINES_PER_T]
    x1, y1, z1, invn1 = make_endpoint_vectors(N_full, device=device, seed=300 + seg.seg_id)
    x2, y2, z2, invn2 = make_endpoint_vectors(N_full, device=device, seed=400 + seg.seg_id)
    return u.contiguous(), v.contiguous(), w.contiguous(), x1, y1, z1, invn1, x2, y2, z2, invn2


class Model(nn.Module):
    def forward(self, B, l, m, n, nm1, tile_cx, tile_cy, tile_cz, tile_cosA, tile_sinA,
                u_list, v_list, w_list, x1_list, y1_list, z1_list, invn1_list,
                x2_list, y2_list, z2_list, invn2_list, cosphi: float):
        ext = get_extension()
        outs = []
        for i in range(len(u_list)):
            out = ext.visibility_tilecone_forward(
                B, l, m, n, nm1, tile_cx, tile_cy, tile_cz, tile_cosA, tile_sinA,
                u_list[i], v_list[i], w_list[i],
                x1_list[i], y1_list[i], z1_list[i], invn1_list[i],
                x2_list[i], y2_list[i], z2_list[i], invn2_list[i],
                float(cosphi),
            )
            outs.append(out)
        return torch.cat(outs, dim=0)


class ModelNew(Model):
    pass


def get_inputs(scale: str = "smoke", segment_profile: str = "all10", fixture: str | None = None):
    cfg = get_scale(scale)
    if not cfg.run_visibility:
        raise ValueError(f"Scale {scale} disables visibility tasks")
    device = "cuda"
    l, m, n = make_unit_vectors_cuda(cfg.npix, device=device, seed=27)
    nm1 = (n - 1.0).contiguous()
    B = _make_B(cfg.npix, device=device)
    tile_meta = make_tile_meta_torch(l, m, n, TILE_PIX)
    segs = get_segments(scale, segment_profile)
    lists = [[] for _ in range(11)]
    for seg in segs:
        tensors = _segment_baseline_tensors(seg, device=device)
        for dst, val in zip(lists, tensors):
            dst.append(val)
    return [B, l, m, n, nm1, *tile_meta, *lists, float(COSPHI_10M)]


def describe_inputs(scale: str = "smoke", segment_profile: str = "all10", fixture: str | None = None):
    cfg = get_scale(scale); segs = get_segments(scale, segment_profile)
    return {"mode": "ws day-level all-segment visibility", "scale": scale, "nside": cfg.nside, "npix": cfg.npix, "tile_pix": TILE_PIX, "num_tiles": cfg.npix // TILE_PIX, "num_segments": len(segs), "fixture_mode": "external_not_yet_implemented" if fixture else "synthetic_segment_aware", "segments": [segment_to_dict(s) for s in segs]}


def get_init_inputs():
    return []
