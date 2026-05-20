"""Level 2 / 3D task: direct 3D reconstruction mixed task-list kernel.

Extracted from ``recon_3d_direct_tasklist_mixed_real``.  This is the mixed
visibility path used by the 3D inverse imaging stage after task-list planning.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.common import make_unit_vectors, make_endpoint_vectors, make_tile_meta_torch

TASK_ID = "level2/11-3d-recon-tasklist-mixed"
SUPPORTED_SCALES = ["smoke", "nside512_full"]
TILE_PIX = 128
TILE_BL = 64

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor recon_3d_mixed_forward(
    torch::Tensor l, torch::Tensor m, torch::Tensor n,
    torch::Tensor tile_cx, torch::Tensor tile_cy, torch::Tensor tile_cz,
    torch::Tensor tile_cosA, torch::Tensor tile_sinA,
    torch::Tensor u, torch::Tensor v, torch::Tensor w,
    torch::Tensor x1, torch::Tensor y1, torch::Tensor z1,
    torch::Tensor x2, torch::Tensor y2, torch::Tensor z2,
    torch::Tensor pair_weight_half, torch::Tensor Viss_half,
    torch::Tensor task_ids, int ntile, int b0, int chunk_n, double cosphi);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
__device__ __forceinline__ void sincos_fast(float x, float* s, float* c) { __sincosf(x, s, c); }
__device__ __forceinline__ float operator_adjoint_phase(float u, float v, float w, float l, float m, float n){
  return 2.0f * (float)M_PI * (u*l + v*m + w*n);
}
__device__ __forceinline__ void classify_tile_cone(float dotc, float cosA, float sinA, float cosphi, bool &all_vis, bool &all_hid){
  dotc = fminf(1.0f, fmaxf(-1.0f, dotc));
  float sind = sqrtf(fmaxf(0.0f, 1.0f - dotc*dotc));
  float lower = dotc * cosA - sind * sinA;
  float upper = dotc * cosA + sind * sinA;
  all_vis = (lower >= cosphi);
  all_hid = (upper <  cosphi);
}
__device__ __forceinline__ void operator_classify_pair_tile_dirs(
    float x1,float y1,float z1,float x2,float y2,float z2,
    float cx,float cy,float cz,float cosA,float sinA,float cosphi,
    bool &allv, bool &allh)
{
  bool v1=false,h1=false,v2=false,h2=false;
  classify_tile_cone(x1*cx + y1*cy + z1*cz, cosA, sinA, cosphi, v1, h1);
  classify_tile_cone(x2*cx + y2*cy + z2*cz, cosA, sinA, cosphi, v2, h2);
  allv = v1 && v2;
  allh = h1 || h2;
}
__device__ __forceinline__ bool operator_point_visible_pair_dirs(
    float x1,float y1,float z1,float x2,float y2,float z2,
    float l,float m,float n,float cosphi)
{
  float c1 = x1*l + y1*m + z1*n;
  float c2 = x2*l + y2*m + z2*n;
  return (c1 >= cosphi) && (c2 >= cosphi);
}

template<int TILE_PIX, int TILE_BL>
__global__ void recon_3d_direct_tasklist_mixed_real(
    long long n_chunk,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
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
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    const float* __restrict__ pair_weight_half,
    const float2* __restrict__ Viss_half,
    int b0,
    int chunk_n,
    float cosphi,
    const int* __restrict__ task_ids,
    int num_tasks,
    float* __restrict__ Cacc)
{
  int task_idx = blockIdx.x;
  if(task_idx >= num_tasks) return;
  int task = task_ids[task_idx];
  int blk = task / ntile;
  int tile = task - blk * ntile;
  int base = b0 + blk * TILE_BL;
  int tileBL = min(TILE_BL, chunk_n - blk * TILE_BL);

  __shared__ float su[TILE_BL], sv[TILE_BL], sw[TILE_BL];
  __shared__ float sdx1[TILE_BL], sdy1[TILE_BL], sdz1[TILE_BL];
  __shared__ float sdx2[TILE_BL], sdy2[TILE_BL], sdz2[TILE_BL];
  __shared__ float spw[TILE_BL];
  __shared__ float2 sV[TILE_BL];
  __shared__ unsigned short svv_idx[TILE_BL], smix_idx[TILE_BL];
  __shared__ int nvv, nmix;

  int lane = threadIdx.x;
  long long pix = (long long)tile * TILE_PIX + lane;
  bool active = (pix < n_chunk);
  float lp=0.0f, mp=0.0f, npv=0.0f;
  if(active){ lp=l[pix]; mp=m[pix]; npv=n[pix]; }

  for(int t=lane; t<tileBL; t+=blockDim.x){
    int ih = base + t;
    su[t]=u[ih]; sv[t]=v[ih]; sw[t]=w[ih];
    sdx1[t]=x1[ih]; sdy1[t]=y1[ih]; sdz1[t]=z1[ih];
    sdx2[t]=x2[ih]; sdy2[t]=y2[ih]; sdz2[t]=z2[ih];
    spw[t]=pair_weight_half[ih];
    sV[t]=Viss_half[ih];
  }
  if(lane==0){ nvv=0; nmix=0; }
  __syncthreads();

  float tcx = tile_cx[tile], tcy = tile_cy[tile], tcz = tile_cz[tile];
  float tcosA = tile_cosA[tile], tsinA = tile_sinA[tile];
  if(lane < tileBL){
    bool allv=false, allh=false;
    operator_classify_pair_tile_dirs(sdx1[lane],sdy1[lane],sdz1[lane],sdx2[lane],sdy2[lane],sdz2[lane],tcx,tcy,tcz,tcosA,tsinA,cosphi,allv,allh);
    if(allv){
      int dst = atomicAdd(&nvv, 1);
      svv_idx[dst] = (unsigned short)lane;
    } else if(!allh){
      int dst = atomicAdd(&nmix, 1);
      smix_idx[dst] = (unsigned short)lane;
    }
  }
  __syncthreads();

  float acc_re = 0.0f;
  if(active){
    #pragma unroll 2
    for(int ii=0; ii<nvv; ++ii){
      int t = (int)svv_idx[ii];
      float pairw = spw[t];
      if(pairw <= 0.0f) continue;
      float phase = operator_adjoint_phase(su[t], sv[t], sw[t], lp, mp, npv);
      float s, c; sincos_fast(phase, &s, &c);
      float2 z = sV[t];
      if(!isfinite(z.x) || !isfinite(z.y)) continue;
      acc_re += pairw * (z.x * c - z.y * s);
    }
    #pragma unroll 2
    for(int ii=0; ii<nmix; ++ii){
      int t = (int)smix_idx[ii];
      float pairw = spw[t];
      if(pairw <= 0.0f) continue;
      if(!operator_point_visible_pair_dirs(sdx1[t],sdy1[t],sdz1[t],sdx2[t],sdy2[t],sdz2[t],lp,mp,npv,cosphi)) continue;
      float phase = operator_adjoint_phase(su[t], sv[t], sw[t], lp, mp, npv);
      float s, c; sincos_fast(phase, &s, &c);
      float2 z = sV[t];
      if(!isfinite(z.x) || !isfinite(z.y)) continue;
      acc_re += pairw * (z.x * c - z.y * s);
    }
  }
  if(active) Cacc[pix] += acc_re;
}

torch::Tensor recon_3d_mixed_forward(
    torch::Tensor l, torch::Tensor m, torch::Tensor n,
    torch::Tensor tile_cx, torch::Tensor tile_cy, torch::Tensor tile_cz,
    torch::Tensor tile_cosA, torch::Tensor tile_sinA,
    torch::Tensor u, torch::Tensor v, torch::Tensor w,
    torch::Tensor x1, torch::Tensor y1, torch::Tensor z1,
    torch::Tensor x2, torch::Tensor y2, torch::Tensor z2,
    torch::Tensor pair_weight_half, torch::Tensor Viss_half,
    torch::Tensor task_ids, int ntile, int b0, int chunk_n, double cosphi_d) {
  TORCH_CHECK(l.is_cuda() && u.is_cuda() && task_ids.is_cuda(), "inputs must be CUDA tensors");
  l=l.contiguous(); m=m.contiguous(); n=n.contiguous();
  tile_cx=tile_cx.contiguous(); tile_cy=tile_cy.contiguous(); tile_cz=tile_cz.contiguous(); tile_cosA=tile_cosA.contiguous(); tile_sinA=tile_sinA.contiguous();
  u=u.contiguous(); v=v.contiguous(); w=w.contiguous(); x1=x1.contiguous(); y1=y1.contiguous(); z1=z1.contiguous(); x2=x2.contiguous(); y2=y2.contiguous(); z2=z2.contiguous();
  pair_weight_half=pair_weight_half.contiguous(); Viss_half=Viss_half.contiguous(); task_ids=task_ids.contiguous();
  long long n_chunk = l.numel();
  int num_tasks = (int)task_ids.numel();
  auto Cacc = torch::zeros({n_chunk}, l.options());
  recon_3d_direct_tasklist_mixed_real<128,64><<<num_tasks,128>>>(
      n_chunk, l.data_ptr<float>(), m.data_ptr<float>(), n.data_ptr<float>(),
      tile_cx.data_ptr<float>(), tile_cy.data_ptr<float>(), tile_cz.data_ptr<float>(), tile_cosA.data_ptr<float>(), tile_sinA.data_ptr<float>(), ntile,
      u.data_ptr<float>(), v.data_ptr<float>(), w.data_ptr<float>(),
      x1.data_ptr<float>(), y1.data_ptr<float>(), z1.data_ptr<float>(), x2.data_ptr<float>(), y2.data_ptr<float>(), z2.data_ptr<float>(),
      pair_weight_half.data_ptr<float>(), reinterpret_cast<float2*>(Viss_half.data_ptr<float>()),
      b0, chunk_n, (float)cosphi_d, task_ids.data_ptr<int>(), num_tasks, Cacc.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return Cacc;
}
"""

_EXT = None

def get_extension():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="rkb_3d_recon_direct_tasklist_mixed_real",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["recon_3d_mixed_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT

class Model(nn.Module):
    def forward(self, l, m, n, tile_cx, tile_cy, tile_cz, tile_cosA, tile_sinA, u, v, w, x1, y1, z1, x2, y2, z2, pair_weight_half, Viss_half, task_ids, ntile: int, b0: int, chunk_n: int, cosphi: float):
        return get_extension().recon_3d_mixed_forward(l, m, n, tile_cx, tile_cy, tile_cz, tile_cosA, tile_sinA, u, v, w, x1, y1, z1, x2, y2, z2, pair_weight_half, Viss_half, task_ids, int(ntile), int(b0), int(chunk_n), float(cosphi))

class ModelNew(Model):
    pass


def _n_pix(scale: str) -> int:
    return 8192 if scale == "smoke" else 512 * 512 * 12


def get_inputs(scale: str = "smoke", **_):
    device = "cuda"
    n_pix = (_n_pix(scale) // TILE_PIX) * TILE_PIX
    l, m, n = make_unit_vectors(n_pix, device=device, seed=401)
    tile_cx, tile_cy, tile_cz, tile_cosA, tile_sinA = make_tile_meta_torch(l, m, n, tile_pix=TILE_PIX)
    ntile = n_pix // TILE_PIX
    chunk_n = 1024 if scale == "smoke" else 4096
    t = torch.linspace(-1.0, 1.0, chunk_n, device=device, dtype=torch.float32)
    u = 700.0 * t
    v = 500.0 * torch.sin(t * 3.0)
    w = 350.0 * torch.cos(t * 2.0)
    x1, y1, z1, _ = make_endpoint_vectors(chunk_n, device=device, seed=402)
    x2, y2, z2, _ = make_endpoint_vectors(chunk_n, device=device, seed=403)
    pair_weight_half = torch.full((chunk_n,), 0.02, device=device, dtype=torch.float32)
    phase = torch.linspace(0, 6.2831853, chunk_n, device=device)
    Viss_half = torch.stack([torch.cos(phase), torch.sin(phase)], dim=1).contiguous().to(torch.float32)
    nblocks = (chunk_n + TILE_BL - 1) // TILE_BL
    num_tasks = min(ntile * nblocks, 4096 if scale == "smoke" else 32768)
    task_ids = torch.arange(num_tasks, device=device, dtype=torch.int32)
    cosphi = 0.15
    return [l, m, n, tile_cx, tile_cy, tile_cz, tile_cosA, tile_sinA, u, v, w, x1, y1, z1, x2, y2, z2, pair_weight_half, Viss_half, task_ids, ntile, 0, chunk_n, cosphi]


def get_init_inputs():
    return []
