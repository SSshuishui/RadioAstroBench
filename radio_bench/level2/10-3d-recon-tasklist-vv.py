"""Level 2 / 3D task: direct 3D reconstruction VV task-list kernel.

Extracted from ``recon_3d_direct_tasklist_vv_real``.  This is the all-visible
fast path used by the 3D inverse imaging stage after task-list planning.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.common import make_unit_vectors

TASK_ID = "level2/10-3d-recon-tasklist-vv"
SUPPORTED_SCALES = ["smoke", "nside512_full"]
TILE_PIX = 128
TILE_BL = 64

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor recon_3d_vv_forward(
    torch::Tensor l, torch::Tensor m, torch::Tensor n,
    torch::Tensor u, torch::Tensor v, torch::Tensor w,
    torch::Tensor pair_weight_half, torch::Tensor Viss_half,
    torch::Tensor task_ids, int ntile, int b0, int chunk_n);
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

template<int TILE_PIX, int TILE_BL>
__global__ void recon_3d_direct_tasklist_vv_real(
    long long n_chunk,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    int ntile,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ pair_weight_half,
    const float2* __restrict__ Viss_half,
    int b0,
    int chunk_n,
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
  __shared__ float spw[TILE_BL];
  __shared__ float2 sV[TILE_BL];

  int lane = threadIdx.x;
  long long pix = (long long)tile * TILE_PIX + lane;
  bool active = (pix < n_chunk);

  float lp=0.0f, mp=0.0f, npv=0.0f;
  if(active){ lp=l[pix]; mp=m[pix]; npv=n[pix]; }

  for(int t=lane; t<tileBL; t+=blockDim.x){
    int ih = base + t;
    su[t] = u[ih]; sv[t] = v[ih]; sw[t] = w[ih];
    spw[t] = pair_weight_half[ih];
    sV[t] = Viss_half[ih];
  }
  __syncthreads();

  float acc_re = 0.0f;
  if(active){
    #pragma unroll 2
    for(int t=0; t<tileBL; ++t){
      float pairw = spw[t];
      if(pairw <= 0.0f) continue;
      float phase = operator_adjoint_phase(su[t], sv[t], sw[t], lp, mp, npv);
      float s, c; sincos_fast(phase, &s, &c);
      float2 z = sV[t];
      if(!isfinite(z.x) || !isfinite(z.y)) continue;
      acc_re += pairw * (z.x * c - z.y * s);
    }
  }
  if(active) Cacc[pix] += acc_re;
}

torch::Tensor recon_3d_vv_forward(
    torch::Tensor l, torch::Tensor m, torch::Tensor n,
    torch::Tensor u, torch::Tensor v, torch::Tensor w,
    torch::Tensor pair_weight_half, torch::Tensor Viss_half,
    torch::Tensor task_ids, int ntile, int b0, int chunk_n) {
  TORCH_CHECK(l.is_cuda() && u.is_cuda() && task_ids.is_cuda(), "inputs must be CUDA tensors");
  l=l.contiguous(); m=m.contiguous(); n=n.contiguous();
  u=u.contiguous(); v=v.contiguous(); w=w.contiguous(); pair_weight_half=pair_weight_half.contiguous(); Viss_half=Viss_half.contiguous(); task_ids=task_ids.contiguous();
  long long n_chunk = l.numel();
  int num_tasks = (int)task_ids.numel();
  auto Cacc = torch::zeros({n_chunk}, l.options());
  recon_3d_direct_tasklist_vv_real<128,64><<<num_tasks,128>>>(
      n_chunk, l.data_ptr<float>(), m.data_ptr<float>(), n.data_ptr<float>(), ntile,
      u.data_ptr<float>(), v.data_ptr<float>(), w.data_ptr<float>(),
      pair_weight_half.data_ptr<float>(), reinterpret_cast<float2*>(Viss_half.data_ptr<float>()),
      b0, chunk_n, task_ids.data_ptr<int>(), num_tasks, Cacc.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return Cacc;
}
"""

_EXT = None

def get_extension():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="rkb_3d_recon_direct_tasklist_vv_real",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["recon_3d_vv_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT

class Model(nn.Module):
    def forward(self, l, m, n, u, v, w, pair_weight_half, Viss_half, task_ids, ntile: int, b0: int, chunk_n: int):
        return get_extension().recon_3d_vv_forward(l, m, n, u, v, w, pair_weight_half, Viss_half, task_ids, int(ntile), int(b0), int(chunk_n))

class ModelNew(Model):
    pass


def _n_pix(scale: str) -> int:
    return 8192 if scale == "smoke" else 512 * 512 * 12


def get_inputs(scale: str = "smoke", **_):
    device = "cuda"
    n_pix = (_n_pix(scale) // TILE_PIX) * TILE_PIX
    l, m, n = make_unit_vectors(n_pix, device=device, seed=301)
    ntile = n_pix // TILE_PIX
    chunk_n = 1024 if scale == "smoke" else 4096
    u = torch.linspace(-700.0, 700.0, chunk_n, device=device, dtype=torch.float32)
    v = torch.sin(torch.linspace(0, 7.0, chunk_n, device=device)) * 500.0
    w = torch.cos(torch.linspace(0, 5.0, chunk_n, device=device)) * 350.0
    pair_weight_half = torch.full((chunk_n,), 0.02, device=device, dtype=torch.float32)
    phase = torch.linspace(0, 6.2831853, chunk_n, device=device)
    Viss_half = torch.stack([torch.cos(phase), torch.sin(phase)], dim=1).contiguous().to(torch.float32)
    # Keep task count modest for smoke/search; task IDs encode (baseline-block, tile).
    nblocks = (chunk_n + TILE_BL - 1) // TILE_BL
    num_tasks = min(ntile * nblocks, 4096 if scale == "smoke" else 32768)
    task_ids = torch.arange(num_tasks, device=device, dtype=torch.int32)
    return [l, m, n, u, v, w, pair_weight_half, Viss_half, task_ids, ntile, 0, chunk_n]


def get_init_inputs():
    return []
