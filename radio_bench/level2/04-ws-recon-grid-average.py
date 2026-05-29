"""Level 2 / task 004: recon_seg_keys_avg_half extracted from the original project."""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.common import make_unit_vectors

from radio_astronomy_cuda_bench.configs.scales import get_scale
from radio_astronomy_cuda_bench.common import make_unit_vectors_cuda
from radio_astronomy_cuda_bench.real_inputs import load_ws_grid_average_fixture

TASK_ID = "level2/04-ws-recon-grid-average"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full"]

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor recon_keys_avg_half_forward(
    torch::Tensor l, torch::Tensor m, torch::Tensor n,
    torch::Tensor keys_unique, torch::Tensor viss_avg,
    long long RES, long long half, double du, double fa, double fb);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

#ifndef SINCOS_FAST_DEFINED
#define SINCOS_FAST_DEFINED
__device__ __forceinline__ void sincos_fast(float x, float* s, float* c) { __sincosf(x, s, c); }
#endif

template<int CHUNK_KEYS>
__global__ void recon_seg_keys_avg_half(
    long long n_chunk,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const int* __restrict__ keys_unique,
    const float2* __restrict__ viss_avg,
    int nuniq,
    int RES, int half, float du,
    float fa, float fb,
    float* __restrict__ Cseg)
{
  __shared__ int shK[64];
  __shared__ float2 shV[64];
  long long pix=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(pix>=n_chunk) return;

  float lp=l[pix] + fa*n[pix];
  float mp=m[pix] + fb*n[pix];
  float acc=0.0f;
  const float TWO_PI=6.2831853071795864769f;

  for(int base=0;base<nuniq;base+=CHUNK_KEYS){
    int t=threadIdx.x;
    if(t<CHUNK_KEYS){
      int j=base+t;
      if(j<nuniq){ shK[t]=keys_unique[j]; shV[t]=viss_avg[j]; }
      else { shK[t]=0; shV[t]=make_float2(0,0); }
    }
    __syncthreads();
    #pragma unroll
    for(int k=0;k<CHUNK_KEYS;k++){
      int key=shK[k];
      if(key==0) continue;
      int tmp=key-1;
      int U=tmp/RES;
      int V=tmp-U*RES;
      int ui=U-half;
      int vi=V-half;
      float ugu=ui*du;
      float vgu=vi*du;
      float phase=TWO_PI*(ugu*lp + vgu*mp);
      float s,c; sincos_fast(phase,&s,&c);
      float2 vv=shV[k];
      acc += vv.x*c - vv.y*s;
    }
    __syncthreads();
  }
  Cseg[pix]=acc;
}

torch::Tensor recon_keys_avg_half_forward(
    torch::Tensor l, torch::Tensor m, torch::Tensor n,
    torch::Tensor keys_unique, torch::Tensor viss_avg,
    long long RES_ll, long long half_ll, double du_d, double fa_d, double fb_d) {
  TORCH_CHECK(l.is_cuda(), "inputs must be CUDA tensors");
  l=l.contiguous(); m=m.contiguous(); n=n.contiguous(); keys_unique=keys_unique.contiguous(); viss_avg=viss_avg.contiguous();
  long long n_chunk = l.numel();
  int nuniq = (int)keys_unique.numel();
  auto Cseg = torch::empty({n_chunk}, l.options());
  int block = 128;
  int grid = (int)((n_chunk + block - 1) / block);
  recon_seg_keys_avg_half<64><<<grid, block>>>(
      n_chunk, l.data_ptr<float>(), m.data_ptr<float>(), n.data_ptr<float>(),
      keys_unique.data_ptr<int>(), reinterpret_cast<float2*>(viss_avg.data_ptr<float>()),
      nuniq, (int)RES_ll, (int)half_ll, (float)du_d, (float)fa_d, (float)fb_d, Cseg.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return Cseg;
}
"""

_EXT = None

def get_extension():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="rkb_task204_recon_keys_avg_half_baseline",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["recon_keys_avg_half_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT

class Model(nn.Module):
    def forward(self, l, m, n, keys_unique, viss_avg, RES: int, half: int, du: float, fa: float, fb: float):
        return get_extension().recon_keys_avg_half_forward(l, m, n, keys_unique, viss_avg, int(RES), int(half), float(du), float(fa), float(fb))

class ModelNew(Model):
    pass


def get_inputs(scale: str = "smoke", segment_profile: str = "all10", fixture: str | None = None):
    if fixture:
        return load_ws_grid_average_fixture(fixture, scale=scale, tile_pix=256)
    device = "cuda"
    cfg = get_scale(scale)
    n_chunk = int(cfg.npix)
    RES = 64 if scale == "smoke" else 512
    half = RES // 2
    nuniq = 256 if scale == "smoke" else 8192
    l, m, n = make_unit_vectors_cuda(n_chunk, device=device, seed=12)
    keys_unique = torch.linspace(1, RES * RES, nuniq, device=device, dtype=torch.float32).round().to(torch.int32)
    phase = torch.linspace(0.0, 6.2831853, nuniq, device=device)
    viss_avg = torch.stack([torch.cos(phase), torch.sin(phase)], dim=1).to(torch.float32).contiguous()
    du = 0.05
    fa = 0.08
    fb = -0.05
    return [l, m, n, keys_unique, viss_avg, RES, half, du, fa, fb]


def get_init_inputs():
    return []
