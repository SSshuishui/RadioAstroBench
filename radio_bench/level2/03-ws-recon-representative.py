"""Level 2 / task 003: recon_seg_blockage_half_rep extracted from the original project.

This is the representative-group reconstruction kernel. Model is the existing CUDA baseline;
ModelNew is an identity candidate for initial smoke testing.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.common import make_unit_vectors, make_float4

from radio_astronomy_cuda_bench.configs.scales import get_scale
from radio_astronomy_cuda_bench.common import make_unit_vectors_cuda
from radio_astronomy_cuda_bench.real_inputs import load_ws_recon_rep_fixture

TASK_ID = "level2/03-ws-recon-representative"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full"]

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> recon_blockage_half_rep_forward(
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

std::vector<torch::Tensor> recon_blockage_half_rep_forward(
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
  int block = 128;
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
            name="rkb_task203_recon_blockage_half_rep_baseline",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["recon_blockage_half_rep_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT

class Model(nn.Module):
    def forward(self, l, m, n, keys_unique, ugu_list, vgu_list, repV, repP1, repP2, fa: float, fb: float, cosphi: float):
        return get_extension().recon_blockage_half_rep_forward(
            l, m, n, keys_unique, ugu_list, vgu_list, repV, repP1, repP2, float(fa), float(fb), float(cosphi)
        )

class ModelNew(Model):
    pass


def get_inputs(scale: str = "smoke", segment_profile: str = "all10", fixture: str | None = None):
    if fixture:
        return load_ws_recon_rep_fixture(fixture, scale=scale, tile_pix=256)
    device = "cuda"
    cfg = get_scale(scale)
    n_chunk = int(cfg.npix)
    nuniq = 256 if scale == "smoke" else 8192
    l, m, n = make_unit_vectors_cuda(n_chunk, device=device, seed=8)
    keys_unique = torch.arange(1, nuniq + 1, device=device, dtype=torch.int32)
    ugu_list = torch.linspace(-8.0, 8.0, nuniq, device=device, dtype=torch.float32)
    vgu_list = torch.linspace(7.0, -7.0, nuniq, device=device, dtype=torch.float32)
    phase = torch.linspace(0, 6.2831853, nuniq, device=device)
    repV = torch.stack([torch.cos(phase), torch.sin(phase)], dim=1).to(torch.float32).contiguous()
    repP1 = make_float4(nuniq, device=device, seed=9)
    repP2 = make_float4(nuniq, device=device, seed=10)
    fa = 0.08
    fb = -0.05
    cosphi = 0.15
    return [l, m, n, keys_unique, ugu_list, vgu_list, repV, repP1, repP2, fa, fb, cosphi]


def get_init_inputs():
    return []
