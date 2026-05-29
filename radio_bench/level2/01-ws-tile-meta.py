"""Level 2 / task 001: build_tile_cone_meta_kernel extracted from the original project."""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.common import make_unit_vectors

from radio_astronomy_cuda_bench.configs.scales import get_scale
from radio_astronomy_cuda_bench.common import make_unit_vectors_cuda

TASK_ID = "level2/01-ws-tile-meta"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full"]
TILE_PIX = 256

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> build_tile_cone_meta_forward(torch::Tensor l, torch::Tensor m, torch::Tensor n, torch::Tensor nm1);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

template<int TILE_PIX>
__global__ void build_tile_cone_meta_kernel(const float* __restrict__ l,
                                            const float* __restrict__ m,
                                            const float* __restrict__ n,
                                            const float* __restrict__ nm1,
                                            long long n_chunk,
                                            float* __restrict__ tile_cx,
                                            float* __restrict__ tile_cy,
                                            float* __restrict__ tile_cz,
                                            float* __restrict__ tile_cosA,
                                            float* __restrict__ tile_sinA,
                                            int ntile)
{
  int tid = blockIdx.x;
  if(tid >= ntile) return;
  long long p0 = (long long)tid * TILE_PIX;
  int tileN = (int)min((long long)TILE_PIX, n_chunk - p0);
  if(tileN <= 0) return;

  float sx=0.0f, sy=0.0f, sz=0.0f;
  for(int k=threadIdx.x; k<tileN; k+=blockDim.x){
    long long p = p0 + k;
    sx += l[p]; sy += m[p]; sz += n[p];
  }
  __shared__ float rsx[256], rsy[256], rsz[256];
  rsx[threadIdx.x]=sx; rsy[threadIdx.x]=sy; rsz[threadIdx.x]=sz;
  __syncthreads();
  for(int off=blockDim.x>>1; off>0; off>>=1){
    if(threadIdx.x < off){
      rsx[threadIdx.x] += rsx[threadIdx.x + off];
      rsy[threadIdx.x] += rsy[threadIdx.x + off];
      rsz[threadIdx.x] += rsz[threadIdx.x + off];
    }
    __syncthreads();
  }
  __shared__ float cx,cy,cz;
  if(threadIdx.x==0){
    float inv = rsqrtf(rsx[0]*rsx[0] + rsy[0]*rsy[0] + rsz[0]*rsz[0] + 1e-30f);
    cx = rsx[0]*inv; cy = rsy[0]*inv; cz = rsz[0]*inv;
  }
  __syncthreads();

  float mind = 1.0f;
  for(int k=threadIdx.x; k<tileN; k+=blockDim.x){
    long long p = p0 + k;
    float d = cx*l[p] + cy*m[p] + cz*n[p];
    mind = fminf(mind, d);
  }
  __shared__ float rmin[256];
  rmin[threadIdx.x] = mind;
  __syncthreads();
  for(int off=blockDim.x>>1; off>0; off>>=1){
    if(threadIdx.x < off) rmin[threadIdx.x] = fminf(rmin[threadIdx.x], rmin[threadIdx.x + off]);
    __syncthreads();
  }
  if(threadIdx.x==0){
    float cA = fminf(1.0f, fmaxf(-1.0f, rmin[0]));
    tile_cx[tid]=cx; tile_cy[tid]=cy; tile_cz[tid]=cz;
    tile_cosA[tid]=cA;
    tile_sinA[tid]=sqrtf(fmaxf(0.0f, 1.0f - cA*cA));
  }
}

std::vector<torch::Tensor> build_tile_cone_meta_forward(torch::Tensor l, torch::Tensor m, torch::Tensor n, torch::Tensor nm1) {
  TORCH_CHECK(l.is_cuda() && m.is_cuda() && n.is_cuda() && nm1.is_cuda(), "all inputs must be CUDA tensors");
  TORCH_CHECK(l.dtype() == torch::kFloat32 && m.dtype() == torch::kFloat32 && n.dtype() == torch::kFloat32 && nm1.dtype() == torch::kFloat32, "all inputs must be float32");
  l = l.contiguous(); m = m.contiguous(); n = n.contiguous(); nm1 = nm1.contiguous();
  long long n_chunk = l.numel();
  TORCH_CHECK(m.numel() == n_chunk && n.numel() == n_chunk && nm1.numel() == n_chunk, "input size mismatch");
  const int TILE = 256;
  int ntile = (int)((n_chunk + TILE - 1) / TILE);
  auto opts = l.options();
  auto tile_cx = torch::empty({ntile}, opts);
  auto tile_cy = torch::empty({ntile}, opts);
  auto tile_cz = torch::empty({ntile}, opts);
  auto tile_cosA = torch::empty({ntile}, opts);
  auto tile_sinA = torch::empty({ntile}, opts);
  build_tile_cone_meta_kernel<256><<<ntile, 256>>>(
      l.data_ptr<float>(), m.data_ptr<float>(), n.data_ptr<float>(), nm1.data_ptr<float>(), n_chunk,
      tile_cx.data_ptr<float>(), tile_cy.data_ptr<float>(), tile_cz.data_ptr<float>(),
      tile_cosA.data_ptr<float>(), tile_sinA.data_ptr<float>(), ntile);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {tile_cx, tile_cy, tile_cz, tile_cosA, tile_sinA};
}
"""

_EXT = None

def get_extension():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="rkb_task201_tile_cone_meta_baseline",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["build_tile_cone_meta_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT

class Model(nn.Module):
    def forward(self, l: torch.Tensor, m: torch.Tensor, n: torch.Tensor, nm1: torch.Tensor):
        return get_extension().build_tile_cone_meta_forward(l, m, n, nm1)

class ModelNew(Model):
    pass


def get_inputs(scale: str = "smoke", segment_profile: str = "all10", fixture: str | None = None):
    cfg = get_scale(scale)
    n_chunk = (int(cfg.npix) // TILE_PIX) * TILE_PIX
    l, m, n = make_unit_vectors_cuda(n_chunk, device="cuda", seed=3)
    nm1 = (n - 1.0).contiguous()
    return [l, m, n, nm1]


def get_init_inputs():
    return []
