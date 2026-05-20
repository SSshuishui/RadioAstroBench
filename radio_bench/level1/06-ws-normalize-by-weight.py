"""Level 1 / task 003: normalize_by_weight extracted from the original project."""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.configs.scales import get_scale

TASK_ID = "level1/06-ws-normalize-by-weight"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full"]

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor normalize_by_weight_forward(torch::Tensor C, torch::Tensor W);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

__global__ void normalize_by_weight(float* C,const uint32_t* W,long long n){
  long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){
    uint32_t w=W[i];
    C[i]= (w>0)? (C[i]/(float)w) : 0.0f;
  }
}

torch::Tensor normalize_by_weight_forward(torch::Tensor C, torch::Tensor W) {
  TORCH_CHECK(C.is_cuda() && W.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
  TORCH_CHECK(W.dtype() == torch::kUInt32, "W must be uint32");
  C = C.contiguous(); W = W.contiguous();
  auto out = C.clone();
  long long n = out.numel();
  int block = 256;
  int grid = (int)((n + block - 1) / block);
  normalize_by_weight<<<grid, block>>>(out.data_ptr<float>(), (const uint32_t*)W.data_ptr<unsigned int>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""

_EXT = None

def get_extension():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="rkb_task003_normalize_by_weight_baseline",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["normalize_by_weight_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT

class Model(nn.Module):
    def forward(self, C: torch.Tensor, W: torch.Tensor):
        return get_extension().normalize_by_weight_forward(C, W)

class ModelNew(Model):
    pass


def get_inputs(scale: str = "smoke", segment_profile: str = "all10", fixture: str | None = None):
    device = "cuda"
    cfg = get_scale(scale)
    n = cfg.npix
    C = torch.linspace(-10, 10, n, device=device, dtype=torch.float32)
    W = (torch.arange(n, device=device, dtype=torch.int64) % 17).to(torch.uint32)
    return [C, W]

def describe_inputs(scale: str = "smoke", segment_profile: str = "all10", fixture: str | None = None):
    cfg = get_scale(scale)
    return {"scale": scale, "nside": cfg.nside, "npix": cfg.npix, "mode": "full-array normalize by weight"}


def get_init_inputs():
    return []
