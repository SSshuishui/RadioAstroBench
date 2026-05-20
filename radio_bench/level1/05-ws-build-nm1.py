"""Level 1 / task 002: build_nm1_kernel extracted from the original project."""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.common import make_unit_vectors, make_unit_vectors_cuda
from radio_astronomy_cuda_bench.configs.scales import get_scale

TASK_ID = "level1/05-ws-build-nm1"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full", "nside16384_full"]

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor build_nm1_forward(torch::Tensor n);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

__global__ void build_nm1_kernel(const float* __restrict__ n,
                                 float* __restrict__ nm1,
                                 long long n_chunk)
{
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_chunk) nm1[idx] = n[idx] - 1.0f;
}

torch::Tensor build_nm1_forward(torch::Tensor n) {
  TORCH_CHECK(n.is_cuda(), "n must be CUDA tensor");
  TORCH_CHECK(n.dtype() == torch::kFloat32, "n must be float32");
  n = n.contiguous();
  auto nm1 = torch::empty_like(n);
  long long n_chunk = n.numel();
  int block = 256;
  int grid = (int)((n_chunk + block - 1) / block);
  build_nm1_kernel<<<grid, block>>>(n.data_ptr<float>(), nm1.data_ptr<float>(), n_chunk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return nm1;
}
"""

_EXT = None

def get_extension():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="rkb_task002_build_nm1_baseline",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["build_nm1_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT

class Model(nn.Module):
    def forward(self, n: torch.Tensor):
        return get_extension().build_nm1_forward(n)

class ModelNew(Model):
    pass


def get_inputs(scale: str = "smoke", segment_profile: str = "all10", fixture: str | None = None):
    cfg = get_scale(scale)
    _, _, n = make_unit_vectors_cuda(cfg.npix, device="cuda", seed=2)
    return [n]

def describe_inputs(scale: str = "smoke", segment_profile: str = "all10", fixture: str | None = None):
    cfg = get_scale(scale)
    return {"scale": scale, "nside": cfg.nside, "npix": cfg.npix, "mode": "full-array nm1 generation"}


def get_init_inputs():
    return []
