"""Level 2 / 3D task: reduce partial visibility parts and apply half-symmetry phase.

Extracted from the 3D pipeline kernel ``reduce_phase_halfsym_fused_kernel``.
Model is the existing CUDA baseline; ModelNew is an identity candidate.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

TASK_ID = "level2/07-3d-phase-reduce"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full"]

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor reduce_phase_forward(torch::Tensor reduce_parts, torch::Tensor w_half);
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

__global__ void reduce_phase_halfsym_fused_kernel(
    const float2* __restrict__ reduce_parts,
    int parts_stride,
    int parts_count,
    const float* __restrict__ w_half,
    int N_half,
    float2* __restrict__ Viss_half_out)
{
  int ih = blockIdx.x * blockDim.x + threadIdx.x;
  if(ih >= N_half) return;
  float re = 0.0f;
  float im = 0.0f;
  for(int p=0; p<parts_count; ++p){
    float2 z = reduce_parts[(size_t)p * (size_t)parts_stride + (size_t)ih];
    re += z.x;
    im += z.y;
  }
  float ang = -2.0f * (float)M_PI * w_half[ih];
  float s, c;
  sincos_fast(ang, &s, &c);
  Viss_half_out[ih] = make_float2(re*c - im*s, re*s + im*c);
}

torch::Tensor reduce_phase_forward(torch::Tensor reduce_parts, torch::Tensor w_half) {
  TORCH_CHECK(reduce_parts.is_cuda() && w_half.is_cuda(), "inputs must be CUDA tensors");
  reduce_parts = reduce_parts.contiguous();
  w_half = w_half.contiguous();
  TORCH_CHECK(reduce_parts.dim() == 3 && reduce_parts.size(2) == 2, "reduce_parts must be [parts, N_half, 2]");
  int parts_count = (int)reduce_parts.size(0);
  int N_half = (int)reduce_parts.size(1);
  auto out = torch::empty({N_half, 2}, reduce_parts.options());
  int block = 256;
  int grid = (N_half + block - 1) / block;
  reduce_phase_halfsym_fused_kernel<<<grid, block>>>(
      reinterpret_cast<const float2*>(reduce_parts.data_ptr<float>()),
      N_half, parts_count, w_half.data_ptr<float>(), N_half,
      reinterpret_cast<float2*>(out.data_ptr<float>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""

_EXT = None

def get_extension():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="rkb_3d_reduce_phase_halfsym_fused",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["reduce_phase_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT

class Model(nn.Module):
    def forward(self, reduce_parts, w_half):
        return get_extension().reduce_phase_forward(reduce_parts, w_half)

class ModelNew(Model):
    pass


def _n_half_for_scale(scale: str) -> int:
    if scale == "smoke":
        return 4096
    if scale == "nside512_full":
        return 32768
    return 195468


def get_inputs(scale: str = "smoke", **_):
    device = "cuda"
    n_half = _n_half_for_scale(scale)
    parts = 4
    idx = torch.arange(n_half, device=device, dtype=torch.float32)
    w_half = torch.sin(idx * 0.001) * 100.0
    p = torch.arange(parts, device=device, dtype=torch.float32).view(parts, 1)
    re = torch.cos(idx.view(1, -1) * 0.01 + p)
    im = torch.sin(idx.view(1, -1) * 0.01 + p * 0.5)
    reduce_parts = torch.stack([re, im], dim=2).contiguous().to(torch.float32)
    return [reduce_parts, w_half]


def get_init_inputs():
    return []
