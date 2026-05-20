"""Level 2 / 3D task: compute half-baseline pair weights from DCF.

Extracted from ``compute_pair_weight_half_kernel`` in the 3D pipeline.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

TASK_ID = "level2/08-3d-pair-weight"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full", "nside16384_full"]

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor compute_pair_weight_forward(torch::Tensor u, torch::Tensor v, torch::Tensor w, torch::Tensor dcf);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

__global__ void compute_pair_weight_half_kernel(
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    int N_half,
    const float* __restrict__ dcf,
    int dcf_len,
    float* __restrict__ pair_weight_half)
{
  int ih = blockIdx.x * blockDim.x + threadIdx.x;
  if(ih >= N_half) return;
  float uu = u[ih], vv = v[ih], ww = w[ih];
  float bl = sqrtf(uu*uu + vv*vv + ww*ww);
  float wgt = 0.0f;
  if(dcf_len > 0){
    if(!isfinite(bl) || bl <= 1e-20f){
      wgt = dcf[0];
    } else {
      int gs = (int)ceilf((bl - 0.25f) / 0.5f) + 1;
      if(gs < 0) gs = 0;
      if(gs >= dcf_len) gs = dcf_len - 1;
      float s = ww / bl;
      s = fminf(1.0f, fmaxf(-1.0f, s));
      float c = sqrtf(fmaxf(0.0f, 1.0f - s*s));
      float a = 1.0f - 4.0f * s * s;
      float mag_sqrt = sqrtf(fabsf(a));
      float gdg = 0.0f;
      if(c > 1e-12f) gdg = 1.5f * mag_sqrt / c;
      wgt = dcf[gs] * gdg;
    }
  }
  if(!isfinite(wgt) || wgt < 0.0f) wgt = 0.0f;
  if(wgt > 0.125f) wgt = 0.125f;
  pair_weight_half[ih] = 2.0f * wgt;
}

torch::Tensor compute_pair_weight_forward(torch::Tensor u, torch::Tensor v, torch::Tensor w, torch::Tensor dcf) {
  TORCH_CHECK(u.is_cuda() && v.is_cuda() && w.is_cuda() && dcf.is_cuda(), "inputs must be CUDA tensors");
  u = u.contiguous(); v = v.contiguous(); w = w.contiguous(); dcf = dcf.contiguous();
  int N_half = (int)u.numel();
  auto out = torch::empty({N_half}, u.options());
  int block = 256;
  int grid = (N_half + block - 1) / block;
  compute_pair_weight_half_kernel<<<grid, block>>>(u.data_ptr<float>(), v.data_ptr<float>(), w.data_ptr<float>(), N_half, dcf.data_ptr<float>(), (int)dcf.numel(), out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""

_EXT = None

def get_extension():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="rkb_3d_compute_pair_weight_half",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["compute_pair_weight_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT

class Model(nn.Module):
    def forward(self, u, v, w, dcf):
        return get_extension().compute_pair_weight_forward(u, v, w, dcf)

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
    n = _n_half_for_scale(scale)
    t = torch.linspace(-1.0, 1.0, n, device=device, dtype=torch.float32)
    u = 700.0 * t
    v = 500.0 * torch.sin(t * 3.0)
    w = 350.0 * torch.cos(t * 2.0)
    dcf = torch.linspace(0.001, 0.12, 4096, device=device, dtype=torch.float32)
    return [u.contiguous(), v.contiguous(), w.contiguous(), dcf.contiguous()]


def get_init_inputs():
    return []
