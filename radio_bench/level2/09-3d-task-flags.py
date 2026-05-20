"""Level 2 / 3D task: build task flags for plan-aware 3D reconstruction.

Extracted from ``build_recon_task_flags_chunk_kernel``. It classifies each
(tile, baseline-block) task as all-visible (vv) or mixed for load-balanced 3D reconstruction.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.common import make_unit_vectors, make_endpoint_vectors, make_tile_meta_torch

TASK_ID = "level2/09-3d-task-flags"
SUPPORTED_SCALES = ["smoke", "nside512_full"]
TILE_PIX = 128
TILE_BL = 64

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> build_recon_task_flags_forward(
    torch::Tensor l, torch::Tensor m, torch::Tensor n,
    torch::Tensor tile_cx, torch::Tensor tile_cy, torch::Tensor tile_cz,
    torch::Tensor tile_cosA, torch::Tensor tile_sinA,
    torch::Tensor x1, torch::Tensor y1, torch::Tensor z1,
    torch::Tensor x2, torch::Tensor y2, torch::Tensor z2,
    double cosphi);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>

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

template<int TILE_BL>
__global__ void build_recon_task_flags_chunk_kernel(
    int ntile,
    const float* __restrict__ tile_cx,
    const float* __restrict__ tile_cy,
    const float* __restrict__ tile_cz,
    const float* __restrict__ tile_cosA,
    const float* __restrict__ tile_sinA,
    const float* __restrict__ x1,
    const float* __restrict__ y1,
    const float* __restrict__ z1,
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    int b0,
    int chunk_n,
    float cosphi,
    unsigned char* __restrict__ vv_flags,
    unsigned char* __restrict__ mixed_flags)
{
  int nblocks = (chunk_n + TILE_BL - 1) / TILE_BL;
  int task_count = ntile * nblocks;
  int task = blockIdx.x * blockDim.x + threadIdx.x;
  if(task >= task_count) return;

  int blk = task / ntile;
  int tile = task - blk * ntile;
  int base = b0 + blk * TILE_BL;
  int tileN = min(TILE_BL, chunk_n - blk * TILE_BL);

  float cx = tile_cx[tile], cy = tile_cy[tile], cz = tile_cz[tile];
  float cosA = tile_cosA[tile], sinA = tile_sinA[tile];

  bool all_visible_all = true;
  bool all_hidden_all = true;
  for(int t=0; t<tileN; ++t){
    int ih = base + t;
    bool allv=false, allh=false;
    operator_classify_pair_tile_dirs(x1[ih], y1[ih], z1[ih], x2[ih], y2[ih], z2[ih], cx, cy, cz, cosA, sinA, cosphi, allv, allh);
    if(!allv) all_visible_all = false;
    if(!allh) all_hidden_all = false;
    if(!all_visible_all && !all_hidden_all) break;
  }
  vv_flags[task] = all_visible_all ? (unsigned char)1 : (unsigned char)0;
  mixed_flags[task] = (!all_visible_all && !all_hidden_all) ? (unsigned char)1 : (unsigned char)0;
}

std::vector<torch::Tensor> build_recon_task_flags_forward(
    torch::Tensor l, torch::Tensor m, torch::Tensor n,
    torch::Tensor tile_cx, torch::Tensor tile_cy, torch::Tensor tile_cz,
    torch::Tensor tile_cosA, torch::Tensor tile_sinA,
    torch::Tensor x1, torch::Tensor y1, torch::Tensor z1,
    torch::Tensor x2, torch::Tensor y2, torch::Tensor z2,
    double cosphi_d) {
  TORCH_CHECK(tile_cx.is_cuda() && x1.is_cuda(), "inputs must be CUDA tensors");
  int ntile = (int)tile_cx.numel();
  int chunk_n = (int)x1.numel();
  int nblocks = (chunk_n + 64 - 1) / 64;
  int task_count = ntile * nblocks;
  auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(tile_cx.device());
  auto vv = torch::empty({task_count}, opts);
  auto mixed = torch::empty({task_count}, opts);
  int block = 256;
  int grid = (task_count + block - 1) / block;
  build_recon_task_flags_chunk_kernel<64><<<grid, block>>>(
      ntile,
      tile_cx.data_ptr<float>(), tile_cy.data_ptr<float>(), tile_cz.data_ptr<float>(),
      tile_cosA.data_ptr<float>(), tile_sinA.data_ptr<float>(),
      x1.data_ptr<float>(), y1.data_ptr<float>(), z1.data_ptr<float>(),
      x2.data_ptr<float>(), y2.data_ptr<float>(), z2.data_ptr<float>(),
      0, chunk_n, (float)cosphi_d,
      vv.data_ptr<unsigned char>(), mixed.data_ptr<unsigned char>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {vv, mixed};
}
"""

_EXT = None

def get_extension():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="rkb_3d_build_recon_task_flags_chunk",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["build_recon_task_flags_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT

class Model(nn.Module):
    def forward(self, l, m, n, tile_cx, tile_cy, tile_cz, tile_cosA, tile_sinA, x1, y1, z1, x2, y2, z2, cosphi: float):
        return get_extension().build_recon_task_flags_forward(l, m, n, tile_cx, tile_cy, tile_cz, tile_cosA, tile_sinA, x1, y1, z1, x2, y2, z2, float(cosphi))

class ModelNew(Model):
    pass


def _n_pix(scale: str) -> int:
    return 8192 if scale == "smoke" else 512 * 512 * 12


def get_inputs(scale: str = "smoke", **_):
    device = "cuda"
    n_pix = _n_pix(scale)
    # Keep smoke/light search divisible by TILE_PIX.
    n_pix = (n_pix // TILE_PIX) * TILE_PIX
    l, m, n = make_unit_vectors(n_pix, device=device, seed=201)
    tile_cx, tile_cy, tile_cz, tile_cosA, tile_sinA = make_tile_meta_torch(l, m, n, tile_pix=TILE_PIX)
    n_half = 4096 if scale == "smoke" else 32768
    x1, y1, z1, _ = make_endpoint_vectors(n_half, device=device, seed=202)
    x2, y2, z2, _ = make_endpoint_vectors(n_half, device=device, seed=203)
    cosphi = 0.15
    return [l, m, n, tile_cx, tile_cy, tile_cz, tile_cosA, tile_sinA, x1, y1, z1, x2, y2, z2, cosphi]


def get_init_inputs():
    return []
