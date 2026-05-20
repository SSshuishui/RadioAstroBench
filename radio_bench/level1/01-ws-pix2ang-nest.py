"""Level 1: HEALPix NEST pixel index -> theta/phi.

This is a legacy/reference-style microbenchmark for the original MATLAB path:

    f_pix2ang_nest(nside, ipix) -> theta_heal, phi_heal

The optimized imaging kernels usually skip theta/phi and directly generate l/m/n.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

TASK_ID = "level1/01-ws-pix2ang-nest"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full"]

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> pix2ang_nest_forward(long long nside, long long base_ipix, long long chunkN);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ __forceinline__ float clamp01(float x) { return fminf(1.0f, fmaxf(-1.0f, x)); }
__device__ __forceinline__ void deinterleave_10(unsigned int ip, int &x, int &y) {
  x = 0; y = 0;
  #pragma unroll
  for (int b = 0; b < 5; ++b) {
    x |= ((ip >> (2*b))     & 1u) << b;
    y |= ((ip >> (2*b + 1)) & 1u) << b;
  }
}
__constant__ int c_jrll[12] = {2,2,2,2,3,3,3,3,4,4,4,4};
__constant__ int c_jpll[12] = {1,3,5,7,0,2,4,6,1,3,5,7};

__global__ void pix2ang_nest_kernel(
    int nside, unsigned int base_ipix, int chunkN,
    float* __restrict__ theta, float* __restrict__ phi) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= chunkN) return;
  unsigned int ipix = base_ipix + (unsigned int)tid;
  unsigned int npface = (unsigned int)nside * (unsigned int)nside;
  unsigned int face_num = ipix / npface;
  unsigned int ipf = ipix - face_num*npface;
  int ix = 0, iy = 0;
  unsigned int v = ipf;
  int scalemlv = 1;
  #pragma unroll
  for (int k = 0; k < 5; ++k) {
    unsigned int low = v & 1023u;
    int x, y;
    deinterleave_10(low, x, y);
    ix += scalemlv * x;
    iy += scalemlv * y;
    scalemlv <<= 5;
    v >>= 10;
  }
  {
    unsigned int low = v & 1023u;
    int x, y;
    deinterleave_10(low, x, y);
    ix += scalemlv * x;
    iy += scalemlv * y;
  }
  int jrt = ix + iy;
  int jpt = ix - iy;
  int nl4 = 4 * nside;
  int jr = c_jrll[face_num] * nside - jrt - 1;
  float fact1 = 1.0f / (3.0f * (float)nside * (float)nside);
  float fact2 = 2.0f / (3.0f * (float)nside);
  int nr, kshift;
  float z;
  if (jr < nside) {
    nr = jr;
    z = 1.0f - (float)(nr * nr) * fact1;
    kshift = 0;
  } else if (jr <= 3*nside) {
    nr = nside;
    z = (float)(2*nside - jr) * fact2;
    kshift = (jr - nside) & 1;
  } else {
    nr = nl4 - jr;
    z = -1.0f + (float)(nr * nr) * fact1;
    kshift = 0;
  }
  z = clamp01(z);
  float th = acosf(z);
  int jp = ((c_jpll[face_num]*nr) + jpt + 1 + kshift) >> 1;
  if (jp > nl4) jp -= nl4;
  if (jp < 1) jp += nl4;
  float ph = (0.5f * (float)M_PI) * ((float)jp - 0.5f*(float)(kshift + 1)) / (float)nr;
  if (ph < 0.0f) ph += 2.0f * (float)M_PI;
  theta[tid] = th;
  phi[tid] = ph;
}

std::vector<torch::Tensor> pix2ang_nest_forward(long long nside_ll, long long base_ipix_ll, long long chunkN_ll) {
  TORCH_CHECK(nside_ll > 0 && chunkN_ll > 0, "nside and chunkN must be positive");
  TORCH_CHECK(chunkN_ll <= 2147483647LL, "chunkN too large for this wrapper");
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto theta = torch::empty({chunkN_ll}, opts);
  auto phi = torch::empty({chunkN_ll}, opts);
  int block = 256;
  int grid = (int)((chunkN_ll + block - 1) / block);
  pix2ang_nest_kernel<<<grid, block>>>((int)nside_ll, (unsigned int)base_ipix_ll, (int)chunkN_ll,
                                       theta.data_ptr<float>(), phi.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {theta, phi};
}
"""

_EXT = None


def get_extension():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="rkb_ws_pix2ang_nest_baseline",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["pix2ang_nest_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT


class Model(nn.Module):
    def forward(self, nside: int, base_ipix: int, chunkN: int):
        return get_extension().pix2ang_nest_forward(int(nside), int(base_ipix), int(chunkN))


class ModelNew(Model):
    pass


def get_inputs(scale: str = "smoke"):
    if scale == "nside512_full":
        return [512, 0, 512 * 512 * 12]
    if scale == "nside4096_full":
        return [4096, 0, 4096 * 4096 * 12]
    return [64, 0, 4096]


def get_init_inputs():
    return []
