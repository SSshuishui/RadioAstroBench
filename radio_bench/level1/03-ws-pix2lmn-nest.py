"""Level 1 / task 001: pix2lmn_nest_kernel extracted from the original project.

Model: existing CUDA baseline.
ModelNew: identity candidate, identical to Model. This is only for smoke testing.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

TASK_ID = "level1/03-ws-pix2lmn-nest"

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> pix2lmn_forward(long long nside, long long base_ipix, long long chunkN);
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

#ifndef SINCOS_FAST_DEFINED
#define SINCOS_FAST_DEFINED
__device__ __forceinline__ void sincos_fast(float x, float* s, float* c) { __sincosf(x, s, c); }
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

// Original kernel: pix2lmn_nest_kernel
__global__ void pix2lmn_nest_kernel(
    int nside,
    unsigned int base_ipix,
    int chunkN,
    float* __restrict__ l,
    float* __restrict__ m,
    float* __restrict__ n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= chunkN) return;

  unsigned int ipix = base_ipix + (unsigned int)tid;
  unsigned int npface = (unsigned int)nside * (unsigned int)nside;
  unsigned int face_num = ipix / npface;
  unsigned int ipf      = ipix - face_num*npface;

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
  int jr  = c_jrll[face_num] * nside - jrt - 1;

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
  float theta = acosf(z);

  int jp = ((c_jpll[face_num]*nr) + jpt + 1 + kshift) >> 1;
  if (jp > nl4) jp -= nl4;
  if (jp < 1)   jp += nl4;

  float phi = (0.5f * (float)M_PI) * (((float)jp) - 0.5f*(float)(kshift + 1)) / (float)nr;
  if (phi < 0.0f) phi += 2.0f * (float)M_PI;

  float th = (float)M_PI * 0.5f - theta;
  if(phi > (float)M_PI) phi -= 2.0f*(float)M_PI;
  phi = -phi;

  float st, ct, sp, cp;
  sincos_fast(th, &st, &ct);
  sincos_fast(phi, &sp, &cp);

  l[tid] = ct * cp;
  m[tid] = ct * sp;
  n[tid] = st;
}

std::vector<torch::Tensor> pix2lmn_forward(long long nside_ll, long long base_ipix_ll, long long chunkN_ll) {
  TORCH_CHECK(nside_ll > 0 && chunkN_ll > 0, "nside and chunkN must be positive");
  TORCH_CHECK(chunkN_ll <= 2147483647LL, "chunkN too large for this test wrapper");
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto l = torch::empty({chunkN_ll}, opts);
  auto m = torch::empty({chunkN_ll}, opts);
  auto n = torch::empty({chunkN_ll}, opts);
  int block = 256;
  int grid = (int)((chunkN_ll + block - 1) / block);
  pix2lmn_nest_kernel<<<grid, block>>>((int)nside_ll, (unsigned int)base_ipix_ll, (int)chunkN_ll,
                                       l.data_ptr<float>(), m.data_ptr<float>(), n.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {l, m, n};
}
"""

_EXT = None

def get_extension():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="rkb_task001_pix2lmn_baseline",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["pix2lmn_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT

class Model(nn.Module):
    def forward(self, nside: int, base_ipix: int, chunkN: int):
        return get_extension().pix2lmn_forward(int(nside), int(base_ipix), int(chunkN))

class ModelNew(Model):
    pass


def get_inputs():
    return [64, 0, 4096]


def get_init_inputs():
    return []
