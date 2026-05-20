"""Level 1: HEALPix RING pixel index -> theta/phi.

This is the RING-order reference-style microbenchmark for the original MATLAB path.
The optimized pipeline generally skips these intermediate theta/phi arrays and uses
``01-ws-pix2lmn-ring.py`` or ``01-ws-pix2lmn-nest.py`` instead.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

TASK_ID = "level1/02-ws-pix2ang-ring"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full"]

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> pix2ang_ring_forward(long long nside, long long base_ipix, long long chunkN);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ __forceinline__ void correct_ring_phi_dev(int location, long long& iring, long long& iphi) {
  long long delta = 0;
  if (iphi < 0) delta = 1;
  if (iphi > 4 * iring) delta = -1;
  if (delta != 0) {
    iring = iring - (long long)location * delta;
    iphi  = iphi  + delta * (4 * iring);
  }
}

__global__ void pix2ang_ring_kernel(
    int nside,
    unsigned long long base_ipix,
    int chunkN,
    float* __restrict__ theta_out,
    float* __restrict__ phi_out)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= chunkN) return;
  unsigned long long ipix_u = base_ipix + (unsigned long long)tid;
  long long ipix = (long long)ipix_u;
  const long long nl2  = 2ll * (long long)nside;
  const long long nl4  = 4ll * (long long)nside;
  const long long npix = 12ll * (long long)nside * (long long)nside;
  const long long nCap = nl2 * ((long long)nside - 1ll);
  double theta = 0.0, phi = 0.0;
  if (ipix < nCap) {
    long long iPixM = ipix;
    double x = ((double)(iPixM + 1ll)) / 2.0;
    long long iRing = (long long)floor(sqrt(x) + 0.5);
    long long iPhi = iPixM - 2ll * iRing * (iRing - 1ll);
    correct_ring_phi_dev(+1, iRing, iPhi);
    double t = ((double)iRing / (double)nside);
    double z = 1.0 - (t * t) / 3.0;
    z = fmax(-1.0, fmin(1.0, z));
    theta = acos(z);
    phi = (M_PI * 0.5) * ((double)iPhi + 0.5) / (double)iRing;
  } else if (ipix < (npix - nCap)) {
    long long ipM = ipix - nCap;
    long long iRing = (ipM / nl4) + (long long)nside;
    long long iPhi = ipM % nl4;
    double fodd = 0.5 * (double)((iRing + (long long)nside + 1ll) & 1ll);
    double z = ((double)(nl2 - iRing)) / (1.5 * (double)nside);
    z = fmax(-1.0, fmin(1.0, z));
    theta = acos(z);
    phi = (M_PI * 0.5) * ((double)iPhi + fodd) / (double)nside;
  } else {
    long long ipM = npix - ipix;
    double x = ((double)ipM) / 2.0;
    long long iRing = (long long)floor(sqrt(x) + 0.5);
    long long iPhi = 2ll * iRing * (iRing + 1ll) - ipM;
    correct_ring_phi_dev(-1, iRing, iPhi);
    double t = ((double)iRing / (double)nside);
    double z = (t * t) / 3.0 - 1.0;
    z = fmax(-1.0, fmin(1.0, z));
    theta = acos(z);
    phi = (M_PI * 0.5) * ((double)iPhi + 0.5) / (double)iRing;
  }
  theta_out[tid] = (float)theta;
  phi_out[tid] = (float)phi;
}

std::vector<torch::Tensor> pix2ang_ring_forward(long long nside_ll, long long base_ipix_ll, long long chunkN_ll) {
  TORCH_CHECK(nside_ll > 0 && chunkN_ll > 0, "nside and chunkN must be positive");
  TORCH_CHECK(chunkN_ll <= 2147483647LL, "chunkN too large for this wrapper");
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto theta = torch::empty({chunkN_ll}, opts);
  auto phi = torch::empty({chunkN_ll}, opts);
  int block = 256;
  int grid = (int)((chunkN_ll + block - 1) / block);
  pix2ang_ring_kernel<<<grid, block>>>((int)nside_ll, (unsigned long long)base_ipix_ll, (int)chunkN_ll,
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
            name="rkb_ws_pix2ang_ring_baseline",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["pix2ang_ring_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT


class Model(nn.Module):
    def forward(self, nside: int, base_ipix: int, chunkN: int):
        return get_extension().pix2ang_ring_forward(int(nside), int(base_ipix), int(chunkN))


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
