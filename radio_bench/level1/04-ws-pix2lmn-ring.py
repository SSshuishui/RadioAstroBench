"""Level 1: HEALPix RING pixel index -> direction cosines (l, m, n).

This task uses the direct RING-order CUDA kernel provided by the user.
It fuses the original MATLAB-style path

    pix2ang_ring -> theta/phi -> latitude/longitude -> l/m/n

into a single online-generation kernel that returns l/m/n directly.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

TASK_ID = "level1/04-ws-pix2lmn-ring"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full"]

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> pix2lmn_ring_forward(long long nside, long long base_ipix, long long chunkN);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <math.h>
#include <stdint.h>

// User-provided optimized RING-order direct l/m/n kernel.
// It uses the HEALPix RING 1-based indexing formula internally and directly
// returns direction cosines, avoiding materialized theta/phi arrays.
__global__ void pix2lmn_ring_kernel(
    int nside,
    unsigned int base_ipix,
    int chunkN,
    float* __restrict__ l,
    float* __restrict__ m,
    float* __restrict__ n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= chunkN) return;

  const double PI = 3.141592653589793238462643383279502884;

  unsigned long long ipix = (unsigned long long)base_ipix + (unsigned int)tid;
  unsigned long long ipix1 = ipix + 1ull;  // HEALPix RING formula uses 1-based index

  unsigned long long ns = (unsigned long long)nside;
  unsigned long long nl2 = 2ull * ns;
  unsigned long long nl4 = 4ull * ns;
  unsigned long long ncap = 2ull * ns * (ns - 1ull);
  unsigned long long npix = 12ull * ns * ns;

  double z;
  double phi;

  if (ipix1 <= ncap) {
    // North polar cap
    int iring = (int)(0.5 * (1.0 + sqrt(1.0 + 2.0 * (double)ipix1)));
    unsigned long long iphi = ipix1 - 2ull * (unsigned long long)iring * (unsigned long long)(iring - 1);

    z = 1.0 - ((double)iring * (double)iring) / (3.0 * (double)ns * (double)ns);
    phi = ((double)iphi - 0.5) * PI / (2.0 * (double)iring);
  }
  else if (ipix1 <= nl2 * (5ull * ns + 1ull)) {
    // Equatorial region
    unsigned long long ip = ipix1 - ncap - 1ull;
    int iring = (int)(ip / nl4) + nside;
    unsigned long long iphi = (ip % nl4) + 1ull;

    double fodd = 0.5 * (1.0 + (double)((iring + nside) & 1));

    z = ((double)(2 * nside - iring)) * 2.0 / (3.0 * (double)nside);
    phi = ((double)iphi - fodd) * PI / (2.0 * (double)nside);
  }
  else {
    // South polar cap
    unsigned long long ip = npix - ipix1 + 1ull;
    int iring = (int)(0.5 * (1.0 + sqrt(1.0 + 2.0 * (double)ip)));
    unsigned long long iphi =
        4ull * (unsigned long long)iring + 1ull
        - (ip - 2ull * (unsigned long long)iring * (unsigned long long)(iring - 1));

    z = -1.0 + ((double)iring * (double)iring) / (3.0 * (double)ns * (double)ns);
    phi = ((double)iphi - 0.5) * PI / (2.0 * (double)iring);
  }

  double sintheta = sqrt(fmax(0.0, 1.0 - z * z));

  l[tid] = (float)(sintheta * cos(phi));
  m[tid] = (float)(sintheta * sin(phi));
  n[tid] = (float)z;
}

std::vector<torch::Tensor> pix2lmn_ring_forward(long long nside_ll, long long base_ipix_ll, long long chunkN_ll) {
  TORCH_CHECK(nside_ll > 0 && chunkN_ll > 0, "nside and chunkN must be positive");
  TORCH_CHECK(chunkN_ll <= 2147483647LL, "chunkN too large for this test wrapper");
  TORCH_CHECK(base_ipix_ll >= 0 && base_ipix_ll <= 4294967295LL, "base_ipix must fit uint32 for this wrapper");

  unsigned long long npix = 12ull * (unsigned long long)nside_ll * (unsigned long long)nside_ll;
  TORCH_CHECK((unsigned long long)base_ipix_ll + (unsigned long long)chunkN_ll <= npix,
              "requested pixel range exceeds 12*nside*nside");

  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto l = torch::empty({chunkN_ll}, opts);
  auto m = torch::empty({chunkN_ll}, opts);
  auto n = torch::empty({chunkN_ll}, opts);
  int block = 256;
  int grid = (int)((chunkN_ll + block - 1) / block);
  pix2lmn_ring_kernel<<<grid, block>>>((int)nside_ll, (unsigned int)base_ipix_ll, (int)chunkN_ll,
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
            name="rkb_ws_pix2lmn_ring_user_baseline",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["pix2lmn_ring_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT


class Model(nn.Module):
    def forward(self, nside: int, base_ipix: int, chunkN: int):
        return get_extension().pix2lmn_ring_forward(int(nside), int(base_ipix), int(chunkN))


class ModelNew(Model):
    pass


def get_inputs(scale: str = "smoke"):
    if scale == "nside512_full":
        return [512, 0, 512 * 512 * 12]
    if scale == "nside4096_full":
        # Full-scale direction arrays require roughly 3 * npix * 4 bytes.
        return [4096, 0, 4096 * 4096 * 12]
    return [64, 0, 4096]


def get_init_inputs():
    return []
