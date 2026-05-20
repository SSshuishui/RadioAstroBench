"""Level 1 / task 004: online signed-56 baseline generation.

This task extracts the online baseline-generation idea from the original pipeline:
8 interferometric nodes produce 28 unique pairwise baselines per timestamp, and
another 28 sign-symmetric records are emitted for conjugate-completion-aware
visibility processing.

Model = existing CUDA baseline implementation.
ModelNew = identity candidate for smoke testing.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

from radio_astronomy_cuda_bench.configs.scales import get_scale
from radio_astronomy_cuda_bench.configs.day1_10m_segments import LAMDA_10M

TASK_ID = "level1/07-ws-generate-baselines-signed56"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full", "nside16384_full"]

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> gen_baselines_signed56_forward(torch::Tensor pos, double lamda);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>

__device__ __forceinline__ float invnorm3(float x, float y, float z){
    return rsqrtf(x*x + y*y + z*z + 1e-20f);
}

__global__ void gen_baselines_signed56_kernel(
    const float* __restrict__ pos,  // [T,8,3]
    long long T,
    float lamda,
    float* __restrict__ uvw,        // [T,56,3]
    float4* __restrict__ p1,        // [T,56]
    float4* __restrict__ p2)        // [T,56]
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = T * 28LL;
    if(tid >= total) return;

    long long t = tid / 28LL;
    int pair = (int)(tid - t * 28LL);

    int a = 0, b = 1, c = 0;
    // map pair index to i<j for 8 nodes.
    #pragma unroll
    for(int i=0;i<8;i++){
        #pragma unroll
        for(int j=i+1;j<8;j++){
            if(c == pair){ a=i; b=j; }
            c++;
        }
    }

    const float* pa = pos + (t*8LL + a)*3LL;
    const float* pb = pos + (t*8LL + b)*3LL;
    float ax=pa[0], ay=pa[1], az=pa[2];
    float bx=pb[0], by=pb[1], bz=pb[2];
    float dx = (bx - ax) / lamda;
    float dy = (by - ay) / lamda;
    float dz = (bz - az) / lamda;

    float ia = invnorm3(ax, ay, az);
    float ib = invnorm3(bx, by, bz);
    float4 fa = make_float4(ax, ay, az, ia);
    float4 fb = make_float4(bx, by, bz, ib);

    long long base0 = (t*56LL + pair) * 3LL;
    uvw[base0 + 0] = dx;
    uvw[base0 + 1] = dy;
    uvw[base0 + 2] = dz;
    p1[t*56LL + pair] = fa;
    p2[t*56LL + pair] = fb;

    int mpair = pair + 28;
    long long base1 = (t*56LL + mpair) * 3LL;
    uvw[base1 + 0] = -dx;
    uvw[base1 + 1] = -dy;
    uvw[base1 + 2] = -dz;
    // For the negative baseline, swap endpoints to preserve baseline direction.
    p1[t*56LL + mpair] = fb;
    p2[t*56LL + mpair] = fa;
}

std::vector<torch::Tensor> gen_baselines_signed56_forward(torch::Tensor pos, double lamda_d){
    TORCH_CHECK(pos.is_cuda(), "pos must be CUDA tensor");
    TORCH_CHECK(pos.dim() == 3 && pos.size(1) == 8 && pos.size(2) == 3, "pos shape must be [T,8,3]");
    pos = pos.contiguous();
    long long T = pos.size(0);
    auto uvw = torch::empty({T, 56, 3}, pos.options());
    auto p1 = torch::empty({T, 56, 4}, pos.options());
    auto p2 = torch::empty({T, 56, 4}, pos.options());
    int block = 256;
    int grid = (int)((T * 28LL + block - 1) / block);
    gen_baselines_signed56_kernel<<<grid, block>>>(
        pos.data_ptr<float>(), T, (float)lamda_d, uvw.data_ptr<float>(),
        reinterpret_cast<float4*>(p1.data_ptr<float>()), reinterpret_cast<float4*>(p2.data_ptr<float>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {uvw, p1, p2};
}
"""

_EXT = None

def get_extension():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="rkb_task104_generate_baselines_signed56_baseline",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["gen_baselines_signed56_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT


class Model(nn.Module):
    def forward(self, pos, lamda: float):
        return get_extension().gen_baselines_signed56_forward(pos, float(lamda))


class ModelNew(Model):
    pass


def _make_positions(T: int, device: str = "cuda") -> torch.Tensor:
    # Synthetic but orbit-like: 8 nodes on a rotating/telescoping ring.
    idx_t = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    idx_n = torch.arange(8, device=device, dtype=torch.float32)[None, :]
    angle = 0.017 * idx_t + 2.0 * 3.141592653589793 * idx_n / 8.0
    radius = 1.7e6 + 2.0e5 * torch.sin(0.003 * idx_t + idx_n)
    z = 3.0e5 * torch.sin(angle * 0.7)
    x = radius * torch.cos(angle)
    y = radius * torch.sin(angle)
    return torch.stack([x, y, z], dim=2).contiguous()


def get_inputs(scale: str = "smoke", segment_profile: str = "all10", fixture: str | None = None):
    cfg = get_scale(scale)
    if scale == "smoke":
        T = 128
    else:
        # Match the original day-1 orbit plan: T=69810 timestamps for one day.
        # This is still small compared with full-sky visibility/reconstruction.
        T = int(os.environ.get("RKB_BASELINE_T", "69810"))
    pos = _make_positions(T, device="cuda")
    return [pos, float(LAMDA_10M)]


def get_init_inputs():
    return []
