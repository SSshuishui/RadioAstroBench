from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.configs.scales import get_scale

TASK_ID = "level1/08-ws-iota"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full", "nside16384_full"]
CPP_SRC = """#include <torch/extension.h>\ntorch::Tensor iota_forward(long long n);\n"""
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
__global__ void iota_kernel(int* idx, long long n){ long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) idx[i]=(int)i; }
torch::Tensor iota_forward(long long n){ auto out=torch::empty({n}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32)); int block=256; int grid=(int)((n+block-1)/block); iota_kernel<<<grid,block>>>(out.data_ptr<int>(), n); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out; }
"""
_EXT=None
def get_extension():
    global _EXT
    if _EXT is None:
        _EXT=load_inline(name="rkb_01_ws_iota", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["iota_forward"], extra_cuda_cflags=["-O3","--use_fast_math","-lineinfo"], verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD","0"))))
    return _EXT
class Model(nn.Module):
    def forward(self, dummy: torch.Tensor): return get_extension().iota_forward(dummy.numel())
class ModelNew(Model): pass
def get_inputs(scale: str="smoke", segment_profile: str="all10", fixture: str|None=None):
    cfg=get_scale(scale); n=min(cfg.npix, 1_048_576) if scale!="smoke" else cfg.npix
    return [torch.empty((n,), device="cuda", dtype=torch.float32)]
def get_init_inputs(): return []
