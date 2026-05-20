from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.configs.scales import get_scale

TASK_ID = "level1/13-ws-add-inplace-u32"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full", "nside16384_full"]
CPP_SRC = """#include <torch/extension.h>
torch::Tensor helper_forward(torch::Tensor a, torch::Tensor b);
"""
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>
__global__ void kern(uint32_t* dst,const uint32_t* src,long long n){ long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]+=src[i]; }
torch::Tensor helper_forward(torch::Tensor a, torch::Tensor b){ TORCH_CHECK(a.is_cuda()&&b.is_cuda(),"inputs must be CUDA"); a=a.contiguous(); b=b.contiguous(); auto out=a.clone(); long long n=a.numel(); int block=256; int grid=(int)((n+block-1)/block); kern<<<grid,block>>>(out.data_ptr<uint32_t>(), b.data_ptr<uint32_t>(), n); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out; }
"""
_EXT=None
def get_extension():
    global _EXT
    if _EXT is None: _EXT=load_inline(name="rkb_level1_01_ws_add_inplace_u32", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["helper_forward"], extra_cuda_cflags=["-O3","--use_fast_math","-lineinfo"], verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD","0"))))
    return _EXT
class Model(nn.Module):
    def forward(self,*args): return get_extension().helper_forward(*args)
class ModelNew(Model): pass
def get_inputs(scale: str="smoke", segment_profile: str="all10", fixture: str|None=None):
    cfg=get_scale(scale); n=cfg.npix if scale=="smoke" else min(cfg.npix, 4_194_304)
    a = (torch.arange(n, device="cuda", dtype=torch.int64) % 100).to(torch.uint32)
    b = ((torch.arange(n, device="cuda", dtype=torch.int64) * 3) % 100).to(torch.uint32)
    return [a, b]
def get_init_inputs(): return []
