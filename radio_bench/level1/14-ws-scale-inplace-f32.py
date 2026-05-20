from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.configs.scales import get_scale

TASK_ID = "level1/14-ws-scale-inplace-f32"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full", "nside16384_full"]
CPP_SRC = """#include <torch/extension.h>
torch::Tensor helper_forward(torch::Tensor a);
"""
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <stdint.h>
__global__ void kern(float* dst,long long n,float inv){ long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]*=inv; }
torch::Tensor helper_forward(torch::Tensor a){ TORCH_CHECK(a.is_cuda(),"input must be CUDA"); a=a.contiguous(); auto out=a.clone(); long long n=a.numel(); int block=256; int grid=(int)((n+block-1)/block); kern<<<grid,block>>>(out.data_ptr<float>(), n, 0.125f); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out; }
"""
_EXT=None
def get_extension():
    global _EXT
    if _EXT is None: _EXT=load_inline(name="rkb_level1_01_ws_scale_inplace_f32", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["helper_forward"], extra_cuda_cflags=["-O3","--use_fast_math","-lineinfo"], verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD","0"))))
    return _EXT
class Model(nn.Module):
    def forward(self,*args): return get_extension().helper_forward(*args)
class ModelNew(Model): pass
def get_inputs(scale: str="smoke", segment_profile: str="all10", fixture: str|None=None):
    cfg=get_scale(scale); n=cfg.npix if scale=="smoke" else min(cfg.npix, 4_194_304)
    return [torch.randn(n,device="cuda",dtype=torch.float32)]
def get_init_inputs(): return []
