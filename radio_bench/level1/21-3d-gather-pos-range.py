from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.configs.scales import get_scale

TASK_ID = "level1/21-3d-gather-pos-range"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full", "nside16384_full"]
CPP_SRC = """#include <torch/extension.h>\ntorch::Tensor gather_pos_range_forward(torch::Tensor pos, int start, int length);\n"""
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
__global__ void k_gather_pos_range(const float* pos,float* out,int start,int length){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=length*8*3) return; out[i]=pos[start*8*3+i]; }
torch::Tensor gather_pos_range_forward(torch::Tensor pos, int start, int length){ pos=pos.contiguous(); auto out=torch::empty({length,8,3}, pos.options()); int n=length*8*3; int block=256; int grid=(n+block-1)/block; k_gather_pos_range<<<grid,block>>>(pos.data_ptr<float>(),out.data_ptr<float>(),start,length); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out; }
"""
_EXT=None
def get_extension():
    global _EXT
    if _EXT is None: _EXT=load_inline(name="rkb_02_3d_gather_pos_range", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["gather_pos_range_forward"], extra_cuda_cflags=["-O3","--use_fast_math","-lineinfo"], verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD","0"))))
    return _EXT
class Model(nn.Module):
    def forward(self,pos): return get_extension().gather_pos_range_forward(pos, 3, max(1, pos.shape[0]-6))
class ModelNew(Model): pass
def get_inputs(scale: str="smoke", segment_profile: str="all10", fixture: str|None=None):
    T=128 if scale=="smoke" else 6981
    return [torch.randn(T+8,8,3,device="cuda",dtype=torch.float32)]
def get_init_inputs(): return []
