from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.configs.scales import get_scale

TASK_ID = "level1/10-ws-gather-viss-by-idx"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full"]
CPP_SRC = """#include <torch/extension.h>\ntorch::Tensor gather_viss_forward(torch::Tensor viss, torch::Tensor idx);\n"""
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
__global__ void gather_viss_by_idx(const float2* __restrict__ in,const int* __restrict__ idx,float2* __restrict__ out,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=in[idx[i]]; }
torch::Tensor gather_viss_forward(torch::Tensor viss, torch::Tensor idx){ TORCH_CHECK(viss.is_cuda()&&idx.is_cuda(),"inputs must be CUDA"); viss=viss.contiguous(); idx=idx.contiguous(); int n=(int)idx.numel(); auto out=torch::empty({n,2}, viss.options()); int block=256; int grid=(n+block-1)/block; gather_viss_by_idx<<<grid,block>>>(reinterpret_cast<const float2*>(viss.data_ptr<float>()), idx.data_ptr<int>(), reinterpret_cast<float2*>(out.data_ptr<float>()), n); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out; }
"""
_EXT=None
def get_extension():
    global _EXT
    if _EXT is None: _EXT=load_inline(name="rkb_01_ws_gather_viss", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["gather_viss_forward"], extra_cuda_cflags=["-O3","--use_fast_math","-lineinfo"], verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD","0"))))
    return _EXT
class Model(nn.Module):
    def forward(self,viss,idx): return get_extension().gather_viss_forward(viss,idx)
class ModelNew(Model): pass
def get_inputs(scale: str="smoke", segment_profile: str="all10", fixture: str|None=None):
    n=8192 if scale=="smoke" else 195468
    viss=torch.randn(n,2,device="cuda",dtype=torch.float32); idx=torch.arange(n-1,-1,-1,device="cuda",dtype=torch.int32); return [viss,idx]
def get_init_inputs(): return []
