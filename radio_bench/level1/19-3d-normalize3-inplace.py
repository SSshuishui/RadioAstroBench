from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.configs.scales import get_scale

TASK_ID = "level1/19-3d-normalize3-inplace"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full", "nside16384_full"]
CPP_SRC = """#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> normalize3_forward(torch::Tensor x, torch::Tensor y, torch::Tensor z);
"""
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>
__global__ void kern(float* x,float* y,float* z,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float inv=rsqrtf(x[i]*x[i]+y[i]*y[i]+z[i]*z[i]+1e-20f); x[i]*=inv; y[i]*=inv; z[i]*=inv; }}
std::vector<torch::Tensor> normalize3_forward(torch::Tensor x, torch::Tensor y, torch::Tensor z){ auto ox=x.contiguous().clone(); auto oy=y.contiguous().clone(); auto oz=z.contiguous().clone(); int n=(int)ox.numel(); int block=256; int grid=(n+block-1)/block; kern<<<grid,block>>>(ox.data_ptr<float>(),oy.data_ptr<float>(),oz.data_ptr<float>(),n); C10_CUDA_KERNEL_LAUNCH_CHECK(); return {ox,oy,oz}; }
"""
_EXT=None
def get_extension():
    global _EXT
    if _EXT is None: _EXT=load_inline(name="rkb_level1_02_3d_normalize3_inplace", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["normalize3_forward"], extra_cuda_cflags=["-O3","--use_fast_math","-lineinfo"], verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD","0"))))
    return _EXT
class Model(nn.Module):
    def forward(self,x,y,z): return get_extension().normalize3_forward(x,y,z)
class ModelNew(Model): pass
def get_inputs(scale: str="smoke", segment_profile: str="all10", fixture: str|None=None):
    n=1024 if scale=="smoke" else 69810
    return [torch.randn(n,device="cuda",dtype=torch.float32), torch.randn(n,device="cuda",dtype=torch.float32), torch.randn(n,device="cuda",dtype=torch.float32)]
def get_init_inputs(): return []
