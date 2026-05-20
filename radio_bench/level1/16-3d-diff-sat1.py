from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.configs.scales import get_scale

TASK_ID = "level1/16-3d-diff-sat1"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full", "nside16384_full"]
CPP_SRC = """#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> diff_sat1_forward(torch::Tensor x, torch::Tensor y, torch::Tensor z);
"""
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>
__global__ void kern(const float* x,const float* y,const float* z,float* dx,float* dy,float* dz,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; int j=(i+1<n)?i+1:i; dx[i]=x[j]-x[i]; dy[i]=y[j]-y[i]; dz[i]=z[j]-z[i]; }
std::vector<torch::Tensor> diff_sat1_forward(torch::Tensor x, torch::Tensor y, torch::Tensor z){ x=x.contiguous(); y=y.contiguous(); z=z.contiguous(); int n=(int)x.numel(); auto dx=torch::empty_like(x), dy=torch::empty_like(x), dz=torch::empty_like(x); int block=256; int grid=(n+block-1)/block; kern<<<grid,block>>>(x.data_ptr<float>(),y.data_ptr<float>(),z.data_ptr<float>(),dx.data_ptr<float>(),dy.data_ptr<float>(),dz.data_ptr<float>(),n); C10_CUDA_KERNEL_LAUNCH_CHECK(); return {dx,dy,dz}; }
"""
_EXT=None
def get_extension():
    global _EXT
    if _EXT is None: _EXT=load_inline(name="rkb_level1_02_3d_diff_sat1", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["diff_sat1_forward"], extra_cuda_cflags=["-O3","--use_fast_math","-lineinfo"], verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD","0"))))
    return _EXT
class Model(nn.Module):
    def forward(self,x,y,z): return get_extension().diff_sat1_forward(x,y,z)
class ModelNew(Model): pass
def get_inputs(scale: str="smoke", segment_profile: str="all10", fixture: str|None=None):
    n=1024 if scale=="smoke" else 69810
    return [torch.randn(n,device="cuda",dtype=torch.float32), torch.randn(n,device="cuda",dtype=torch.float32), torch.randn(n,device="cuda",dtype=torch.float32)]
def get_init_inputs(): return []
