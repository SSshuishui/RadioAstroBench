from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.configs.scales import get_scale

TASK_ID = "level1/18-3d-invnorm3"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full", "nside16384_full"]
CPP_SRC = """#include <torch/extension.h>
#include <vector>
torch::Tensor invnorm3_forward(torch::Tensor x, torch::Tensor y, torch::Tensor z);
"""
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>
__global__ void kern(const float* x,const float* y,const float* z,float* out,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=rsqrtf(x[i]*x[i]+y[i]*y[i]+z[i]*z[i]+1e-20f); }
torch::Tensor invnorm3_forward(torch::Tensor x, torch::Tensor y, torch::Tensor z){ x=x.contiguous(); y=y.contiguous(); z=z.contiguous(); int n=(int)x.numel(); auto out=torch::empty_like(x); int block=256; int grid=(n+block-1)/block; kern<<<grid,block>>>(x.data_ptr<float>(),y.data_ptr<float>(),z.data_ptr<float>(),out.data_ptr<float>(),n); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out; }
"""
_EXT=None
def get_extension():
    global _EXT
    if _EXT is None: _EXT=load_inline(name="rkb_level1_02_3d_invnorm3", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["invnorm3_forward"], extra_cuda_cflags=["-O3","--use_fast_math","-lineinfo"], verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD","0"))))
    return _EXT
class Model(nn.Module):
    def forward(self,x,y,z): return get_extension().invnorm3_forward(x,y,z)
class ModelNew(Model): pass
def get_inputs(scale: str="smoke", segment_profile: str="all10", fixture: str|None=None):
    n=1024 if scale=="smoke" else 69810
    return [torch.randn(n,device="cuda",dtype=torch.float32), torch.randn(n,device="cuda",dtype=torch.float32), torch.randn(n,device="cuda",dtype=torch.float32)]
def get_init_inputs(): return []
