from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.configs.scales import get_scale

TASK_ID = "level1/09-ws-build-locg-half"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full"]
CPP_SRC = """#include <torch/extension.h>\ntorch::Tensor build_locg_half_forward(torch::Tensor u, torch::Tensor v, double umin, double vmin, double du, int RES);\n"""
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
__global__ void build_locg_half_kernel(const float* __restrict__ u,const float* __restrict__ v,int* __restrict__ keys,int n,float umin,float vmin,float du,int RES){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; int iu=(int)floorf((u[i]-umin)/du+0.5f); int iv=(int)floorf((v[i]-vmin)/du+0.5f); if(iu<0) iu=0; if(iu>=RES) iu=RES-1; if(iv<0) iv=0; if(iv>=RES) iv=RES-1; keys[i]=iu*RES+iv+1; }
torch::Tensor build_locg_half_forward(torch::Tensor u, torch::Tensor v, double umin, double vmin, double du, int RES){ TORCH_CHECK(u.is_cuda()&&v.is_cuda(),"u/v must be CUDA"); u=u.contiguous(); v=v.contiguous(); int n=(int)u.numel(); auto out=torch::empty({n}, u.options().dtype(torch::kInt32)); int block=256; int grid=(n+block-1)/block; build_locg_half_kernel<<<grid,block>>>(u.data_ptr<float>(),v.data_ptr<float>(),out.data_ptr<int>(),n,(float)umin,(float)vmin,(float)du,RES); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out; }
"""
_EXT=None
def get_extension():
    global _EXT
    if _EXT is None: _EXT=load_inline(name="rkb_01_ws_build_locg_half", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["build_locg_half_forward"], extra_cuda_cflags=["-O3","--use_fast_math","-lineinfo"], verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD","0"))))
    return _EXT
class Model(nn.Module):
    def forward(self,u,v): return get_extension().build_locg_half_forward(u,v,-800.0,-800.0,0.25,4096)
class ModelNew(Model): pass
def get_inputs(scale: str="smoke", segment_profile: str="all10", fixture: str|None=None):
    cfg=get_scale(scale); n=8192 if scale=="smoke" else 195468
    t=torch.linspace(-700,700,n,device="cuda",dtype=torch.float32); return [t.contiguous(), torch.flip(t,[0]).contiguous()]
def get_init_inputs(): return []
