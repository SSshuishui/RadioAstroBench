from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.configs.scales import get_scale

TASK_ID = "level1/20-3d-compute-other-sats"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full", "nside16384_full"]
CPP_SRC = """#include <torch/extension.h>\ntorch::Tensor compute_other_sats_forward(torch::Tensor cx, torch::Tensor cy, torch::Tensor cz);\n"""
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
__global__ void k_compute_other_sats(const float* cx,const float* cy,const float* cz,float* out,int T){ int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=T*8) return; int t=tid/8; int s=tid%8; float ang=6.28318530718f*(float)s/8.0f; float r=10.0f+0.01f*(float)t; out[(tid*3)+0]=cx[t]+r*cosf(ang); out[(tid*3)+1]=cy[t]+r*sinf(ang); out[(tid*3)+2]=cz[t]+0.1f*r*sinf(2.0f*ang); }
torch::Tensor compute_other_sats_forward(torch::Tensor cx, torch::Tensor cy, torch::Tensor cz){ cx=cx.contiguous(); cy=cy.contiguous(); cz=cz.contiguous(); int T=(int)cx.numel(); auto out=torch::empty({T,8,3}, cx.options()); int n=T*8; int block=256; int grid=(n+block-1)/block; k_compute_other_sats<<<grid,block>>>(cx.data_ptr<float>(),cy.data_ptr<float>(),cz.data_ptr<float>(),out.data_ptr<float>(),T); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out; }
"""
_EXT=None
def get_extension():
    global _EXT
    if _EXT is None: _EXT=load_inline(name="rkb_02_3d_compute_other_sats", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["compute_other_sats_forward"], extra_cuda_cflags=["-O3","--use_fast_math","-lineinfo"], verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD","0"))))
    return _EXT
class Model(nn.Module):
    def forward(self,cx,cy,cz): return get_extension().compute_other_sats_forward(cx,cy,cz)
class ModelNew(Model): pass
def get_inputs(scale: str="smoke", segment_profile: str="all10", fixture: str|None=None):
    T=1024 if scale=="smoke" else 69810
    a=torch.linspace(0,10,T,device="cuda",dtype=torch.float32); return [torch.cos(a)*2000, torch.sin(a)*2000, torch.sin(a*0.3)*1000]
def get_init_inputs(): return []
