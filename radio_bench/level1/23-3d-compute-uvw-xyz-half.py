from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.configs.scales import get_scale

TASK_ID = "level1/23-3d-compute-uvw-xyz-half"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full", "nside16384_full"]
PAIRS = 28
CPP_SRC = """#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> compute_uvw_xyz_forward(torch::Tensor pos, double lamda);
"""
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>
__device__ __forceinline__ int pair_i(int p){ int c=0; for(int i=0;i<8;i++){ for(int j=i+1;j<8;j++){ if(c==p) return i; c++; } } return 0; }
__device__ __forceinline__ int pair_j(int p){ int c=0; for(int i=0;i<8;i++){ for(int j=i+1;j<8;j++){ if(c==p) return j; c++; } } return 1; }
__global__ void k_compute_uvw_xyz(const float* pos,float* u,float* v,float* w,float4* p1,float4* p2,int T,float lamda){ int idx=blockIdx.x*blockDim.x+threadIdx.x; int total=T*28; if(idx>=total) return; int t=idx/28; int pp=idx%28; int sign=1; int p=pp; if(false && pp>=28){ p=pp-28; sign=-1; } int a=pair_i(p), b=pair_j(p); const float* A=pos+(t*8+a)*3; const float* B=pos+(t*8+b)*3; float dx=(B[0]-A[0])/(float)lamda*sign; float dy=(B[1]-A[1])/(float)lamda*sign; float dz=(B[2]-A[2])/(float)lamda*sign; u[idx]=dx; v[idx]=dy; w[idx]=dz; float n1=rsqrtf(A[0]*A[0]+A[1]*A[1]+A[2]*A[2]+1e-20f); float n2=rsqrtf(B[0]*B[0]+B[1]*B[1]+B[2]*B[2]+1e-20f); p1[idx]=make_float4(A[0],A[1],A[2],n1); p2[idx]=make_float4(B[0],B[1],B[2],n2); }
std::vector<torch::Tensor> compute_uvw_xyz_forward(torch::Tensor pos, double lamda){ pos=pos.contiguous(); int T=(int)pos.size(0); auto opt=pos.options(); auto u=torch::empty({T,28},opt); auto v=torch::empty({T,28},opt); auto w=torch::empty({T,28},opt); auto p1=torch::empty({T,28,4},opt); auto p2=torch::empty({T,28,4},opt); int total=T*28; int block=256; int grid=(total+block-1)/block; k_compute_uvw_xyz<<<grid,block>>>(pos.data_ptr<float>(),u.data_ptr<float>(),v.data_ptr<float>(),w.data_ptr<float>(),reinterpret_cast<float4*>(p1.data_ptr<float>()),reinterpret_cast<float4*>(p2.data_ptr<float>()),T,(float)lamda); C10_CUDA_KERNEL_LAUNCH_CHECK(); return {u,v,w,p1,p2}; }
"""
_EXT=None
def get_extension():
    global _EXT
    if _EXT is None: _EXT=load_inline(name="rkb_level1_02_3d_compute_uvw_xyz_half", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["compute_uvw_xyz_forward"], extra_cuda_cflags=["-O3","--use_fast_math","-lineinfo"], verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD","0"))))
    return _EXT
class Model(nn.Module):
    def forward(self,pos): return get_extension().compute_uvw_xyz_forward(pos, 30.0)
class ModelNew(Model): pass
def get_inputs(scale: str="smoke", segment_profile: str="all10", fixture: str|None=None):
    T=128 if scale=="smoke" else 6981
    return [torch.randn(T,8,3,device="cuda",dtype=torch.float32)*2000.0]
def get_init_inputs(): return []
