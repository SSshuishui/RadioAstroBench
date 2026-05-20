from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.configs.scales import get_scale

TASK_ID = "level1/24-3d-healpix-lmn-from-theta-phi"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full", "nside16384_full"]
CPP_SRC = """#include <torch/extension.h>\n#include <vector>\nstd::vector<torch::Tensor> lmn_from_theta_phi_forward(torch::Tensor theta, torch::Tensor phi);\n"""
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>
__global__ void healpix_lmn_from_theta_phi_chunk(const float* theta,const float* phi,float* l,float* m,float* n,int N){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return; float st=sinf(theta[i]); l[i]=st*cosf(phi[i]); m[i]=st*sinf(phi[i]); n[i]=cosf(theta[i]); }
std::vector<torch::Tensor> lmn_from_theta_phi_forward(torch::Tensor theta, torch::Tensor phi){ theta=theta.contiguous(); phi=phi.contiguous(); int N=(int)theta.numel(); auto l=torch::empty_like(theta), m=torch::empty_like(theta), n=torch::empty_like(theta); int block=256; int grid=(N+block-1)/block; healpix_lmn_from_theta_phi_chunk<<<grid,block>>>(theta.data_ptr<float>(),phi.data_ptr<float>(),l.data_ptr<float>(),m.data_ptr<float>(),n.data_ptr<float>(),N); C10_CUDA_KERNEL_LAUNCH_CHECK(); return {l,m,n}; }
"""
_EXT=None
def get_extension():
    global _EXT
    if _EXT is None: _EXT=load_inline(name="rkb_02_3d_lmn_theta_phi", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["lmn_from_theta_phi_forward"], extra_cuda_cflags=["-O3","--use_fast_math","-lineinfo"], verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD","0"))))
    return _EXT
class Model(nn.Module):
    def forward(self,theta,phi): return get_extension().lmn_from_theta_phi_forward(theta,phi)
class ModelNew(Model): pass
def get_inputs(scale: str="smoke", segment_profile: str="all10", fixture: str|None=None):
    cfg=get_scale(scale); N=cfg.npix if scale=="smoke" else min(cfg.npix, 4_194_304)
    theta=torch.linspace(0.01,3.13,N,device="cuda",dtype=torch.float32); phi=torch.linspace(-3.14,3.14,N,device="cuda",dtype=torch.float32); return [theta,phi]
def get_init_inputs(): return []
