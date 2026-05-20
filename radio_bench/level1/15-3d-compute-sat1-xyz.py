from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.configs.scales import get_scale

TASK_ID = "level1/15-3d-compute-sat1-xyz"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full", "nside16384_full"]
CPP_SRC = """#include <torch/extension.h>\n#include <vector>\nstd::vector<torch::Tensor> compute_sat1_xyz_forward(torch::Tensor t);\n"""
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cmath>
__global__ void k_compute_sat1_xyz(const float* t,float* x,float* y,float* z,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; float a=t[i]*0.001f; float r=2037.0f; x[i]=r*cosf(a); y[i]=r*sinf(a); z[i]=0.5f*r*sinf(0.3f*a); }
std::vector<torch::Tensor> compute_sat1_xyz_forward(torch::Tensor t){ TORCH_CHECK(t.is_cuda(),"t must be CUDA"); t=t.contiguous(); int n=(int)t.numel(); auto x=torch::empty_like(t), y=torch::empty_like(t), z=torch::empty_like(t); int block=256; int grid=(n+block-1)/block; k_compute_sat1_xyz<<<grid,block>>>(t.data_ptr<float>(),x.data_ptr<float>(),y.data_ptr<float>(),z.data_ptr<float>(),n); C10_CUDA_KERNEL_LAUNCH_CHECK(); return {x,y,z}; }
"""
_EXT=None
def get_extension():
    global _EXT
    if _EXT is None: _EXT=load_inline(name="rkb_02_3d_compute_sat1_xyz", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["compute_sat1_xyz_forward"], extra_cuda_cflags=["-O3","--use_fast_math","-lineinfo"], verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD","0"))))
    return _EXT
class Model(nn.Module):
    def forward(self,t): return get_extension().compute_sat1_xyz_forward(t)
class ModelNew(Model): pass
def get_inputs(scale: str="smoke", segment_profile: str="all10", fixture: str|None=None):
    n=1024 if scale=="smoke" else 69810
    return [torch.arange(n,device="cuda",dtype=torch.float32)]
def get_init_inputs(): return []
