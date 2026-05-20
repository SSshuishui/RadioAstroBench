from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from radio_astronomy_cuda_bench.configs.scales import get_scale

TASK_ID = "level1/11-ws-compute-avg"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full"]
CPP_SRC = """#include <torch/extension.h>\ntorch::Tensor compute_avg_forward(torch::Tensor sum, torch::Tensor counts);\n"""
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
__global__ void compute_avg(float2* avg,const float2* sum,const int* counts,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; int c=counts[i]; float inv=(c>0)?1.0f/(float)c:0.0f; avg[i]=make_float2(sum[i].x*inv,sum[i].y*inv); }
torch::Tensor compute_avg_forward(torch::Tensor sum, torch::Tensor counts){ TORCH_CHECK(sum.is_cuda()&&counts.is_cuda(),"inputs must be CUDA"); sum=sum.contiguous(); counts=counts.contiguous(); int n=(int)counts.numel(); auto out=torch::empty_like(sum); int block=256; int grid=(n+block-1)/block; compute_avg<<<grid,block>>>(reinterpret_cast<float2*>(out.data_ptr<float>()), reinterpret_cast<const float2*>(sum.data_ptr<float>()), counts.data_ptr<int>(), n); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out; }
"""
_EXT=None
def get_extension():
    global _EXT
    if _EXT is None: _EXT=load_inline(name="rkb_01_ws_compute_avg", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["compute_avg_forward"], extra_cuda_cflags=["-O3","--use_fast_math","-lineinfo"], verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD","0"))))
    return _EXT
class Model(nn.Module):
    def forward(self,s,c): return get_extension().compute_avg_forward(s,c)
class ModelNew(Model): pass
def get_inputs(scale: str="smoke", segment_profile: str="all10", fixture: str|None=None):
    n=4096 if scale=="smoke" else 49152
    return [torch.randn(n,2,device="cuda",dtype=torch.float32), torch.randint(1,16,(n,),device="cuda",dtype=torch.int32)]
def get_init_inputs(): return []
