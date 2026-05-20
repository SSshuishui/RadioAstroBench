"""3D Level 3: day-level all-10-segment visibility halfsym tile-cone bench.

Self-contained; does not import level2 task files.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from typing import List

from radio_astronomy_cuda_bench.common import make_unit_vectors_cuda, make_endpoint_vectors, make_tile_meta_torch
from radio_astronomy_cuda_bench.configs.scales import get_scale
from radio_astronomy_cuda_bench.configs.day1_10m_segments import COSPHI_10M, get_segments, segment_to_dict

TASK_ID = "level3/03-3d-day-visibility-forward"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full"]
TILE_PIX = 512

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor visibility_3d_halfsym_tilecone_forward(
    torch::Tensor B, torch::Tensor l, torch::Tensor m, torch::Tensor n, torch::Tensor nm1,
    torch::Tensor tile_cx, torch::Tensor tile_cy, torch::Tensor tile_cz,
    torch::Tensor tile_cosA, torch::Tensor tile_sinA,
    torch::Tensor u, torch::Tensor v, torch::Tensor w,
    torch::Tensor x1, torch::Tensor y1, torch::Tensor z1,
    torch::Tensor x2, torch::Tensor y2, torch::Tensor z2,
    double cosphi);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
__device__ __forceinline__ void sincos_fast(float x, float* s, float* c) { __sincosf(x, s, c); }
static constexpr int SIGNED_BASELINES_PER_T = 56;
static constexpr int UNIQUE_BASELINES_PER_T = 28;

__device__ __forceinline__ void classify_tile_cone(float dotc, float cosA, float sinA, float cosphi, bool &all_vis, bool &all_hid){
  dotc = fminf(1.0f, fmaxf(-1.0f, dotc));
  float sind = sqrtf(fmaxf(0.0f, 1.0f - dotc*dotc));
  float lower = dotc * cosA - sind * sinA;
  float upper = dotc * cosA + sind * sinA;
  all_vis = (lower >= cosphi);
  all_hid = (upper <  cosphi);
}
__device__ __forceinline__ void operator_classify_pair_tile_dirs(
    float dx1, float dy1, float dz1, float dx2, float dy2, float dz2,
    float cx, float cy, float cz, float cosA, float sinA, float cosphi,
    bool &all_visible, bool &all_hidden)
{
  bool vis1=false, hid1=false, vis2=false, hid2=false;
  classify_tile_cone(dx1*cx + dy1*cy + dz1*cz, cosA, sinA, cosphi, vis1, hid1);
  classify_tile_cone(dx2*cx + dy2*cy + dz2*cz, cosA, sinA, cosphi, vis2, hid2);
  all_hidden  = hid1 || hid2;
  all_visible = vis1 && vis2;
}

template<int TILE_PIX, bool DO_BLOCKAGE>
__global__ void viss_partial_all_halfsym_tilecone_3d(
    const float* __restrict__ B,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const float* __restrict__ nm1,
    long long n_chunk,
    const float* __restrict__ tile_cx,
    const float* __restrict__ tile_cy,
    const float* __restrict__ tile_cz,
    const float* __restrict__ tile_cosA,
    const float* __restrict__ tile_sinA,
    int ntile,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ x1,
    const float* __restrict__ y1,
    const float* __restrict__ z1,
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    int N_half,
    float cosphi,
    float2* __restrict__ Vpart)
{
  int ih = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = (ih < N_half);
  float u0=0.0f, v0=0.0f, w0=0.0f, dx1=0.0f, dy1=0.0f, dz1=0.0f, dx2=0.0f, dy2=0.0f, dz2=0.0f;
  if(active){
    u0 = u[ih]; v0 = v[ih]; w0 = w[ih];
    if constexpr (DO_BLOCKAGE){ dx1=x1[ih]; dy1=y1[ih]; dz1=z1[ih]; dx2=x2[ih]; dy2=y2[ih]; dz2=z2[ih]; }
  }
  float acc_re=0.0f, acc_im=0.0f;
  const float k = -2.0f * (float)M_PI;
  float ku=active ? k*u0 : 0.0f, kv=active ? k*v0 : 0.0f, kw=active ? k*w0 : 0.0f;
  extern __shared__ float smem[];
  float* sB = smem;
  float* sL = sB + TILE_PIX;
  float* sM = sL + TILE_PIX;
  float* sNM1 = sM + TILE_PIX;
  float* sN = sNM1 + TILE_PIX;
  for(int tid=0; tid<ntile; ++tid){
    long long p0 = (long long)tid * TILE_PIX;
    int tileN = (int)min((long long)TILE_PIX, n_chunk - p0);
    for(int lane=threadIdx.x; lane<tileN; lane+=blockDim.x){
      long long p=p0+lane;
      sB[lane]=B[p]; sL[lane]=l[p]; sM[lane]=m[p]; sNM1[lane]=nm1[p]; sN[lane]=n[p];
    }
    __syncthreads();
    bool all_visible=false, all_hidden=false;
    if(active && DO_BLOCKAGE){
      operator_classify_pair_tile_dirs(dx1,dy1,dz1,dx2,dy2,dz2,
        tile_cx[tid],tile_cy[tid],tile_cz[tid],tile_cosA[tid],tile_sinA[tid],cosphi,all_visible,all_hidden);
    }
    if(active && !(DO_BLOCKAGE && all_hidden)){
      if(!DO_BLOCKAGE || all_visible){
        #pragma unroll 4
        for(int k0=0;k0<tileN;++k0){
          float ang=fmaf(kw,sNM1[k0],fmaf(kv,sM[k0],ku*sL[k0]));
          float s,c; sincos_fast(ang,&s,&c); float bp=sB[k0]; acc_re += bp*c; acc_im += bp*s;
        }
      }else{
        #pragma unroll 4
        for(int k0=0;k0<tileN;++k0){
          float lp=sL[k0], mp=sM[k0], npv=sN[k0];
          if(lp*dx1 + mp*dy1 + npv*dz1 < cosphi) continue;
          if(lp*dx2 + mp*dy2 + npv*dz2 < cosphi) continue;
          float ang=fmaf(kw,sNM1[k0],fmaf(kv,mp,ku*lp));
          float s,c; sincos_fast(ang,&s,&c); float bp=sB[k0]; acc_re += bp*c; acc_im += bp*s;
        }
      }
    }
    __syncthreads();
  }
  if(active) Vpart[ih] = make_float2(acc_re, acc_im);
}

torch::Tensor visibility_3d_halfsym_tilecone_forward(
    torch::Tensor B, torch::Tensor l, torch::Tensor m, torch::Tensor n, torch::Tensor nm1,
    torch::Tensor tile_cx, torch::Tensor tile_cy, torch::Tensor tile_cz,
    torch::Tensor tile_cosA, torch::Tensor tile_sinA,
    torch::Tensor u, torch::Tensor v, torch::Tensor w,
    torch::Tensor x1, torch::Tensor y1, torch::Tensor z1,
    torch::Tensor x2, torch::Tensor y2, torch::Tensor z2,
    double cosphi_d) {
  TORCH_CHECK(B.is_cuda(), "inputs must be CUDA tensors");
  B=B.contiguous(); l=l.contiguous(); m=m.contiguous(); n=n.contiguous(); nm1=nm1.contiguous();
  tile_cx=tile_cx.contiguous(); tile_cy=tile_cy.contiguous(); tile_cz=tile_cz.contiguous(); tile_cosA=tile_cosA.contiguous(); tile_sinA=tile_sinA.contiguous();
  u=u.contiguous(); v=v.contiguous(); w=w.contiguous(); x1=x1.contiguous(); y1=y1.contiguous(); z1=z1.contiguous(); x2=x2.contiguous(); y2=y2.contiguous(); z2=z2.contiguous();
  long long n_chunk=B.numel(); int ntile=(int)tile_cx.numel(); int N_half=(int)u.numel();
  auto out=torch::empty({N_half,2}, B.options());
  int block=128; int grid=(N_half+block-1)/block; size_t smem=5*512*sizeof(float);
  viss_partial_all_halfsym_tilecone_3d<512,true><<<grid,block,smem>>>(B.data_ptr<float>(),l.data_ptr<float>(),m.data_ptr<float>(),n.data_ptr<float>(),nm1.data_ptr<float>(),n_chunk,
      tile_cx.data_ptr<float>(),tile_cy.data_ptr<float>(),tile_cz.data_ptr<float>(),tile_cosA.data_ptr<float>(),tile_sinA.data_ptr<float>(),ntile,
      u.data_ptr<float>(),v.data_ptr<float>(),w.data_ptr<float>(),x1.data_ptr<float>(),y1.data_ptr<float>(),z1.data_ptr<float>(),x2.data_ptr<float>(),y2.data_ptr<float>(),z2.data_ptr<float>(),N_half,(float)cosphi_d,reinterpret_cast<float2*>(out.data_ptr<float>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""


_EXT = None

def get_extension():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="rkb_3d_day_visibility_halfsym_tilecone",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["visibility_3d_halfsym_tilecone_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT


def _make_B(n: int, device: str = "cuda"):
    idx=torch.arange(n,device=device,dtype=torch.float32)
    return (1.0 + 0.08*torch.sin(idx*0.00017) + 0.02*torch.cos(idx*0.0011)).contiguous()


def _segment_baseline_tensors(seg, device: str = "cuda"):
    N_half = int(seg.segN_half)
    u=torch.linspace(float(seg.u_min or -100.0),float(seg.u_max or 100.0),N_half,device=device,dtype=torch.float32)
    v=torch.linspace(float(seg.v_min or -100.0),float(seg.v_max or 100.0),N_half,device=device,dtype=torch.float32)
    w=torch.linspace(float(seg.w_min or -50.0),float(seg.w_max or 50.0),N_half,device=device,dtype=torch.float32)
    x1,y1,z1,_=make_endpoint_vectors(N_half,device=device,seed=610+seg.seg_id)
    x2,y2,z2,_=make_endpoint_vectors(N_half,device=device,seed=710+seg.seg_id)
    return u.contiguous(),v.contiguous(),w.contiguous(),x1,y1,z1,x2,y2,z2

class Model(nn.Module):
    def forward(self,B,l,m,n,nm1,tile_cx,tile_cy,tile_cz,tile_cosA,tile_sinA,u_list,v_list,w_list,x1_list,y1_list,z1_list,x2_list,y2_list,z2_list,cosphi: float):
        ext=get_extension(); outs=[]
        for i in range(len(u_list)):
            outs.append(ext.visibility_3d_halfsym_tilecone_forward(B,l,m,n,nm1,tile_cx,tile_cy,tile_cz,tile_cosA,tile_sinA,u_list[i],v_list[i],w_list[i],x1_list[i],y1_list[i],z1_list[i],x2_list[i],y2_list[i],z2_list[i],float(cosphi)))
        return torch.cat(outs, dim=0)

class ModelNew(Model): pass


def get_inputs(scale: str = "smoke", segment_profile: str = "all10", fixture: str | None = None):
    cfg=get_scale(scale)
    if not cfg.run_visibility:
        raise ValueError(f"Scale {scale} disables visibility tasks")
    device="cuda"
    n_pix=(cfg.npix//TILE_PIX)*TILE_PIX
    l,m,n=make_unit_vectors_cuda(n_pix,device=device,seed=607); nm1=(n-1.0).contiguous()
    B=_make_B(n_pix,device=device); tile_meta=make_tile_meta_torch(l,m,n,TILE_PIX)
    segs=get_segments(scale,segment_profile)
    lists=[[] for _ in range(9)]
    for seg in segs:
        vals=_segment_baseline_tensors(seg,device=device)
        for dst,val in zip(lists,vals): dst.append(val)
    return [B,l,m,n,nm1,*tile_meta,*lists,float(COSPHI_10M)]


def describe_inputs(scale: str = "smoke", segment_profile: str = "all10", fixture: str | None = None):
    cfg=get_scale(scale); segs=get_segments(scale,segment_profile)
    return {"mode":"3d day-level all-segment visibility", "scale":scale, "nside":cfg.nside, "npix":cfg.npix, "tile_pix":TILE_PIX, "num_segments":len(segs), "segments":[segment_to_dict(s) for s in segs]}


def get_init_inputs(): return []
