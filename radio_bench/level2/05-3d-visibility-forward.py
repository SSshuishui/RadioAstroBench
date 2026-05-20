"""3D Level 2: visibility half-symmetry tile-cone kernel.

This is the 3D algorithm's visibility-stage kernel form.  Compared with the WS
wrapper, endpoint vectors are passed as normalized directions (x/y/z only),
matching the 3D ``viss_recon_kernel.cuh`` variant.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

from radio_astronomy_cuda_bench.common import make_unit_vectors, make_endpoint_vectors, make_tile_meta_torch

TASK_ID = "level2/05-3d-visibility-forward"
SUPPORTED_SCALES = ["smoke", "nside512_full"]
TILE_PIX = 512
UNIQUE_BASELINES_PER_T = 28
SIGNED_BASELINES_PER_T = 56

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
            name="rkb_3d_visibility_halfsym_tilecone",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["visibility_3d_halfsym_tilecone_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD", "0"))),
        )
    return _EXT

class Model(nn.Module):
    def forward(self, B,l,m,n,nm1,tile_cx,tile_cy,tile_cz,tile_cosA,tile_sinA,u,v,w,x1,y1,z1,x2,y2,z2,cosphi: float):
        return get_extension().visibility_3d_halfsym_tilecone_forward(B,l,m,n,nm1,tile_cx,tile_cy,tile_cz,tile_cosA,tile_sinA,u,v,w,x1,y1,z1,x2,y2,z2,float(cosphi))

class ModelNew(Model):
    pass

def _n_pix(scale: str) -> int:
    return 8192 if scale == "smoke" else 512 * 512 * 12

def get_inputs(scale: str = "smoke", **_):
    device="cuda"; n_chunk=(_n_pix(scale)//TILE_PIX)*TILE_PIX
    l,m,n=make_unit_vectors(n_chunk,device=device,seed=501); nm1=(n-1.0).contiguous()
    B=(1.0+0.05*torch.sin(torch.arange(n_chunk,device=device,dtype=torch.float32)*0.001)).contiguous()
    tile_meta=make_tile_meta_torch(l,m,n,TILE_PIX)
    N_half=112 if scale=="smoke" else 4096
    u=torch.linspace(-500,500,N_half,device=device,dtype=torch.float32); v=torch.sin(torch.linspace(0,5,N_half,device=device))*300; w=torch.cos(torch.linspace(0,3,N_half,device=device))*200
    x1,y1,z1,_=make_endpoint_vectors(N_half,device=device,seed=502); x2,y2,z2,_=make_endpoint_vectors(N_half,device=device,seed=503)
    cosphi=0.15
    return [B,l,m,n,nm1,*tile_meta,u,v,w,x1,y1,z1,x2,y2,z2,cosphi]

def get_init_inputs(): return []
