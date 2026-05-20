"""3D Level 3: day-level all-10-segment direct 3D reconstruction bench.

Self-contained; no imports from level2 task files.  This uses the direct
half-symmetry 3D inverse accumulation kernel for each segment, then sums the
segment contributions.
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

TASK_ID = "level3/04-3d-day-recon-direct"
SUPPORTED_SCALES = ["smoke", "nside512_full", "nside4096_full"]
TILE_PIX = 256
TILE_BL = 128

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor recon_3d_direct_halfsym_tilecone_forward(
    torch::Tensor l, torch::Tensor m, torch::Tensor n,
    torch::Tensor tile_cx, torch::Tensor tile_cy, torch::Tensor tile_cz,
    torch::Tensor tile_cosA, torch::Tensor tile_sinA,
    torch::Tensor u, torch::Tensor v, torch::Tensor w,
    torch::Tensor x1, torch::Tensor y1, torch::Tensor z1,
    torch::Tensor x2, torch::Tensor y2, torch::Tensor z2,
    torch::Tensor pair_weight_half, torch::Tensor Viss_half,
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
__device__ __forceinline__ void sincos_fast(float x, float* s, float* c){ __sincosf(x,s,c); }
__device__ __forceinline__ void classify_tile_cone(float dotc, float cosA, float sinA, float cosphi, bool &all_vis, bool &all_hid){
  dotc=fminf(1.0f,fmaxf(-1.0f,dotc)); float sind=sqrtf(fmaxf(0.0f,1.0f-dotc*dotc));
  float lower=dotc*cosA-sind*sinA; float upper=dotc*cosA+sind*sinA;
  all_vis=(lower>=cosphi); all_hid=(upper<cosphi);
}
__device__ __forceinline__ void operator_classify_pair_tile_dirs(float x1,float y1,float z1,float x2,float y2,float z2,float cx,float cy,float cz,float cosA,float sinA,float cosphi,bool &allv,bool &allh){
  bool v1=false,h1=false,v2=false,h2=false;
  classify_tile_cone(x1*cx+y1*cy+z1*cz,cosA,sinA,cosphi,v1,h1);
  classify_tile_cone(x2*cx+y2*cy+z2*cz,cosA,sinA,cosphi,v2,h2);
  allv=v1&&v2; allh=h1||h2;
}
__device__ __forceinline__ bool operator_point_visible_pair_dirs(float x1,float y1,float z1,float x2,float y2,float z2,float l,float m,float n,float cosphi){
  if(l*x1+m*y1+n*z1<cosphi) return false; return (l*x2+m*y2+n*z2>=cosphi);
}
__device__ __forceinline__ float operator_adjoint_phase(float u,float v,float w,float l,float m,float n){ return 2.0f*(float)M_PI*(u*l+v*m+w*n); }

template<int TILE_PIX, int TILE_BL, bool DO_BLOCKAGE>
__global__ void recon_3d_direct_halfsym_tilecone_real(
    long long n_chunk,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
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
    const float* __restrict__ pair_weight_half,
    const float2* __restrict__ Viss_half,
    int N_half,
    float cosphi,
    float* __restrict__ Cacc)
{
  __shared__ float su[TILE_BL], sv[TILE_BL], sw[TILE_BL];
  __shared__ float sx1[TILE_BL], sy1[TILE_BL], sz1[TILE_BL], sx2[TILE_BL], sy2[TILE_BL], sz2[TILE_BL];
  __shared__ float spw[TILE_BL]; __shared__ float2 sV[TILE_BL];
  __shared__ unsigned char sallv[TILE_BL], sallh[TILE_BL];
  int tile=blockIdx.x; if(tile>=ntile) return;
  int lane=threadIdx.x; long long pix=(long long)tile*TILE_PIX+lane; bool active=(pix<n_chunk);
  float lp=0.0f,mp=0.0f,npv=0.0f; if(active){ lp=l[pix]; mp=m[pix]; npv=n[pix]; }
  float tcx=tile_cx[tile], tcy=tile_cy[tile], tcz=tile_cz[tile], tcos=tile_cosA[tile], tsin=tile_sinA[tile];
  float acc=0.0f;
  for(int b0=0;b0<N_half;b0+=TILE_BL){
    int tileN=min(TILE_BL,N_half-b0);
    for(int t=lane;t<tileN;t+=blockDim.x){
      int ih=b0+t; su[t]=u[ih]; sv[t]=v[ih]; sw[t]=w[ih]; sx1[t]=x1[ih]; sy1[t]=y1[ih]; sz1[t]=z1[ih]; sx2[t]=x2[ih]; sy2[t]=y2[ih]; sz2[t]=z2[ih]; spw[t]=pair_weight_half[ih]; sV[t]=Viss_half[ih];
      bool av=false, ah=false; operator_classify_pair_tile_dirs(sx1[t],sy1[t],sz1[t],sx2[t],sy2[t],sz2[t],tcx,tcy,tcz,tcos,tsin,cosphi,av,ah); sallv[t]=av?1:0; sallh[t]=ah?1:0;
    }
    __syncthreads();
    if(active){
      #pragma unroll 2
      for(int t=0;t<tileN;++t){
        float pw=spw[t]; if(pw<=0.0f || sallh[t]) continue;
        if(!sallv[t] && !operator_point_visible_pair_dirs(sx1[t],sy1[t],sz1[t],sx2[t],sy2[t],sz2[t],lp,mp,npv,cosphi)) continue;
        float phase=operator_adjoint_phase(su[t],sv[t],sw[t],lp,mp,npv); float s,c; sincos_fast(phase,&s,&c); float2 z=sV[t]; if(!isfinite(z.x)||!isfinite(z.y)) continue; acc += pw*(z.x*c-z.y*s);
      }
    }
    __syncthreads();
  }
  if(active) Cacc[pix]=acc;
}

torch::Tensor recon_3d_direct_halfsym_tilecone_forward(
    torch::Tensor l, torch::Tensor m, torch::Tensor n,
    torch::Tensor tile_cx, torch::Tensor tile_cy, torch::Tensor tile_cz,
    torch::Tensor tile_cosA, torch::Tensor tile_sinA,
    torch::Tensor u, torch::Tensor v, torch::Tensor w,
    torch::Tensor x1, torch::Tensor y1, torch::Tensor z1,
    torch::Tensor x2, torch::Tensor y2, torch::Tensor z2,
    torch::Tensor pair_weight_half, torch::Tensor Viss_half,
    double cosphi_d) {
  TORCH_CHECK(l.is_cuda() && u.is_cuda(), "inputs must be CUDA tensors");
  l=l.contiguous(); m=m.contiguous(); n=n.contiguous(); tile_cx=tile_cx.contiguous(); tile_cy=tile_cy.contiguous(); tile_cz=tile_cz.contiguous(); tile_cosA=tile_cosA.contiguous(); tile_sinA=tile_sinA.contiguous();
  u=u.contiguous(); v=v.contiguous(); w=w.contiguous(); x1=x1.contiguous(); y1=y1.contiguous(); z1=z1.contiguous(); x2=x2.contiguous(); y2=y2.contiguous(); z2=z2.contiguous(); pair_weight_half=pair_weight_half.contiguous(); Viss_half=Viss_half.contiguous();
  long long n_chunk=l.numel(); int ntile=(int)tile_cx.numel(); int N_half=(int)u.numel(); auto C=torch::empty({n_chunk}, l.options());
  recon_3d_direct_halfsym_tilecone_real<256,128,true><<<ntile,256>>>(n_chunk,l.data_ptr<float>(),m.data_ptr<float>(),n.data_ptr<float>(),tile_cx.data_ptr<float>(),tile_cy.data_ptr<float>(),tile_cz.data_ptr<float>(),tile_cosA.data_ptr<float>(),tile_sinA.data_ptr<float>(),ntile,u.data_ptr<float>(),v.data_ptr<float>(),w.data_ptr<float>(),x1.data_ptr<float>(),y1.data_ptr<float>(),z1.data_ptr<float>(),x2.data_ptr<float>(),y2.data_ptr<float>(),z2.data_ptr<float>(),pair_weight_half.data_ptr<float>(),reinterpret_cast<float2*>(Viss_half.data_ptr<float>()),N_half,(float)cosphi_d,C.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return C;
}
"""


_EXT=None

def get_extension():
    global _EXT
    if _EXT is None:
        _EXT=load_inline(name="rkb_3d_day_recon_direct_halfsym_tilecone_real", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["recon_3d_direct_halfsym_tilecone_forward"], extra_cuda_cflags=["-O3","--use_fast_math","-lineinfo"], verbose=bool(int(os.environ.get("RKB_VERBOSE_BUILD","0"))))
    return _EXT


def _segment_tensors(seg, device="cuda"):
    N_half = int(seg.segN_half)
    u=torch.linspace(float(seg.u_min or -100.0),float(seg.u_max or 100.0),N_half,device=device,dtype=torch.float32)
    v=torch.linspace(float(seg.v_min or -100.0),float(seg.v_max or 100.0),N_half,device=device,dtype=torch.float32)
    w=torch.linspace(float(seg.w_min or -50.0),float(seg.w_max or 50.0),N_half,device=device,dtype=torch.float32)
    x1,y1,z1,_=make_endpoint_vectors(N_half,device=device,seed=900+seg.seg_id)
    x2,y2,z2,_=make_endpoint_vectors(N_half,device=device,seed=950+seg.seg_id)
    # Synthetic pair weights and visibilities; formal runs should replace with fixtures from the original 3D pipeline.
    pair_weight_half=torch.full((N_half,),0.02,device=device,dtype=torch.float32)
    phase=torch.linspace(0,6.2831853,N_half,device=device)+0.17*seg.seg_id
    Viss_half=torch.stack([torch.cos(phase),torch.sin(phase)],dim=1).contiguous().to(torch.float32)
    return u,v,w,x1,y1,z1,x2,y2,z2,pair_weight_half,Viss_half

class Model(nn.Module):
    def forward(self,l,m,n,tile_cx,tile_cy,tile_cz,tile_cosA,tile_sinA,u_list,v_list,w_list,x1_list,y1_list,z1_list,x2_list,y2_list,z2_list,pair_weight_list,Viss_list,cosphi: float):
        ext=get_extension(); C_total=torch.zeros_like(l)
        for i in range(len(u_list)):
            C_total = C_total + ext.recon_3d_direct_halfsym_tilecone_forward(l,m,n,tile_cx,tile_cy,tile_cz,tile_cosA,tile_sinA,u_list[i],v_list[i],w_list[i],x1_list[i],y1_list[i],z1_list[i],x2_list[i],y2_list[i],z2_list[i],pair_weight_list[i],Viss_list[i],float(cosphi))
        return C_total

class ModelNew(Model): pass


def get_inputs(scale: str="smoke", segment_profile: str="all10", fixture: str | None = None):
    cfg=get_scale(scale)
    if not cfg.run_reconstruction:
        raise ValueError(f"Scale {scale} disables reconstruction tasks")
    device="cuda"; n_pix=(cfg.npix//TILE_PIX)*TILE_PIX
    l,m,n=make_unit_vectors_cuda(n_pix,device=device,seed=907); tile_meta=make_tile_meta_torch(l,m,n,TILE_PIX)
    segs=get_segments(scale,segment_profile); lists=[[] for _ in range(11)]
    for seg in segs:
        vals=_segment_tensors(seg,device=device)
        for dst,val in zip(lists,vals): dst.append(val)
    return [l,m,n,*tile_meta,*lists,float(COSPHI_10M)]


def describe_inputs(scale: str="smoke", segment_profile: str="all10", fixture: str | None=None):
    cfg=get_scale(scale); segs=get_segments(scale,segment_profile)
    return {"mode":"3d day-level all-segment direct reconstruction", "scale":scale, "nside":cfg.nside, "npix":cfg.npix, "tile_pix":TILE_PIX, "num_segments":len(segs), "segments":[segment_to_dict(s) for s in segs], "warning":"Synthetic Viss/pair weights; replace with 3D pipeline fixtures for formal experiments."}


def get_init_inputs(): return []
