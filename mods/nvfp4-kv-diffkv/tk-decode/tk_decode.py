# TK-based nvfp4 DiffKV flash-decode: split-K + MTP-fusion + pipelined dequant + boundary masking.
# Drop-in for wmma_decode.try_wmma_decode (same signature). ~2.6x over WMMA on MTP (q_len=3) decode.
# cudagraph-safe: NSPLIT=f(num_seqs) (static per capture), static scratch (NSTRIDE+rows fixed, never realloc).
import os, torch
_M=None; _OK=None
_HK,_HV,_SB,_G = 192,128,180,16
_QLEN_CAP=3
_NSTRIDE=256
_MAX_BATCH=max(1,int(os.environ.get("VLLM_TK_MAX_BATCH","64")))
def _nsplit(num_seqs): return max(32, min(96, 256//max(1,num_seqs)))   # 96 = 1-wave (192 blk @ 4/SM x 48 SM); NS256 wasted ~2.7 waves

_CUDA_SRC=r'''
#include "kittens.cuh"
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAStream.h>
using namespace kittens;
__device__ __forceinline__ float e2m1(unsigned int n){
  const float L[16]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f}; return L[n&15]; }
__device__ __forceinline__ float fp8d(unsigned char b){ __nv_fp8_e4m3 v; *reinterpret_cast<unsigned char*>(&v)=b; return (float)v; }
#define G 16
#define DQK 192
#define DVO 128
#define NB 16
#define SB 180
#define KF 0
#define KS 96
#define VF 108
#define VS 172
#define NSTRIDE 256
using q_gl = gl<bf16,-1,-1,-1,-1>;
using pa_gl = gl<float,-1,-1,-1,-1>;
struct cglobals { q_gl q; pa_gl pa; const unsigned char* cache; const int* bt; const int* seqused;
  float* pm; float* pl; int NKVH,NQH,maxblk,BS,BSL,NSPLIT,QLEN,Ralloc; float scale; };
__device__ __forceinline__ void deq(st_bf<NB,DQK>& Ks, st_bf<NB,DVO>& Vs,
    const unsigned char* cache, const int* bt_s, int jt, int L, int kh, int NKVH, int BS, int BSL, int tid, int nthreads){
  for(int i=tid;i<NB*DQK;i+=nthreads){ int r=i/DQK,d=i%DQK; int p=jt+r;
    if(p<L){ int phys=bt_s[p>>BSL]; const unsigned char* rb=cache+((size_t)((size_t)phys*BS+(p&(BS-1)))*NKVH+kh)*SB;
      unsigned char by=rb[KF+(d>>1)]; unsigned int nib=(d&1)?(by>>4):(by&15);
      Ks[{r,d}]=__float2bfloat16(e2m1(nib)*fp8d(rb[KS+(d>>4)])); } else Ks[{r,d}]=__float2bfloat16(0.f); }
  for(int i=tid;i<NB*DVO;i+=nthreads){ int r=i/DVO,d=i%DVO; int p=jt+r;
    if(p<L){ int phys=bt_s[p>>BSL]; const unsigned char* rb=cache+((size_t)((size_t)phys*BS+(p&(BS-1)))*NKVH+kh)*SB;
      unsigned char by=rb[VF+(d>>1)]; unsigned int nib=(d&1)?(by>>4):(by&15);
      Vs[{r,d}]=__float2bfloat16(e2m1(nib)*fp8d(rb[VS+(d>>4)])); } else Vs[{r,d}]=__float2bfloat16(0.f); }
}
__global__ void tk_prod(const __grid_constant__ cglobals g){
  int kh=blockIdx.x, sp=blockIdx.y, seq=blockIdx.z;
  int w=threadIdx.x>>5, lane=threadIdx.x&31, tid=threadIdx.x, nthreads=g.QLEN*32;
  int Lmax=g.seqused[seq]; int su=Lmax-g.QLEN+1+w;
  __shared__ st_bf<NB,DQK> Ksb[2]; __shared__ st_bf<NB,DVO> Vsb[2]; __shared__ sv_fl<G> msh[8], lsh[8];
  int base_row=(seq*g.QLEN+w)*g.NQH + kh*16; int row_tile=base_row/16;
  rt_bf<G,DQK> qf; warp::load(qf,g.q,{0,0,row_tile,0});
  rt_fl<G,DVO> o_reg; warp::zero(o_reg);
  col_vec<rt_fl<G,NB>> max_vec,norm_vec,corr,max_last;
  warp::neg_infty(max_vec); warp::zero(norm_vec); warp::copy(max_last,max_vec);
  const int* bt_s=g.bt+(size_t)seq*g.maxblk;
  int nchunk=(Lmax+NB-1)/NB; int per=(nchunk+g.NSPLIT-1)/g.NSPLIT; int c0=sp*per, c1=min(nchunk,c0+per);
  if(c0<c1) deq(Ksb[0],Vsb[0],g.cache,bt_s,c0*NB,Lmax,kh,g.NKVH,g.BS,g.BSL,tid,nthreads);
  for(int j=c0;j<c1;j++){ int b=(j-c0)&1; int jt=j*NB;
    __syncthreads();
    if(j+1<c1) deq(Ksb[b^1],Vsb[b^1],g.cache,bt_s,(j+1)*NB,Lmax,kh,g.NKVH,g.BS,g.BSL,tid,nthreads);
    if(jt<su){ int nv=su-jt; if(nv>NB)nv=NB;
      rt_bf<NB,DQK> kt; warp::load(kt,Ksb[b]);
      rt_fl<G,NB> s; warp::zero(s); warp::mma_ABt(s,qf,kt,s); warp::mul(s,s,g.scale);
      if(nv<NB) warp::right_fill(s,s,nv,-1e30f);
      warp::row_max(max_vec,s,max_vec); warp::sub_row(s,s,max_vec); warp::exp(s,s);
      warp::sub(corr,max_last,max_vec); warp::exp(corr,corr);
      warp::mul(norm_vec,norm_vec,corr); warp::row_sum(norm_vec,s,norm_vec);
      rt_bf<G,NB> p; warp::copy(p,s);
      rt_bf<NB,DVO,ducks::rt_layout::col> vt; warp::load(vt,Vsb[b]);
      warp::mul_row(o_reg,o_reg,corr); warp::mma_AB(o_reg,p,vt,o_reg);
      warp::copy(max_last,max_vec);
    }
  }
  warp::store(g.pa, o_reg, {sp,0,row_tile,0});
  warp::store(msh[w], max_vec); warp::store(lsh[w], norm_vec); __syncwarp();
  for(int r=lane;r<G;r+=32){ size_t o=(size_t)sp*g.Ralloc+base_row+r; g.pm[o]=msh[w][r]; g.pl[o]=lsh[w][r]; }
}
__global__ void tk_reduce(const float* pm,const float* pl,const float* pa, bf16* o,int NSPLIT,int Ralloc){
  int grow=blockIdx.x, lane=threadIdx.x; const int VD=DVO/32; float m=-1e30f,l=0.f,a[VD];
  #pragma unroll
  for(int i=0;i<VD;i++)a[i]=0.f;
  for(int sp=0;sp<NSPLIT;sp++){ float ms=pm[(size_t)sp*Ralloc+grow]; float mn=fmaxf(m,ms),c1=__expf(m-mn),c2=__expf(ms-mn);
    #pragma unroll
    for(int i=0;i<VD;i++) a[i]=a[i]*c1+pa[((size_t)sp*Ralloc+grow)*DVO+lane*VD+i]*c2;
    l=l*c1+pl[(size_t)sp*Ralloc+grow]*c2; m=mn; }
  float inv=(l>0.f)?1.f/l:0.f;
  #pragma unroll
  for(int i=0;i<VD;i++) o[(size_t)grow*DVO+lane*VD+i]=__float2bfloat16(a[i]*inv);
}
// static scratch (alloc once at [NSTRIDE, Ralloc]; Ralloc pre-grown, never shrinks -> cudagraph-safe)
static torch::Tensor g_pm,g_pl,g_pa; static long g_rows=0; static void* g_dev=nullptr;
static void ensure(long want,const torch::TensorOptions& fo,void* dev){
  if(g_pm.defined() && want<=g_rows && dev==g_dev) return;
  if(want<g_rows) want=g_rows; if(want<1) want=1;
  g_pm=torch::empty({(long)NSTRIDE,want},fo); g_pl=torch::empty({(long)NSTRIDE,want},fo);
  g_pa=torch::empty({(long)NSTRIDE,want,DVO},fo); g_rows=want; g_dev=dev;
}
void run(torch::Tensor q,torch::Tensor cache,torch::Tensor bt,torch::Tensor seqused,torch::Tensor out,
         int NKVH,double scale,int BS,int NSPLIT,int QLEN,int min_rows){
  int num_seqs=seqused.size(0), NQH=q.size(1), maxblk=bt.size(1); int Ract=num_seqs*QLEN*NQH;
  int BSL=(BS==64)?6:(BS==32)?5:(BS==16)?4:0;
  long want=(Ract>min_rows)?Ract:min_rows;
  auto fo=torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
  ensure(want,fo,q.device().has_index()?(void*)(intptr_t)q.device().index():(void*)0);
  long Ralloc=g_rows;
  auto cmsg=[&](){};
  cglobals g{ q_gl{(bf16*)q.data_ptr(),1,1,(int)q.size(0),DQK}, pa_gl{g_pa.data_ptr<float>(),NSTRIDE,1,(int)Ralloc,DVO},
             (const unsigned char*)cache.data_ptr(), bt.data_ptr<int>(), seqused.data_ptr<int>(),
             g_pm.data_ptr<float>(), g_pl.data_ptr<float>(), NKVH,NQH,maxblk,BS,BSL,NSPLIT,QLEN,(int)Ralloc,(float)scale };
  dim3 grid(NKVH,NSPLIT,num_seqs);
  auto st=at::cuda::getCurrentCUDAStream();
  tk_prod<<<grid,QLEN*32,0,st>>>(g);
  tk_reduce<<<Ract,32,0,st>>>(g_pm.data_ptr<float>(),g_pl.data_ptr<float>(),g_pa.data_ptr<float>(),(bf16*)out.data_ptr(),NSPLIT,(int)Ralloc);
}
'''

def _compile():
    global _M,_OK
    if _OK is not None: return _OK
    try:
        from torch.utils.cpp_extension import load_inline
        inc=os.environ.get("TK_INC","/tmp/tkinc")
        _M=load_inline(name="tk_decode_prod",
            cpp_sources="void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int,double,int,int,int,int);",
            cuda_sources=_CUDA_SRC, functions=["run"], verbose=False,
            extra_cuda_cflags=["-O3","-std=c++20","--use_fast_math","--expt-relaxed-constexpr","--expt-extended-lambda",
                               "-gencode","arch=compute_121a,code=sm_121a","-DKITTENS_SM120",
                               "-I"+inc+"/include","-I"+inc+"/prototype"],
            extra_cflags=["-std=c++20"], extra_include_paths=[inc+"/include",inc+"/prototype"])
        _OK=True
    except Exception as e:
        import sys; print(f"[tk_decode] compile FAILED -> fallback: {e}",file=sys.stderr); _OK=False
    return _OK

def try_wmma_decode(q,k_cache,out,seqused_k,block_table,softmax_scale,num_kv_heads,head_size_qk,head_size_v,
                    block_size,sinks,softcap,window_left,cu_seqlens_q,max_seqlen_q,force=False):
    if not force and os.environ.get("VLLM_WMMA_DECODE","1")=="0": return False
    if (head_size_qk!=_HK or head_size_v!=_HV or block_size not in (32,64)
            or k_cache.shape[-1]!=_SB or q.shape[1]!=num_kv_heads*_G): return False
    if sinks is not None or softcap not in (0.0,None) or window_left>=0: return False
    if q.dtype!=torch.bfloat16 or cu_seqlens_q is None: return False
    if max_seqlen_q is None or max_seqlen_q>_QLEN_CAP: return False
    if not _compile(): return False
    dev=q.device; total_q=q.shape[0]; NQH=q.shape[1]
    cu=cu_seqlens_q.to(torch.int64); num_seqs=cu.shape[0]-1; QLEN=int(max_seqlen_q)
    if total_q!=num_seqs*QLEN: return False
    if not torch.cuda.is_current_stream_capturing():
        if int(seqused_k.min().item())<=QLEN: return False
        if not bool(torch.all((cu[1:]-cu[:-1])==QLEN)): return False
    NSPLIT=_nsplit(num_seqs)
    bt=block_table.to(torch.int32).contiguous(); su=seqused_k.to(torch.int32).contiguous()
    cache_u8=k_cache.reshape(k_cache.shape[0],block_size,num_kv_heads,_SB)
    cache_u8=cache_u8.contiguous() if not cache_u8.is_contiguous() else cache_u8
    min_rows=_MAX_BATCH*QLEN*NQH; qf=q.contiguous()
    if out.is_contiguous():
        _M.run(qf,cache_u8,bt,su,out.view(total_q*NQH,_HV),num_kv_heads,float(softmax_scale),int(block_size),int(NSPLIT),int(QLEN),int(min_rows))
    else:
        tmp=torch.empty((total_q*NQH,_HV),dtype=out.dtype,device=dev)
        _M.run(qf,cache_u8,bt,su,tmp,num_kv_heads,float(softmax_scale),int(block_size),int(NSPLIT),int(QLEN),int(min_rows))
        out.copy_(tmp.view(total_q,NQH,_HV))
    return True
