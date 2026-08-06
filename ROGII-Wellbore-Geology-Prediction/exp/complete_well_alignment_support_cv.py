"""Support-conditioned V4 lattice: visible TVT prefix calibrates each well."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch.nn as nn

import complete_well_alignment_unet_v4center_cv as base

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/train"
base.OUT=ROOT/"exp/results/alignment_unet_support"

def load_well(path,v4_delta,global_idx):
    wid=Path(path).name.split("__")[0]
    h=pd.read_csv(path); tw=pd.read_csv(DATA/f"{wid}__typewell.csv").sort_values("TVT")
    ps=int(h.TVT_input.notna().sum())
    if ps<8 or ps>=len(h): return None
    anchor=float(h.TVT_input.iloc[ps-1])
    # Retain a long labeled tail, so the CNN can infer the well-specific datum
    # and direction before decoding the hidden suffix.
    support_start=max(0,ps-80*base.STRIDE)
    take=np.unique(np.r_[np.arange(support_start,len(h),base.STRIDE),ps-1,len(h)-1])
    gr=h.GR.interpolate(limit_direction="both").fillna(h.GR.median()).to_numpy(float)
    gr5=pd.Series(gr).rolling(11,center=True,min_periods=1).mean().to_numpy()
    gr25=pd.Series(gr).rolling(41,center=True,min_periods=1).mean().to_numpy()
    tg=tw.GR.interpolate(limit_direction="both").fillna(tw.GR.median()).to_numpy(float)
    tt=tw.TVT.to_numpy(float)

    # Legal heel affine calibration: transform typewell GR into this lateral's
    # GR scale using only visible TVT_input/GR pairs.
    vis_tvt=h.TVT_input.iloc[:ps].to_numpy(float)
    ref=np.interp(vis_tvt,tt,tg)
    obs=gr5[:ps]
    ok=np.isfinite(ref)&np.isfinite(obs)
    if ok.sum()>=8 and np.std(ref[ok])>1e-5:
        A=np.c_[ref[ok],np.ones(ok.sum())]
        slope,intercept=np.linalg.lstsq(A,obs[ok],rcond=None)[0]
        slope=float(np.clip(slope,.25,2.5))
        intercept=float(np.clip(intercept,-100,100))
    else: slope,intercept=1.,0.

    base_full=np.zeros(len(h),np.float32)
    known=np.arange(len(h))<ps
    base_full[known]=h.TVT_input.iloc[:ps].to_numpy(float)-anchor
    if len(v4_delta)!=len(h)-ps: raise ValueError(f"{wid}: suffix mismatch")
    base_full[ps:]=v4_delta
    center=base_full[take]
    cand=anchor+center[:,None]+base.OFFSETS[None,:]
    tgr=np.interp(cand,tt,tg); tgr_aff=slope*tgr+intercept
    tgr_lo=np.interp(cand-4,tt,tg); tgr_hi=np.interp(cand+4,tt,tg)
    hg=base.robust(gr5[take])[:,None]; hc=base.robust(gr25[take])[:,None]
    tg_n=base.robust(tgr); ta_n=base.robust(tgr_aff)
    z=h.Z.to_numpy(float); md=h.MD.to_numpy(float)
    dz=(z[take]-z[ps-1])/50.; bgrad=np.gradient(center)/2.
    bcurv=np.gradient(np.gradient(center))/2.
    prog=(md[take]-md[ps-1])/max(md[-1]-md[ps-1],1.)
    known_take=known[take].astype(np.float32)
    zero_support=known_take[:,None]*np.exp(-.5*(base.OFFSETS[None,:]/1.5)**2)
    shp=(len(take),len(base.OFFSETS))
    X=np.stack([
      np.broadcast_to(hg,shp),np.broadcast_to(hc,shp),
      np.broadcast_to(tg_n,shp),np.broadcast_to(hg,shp)-np.broadcast_to(tg_n,shp),
      np.abs(np.broadcast_to(hg,shp)-np.broadcast_to(tg_n,shp)),
      np.broadcast_to(base.robust(tgr_hi-tgr_lo),shp),
      np.broadcast_to(dz[:,None],shp),np.broadcast_to(prog[:,None],shp),
      np.broadcast_to((base.OFFSETS/50.)[None,:],shp),
      np.broadcast_to((center/30.)[:,None],shp),
      np.broadcast_to(bgrad[:,None],shp),np.broadcast_to(bcurv[:,None],shp),
      np.broadcast_to(known_take[:,None],shp),zero_support,
      np.abs(np.broadcast_to(hg,shp)-np.broadcast_to(ta_n,shp)),
    ],0).astype(np.float32)
    truth=h.TVT.to_numpy(float)[take]-anchor
    y=truth-center
    cls=np.rint((y-base.OFFSETS[0])/(base.OFFSETS[1]-base.OFFSETS[0])).astype(np.int64)
    eval_mask=take>=ps
    # Loss remains strictly on hidden suffix; support is input evidence only.
    valid=eval_mask&(cls>=0)&(cls<len(base.OFFSETS))
    return dict(well=wid,X=X,cls=cls,valid=valid,eval=eval_mask,
      y=y.astype(np.float32),rows=take,n=len(h),ps=ps,anchor=anchor,base=center,
      base_suffix=np.asarray(v4_delta,np.float32),
      global_idx=np.asarray(global_idx,np.int64))

class AlignNet(base.AlignNet):
    def __init__(self,cin=15,width=32):
        nn.Module.__init__(self)
        self.inp=nn.Conv2d(cin,width,1)
        blocks=[]
        for d in (1,2,4,8,16,32):
            blocks.append(nn.Sequential(
              nn.Conv2d(width,width,(7,5),padding=(3*d,2),dilation=(d,1),groups=width),
              nn.Conv2d(width,width,1),nn.GELU(),nn.GroupNorm(4,width)))
        self.blocks=nn.ModuleList(blocks); self.head=nn.Conv2d(width,1,1)

base.load_well=load_well
base.AlignNet=AlignNet

if __name__=="__main__":
    base.main()
