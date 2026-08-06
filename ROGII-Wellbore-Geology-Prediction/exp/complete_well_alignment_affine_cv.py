"""Original V4 lattice plus legal visible-prefix affine GR calibration only."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch.nn as nn
import complete_well_alignment_unet_v4center_cv as base

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/train"
base.OUT=ROOT/"exp/results/alignment_unet_affine"
original_load=base.load_well

def load_well(path,v4_delta,global_idx):
    item=original_load(path,v4_delta,global_idx)
    if item is None: return None
    h=pd.read_csv(path)
    tw=pd.read_csv(DATA/f"{item['well']}__typewell.csv").sort_values("TVT")
    ps=item["ps"]
    tg=tw.GR.interpolate(limit_direction="both").fillna(tw.GR.median()).to_numpy(float)
    tt=tw.TVT.to_numpy(float)
    hg_full=h.GR.interpolate(limit_direction="both").fillna(h.GR.median()).to_numpy(float)
    hg5=pd.Series(hg_full).rolling(11,center=True,min_periods=1).mean().to_numpy()
    ref=np.interp(h.TVT_input.iloc[:ps].to_numpy(float),tt,tg)
    obs=hg5[:ps]; ok=np.isfinite(ref)&np.isfinite(obs)
    if ok.sum()>=8 and np.std(ref[ok])>1e-5:
        slope,intercept=np.linalg.lstsq(
            np.c_[ref[ok],np.ones(ok.sum())],obs[ok],rcond=None)[0]
        slope=float(np.clip(slope,.25,2.5)); intercept=float(np.clip(intercept,-100,100))
    else: slope,intercept=1.,0.
    cand=item["anchor"]+item["base"][:,None]+base.OFFSETS[None,:]
    aff=slope*np.interp(cand,tt,tg)+intercept
    affn=base.robust(aff)
    hgn=base.robust(hg5[item["rows"]])[:,None]
    shp=aff.shape
    extra=np.stack([affn,np.abs(np.broadcast_to(hgn,shp)-affn)],0).astype(np.float32)
    item["X"]=np.concatenate([item["X"],extra],axis=0)
    return item

class AlignNet(base.AlignNet):
    def __init__(self,cin=14,width=32):
        nn.Module.__init__(self)
        self.inp=nn.Conv2d(cin,width,1)
        self.blocks=nn.ModuleList([nn.Sequential(
          nn.Conv2d(width,width,(7,5),padding=(3*d,2),dilation=(d,1),groups=width),
          nn.Conv2d(width,width,1),nn.GELU(),nn.GroupNorm(4,width))
          for d in (1,2,4,8,16,32)])
        self.head=nn.Conv2d(width,1,1)

base.load_well=load_well
base.AlignNet=AlignNet
if __name__=="__main__": base.main()
