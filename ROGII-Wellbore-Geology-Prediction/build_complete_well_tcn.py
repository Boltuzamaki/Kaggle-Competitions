"""Build a grouped-CV complete-well sequence model for ROGII."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "kernels" / "complete_well_tcn"
OUT.mkdir(parents=True, exist_ok=True)

CODE = r'''
import os, glob, random, time, json
from pathlib import Path
import numpy as np, pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold

SEED=20260730; L=1024; EPOCHS=25; BATCH=8
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
cuda_ok=False
if torch.cuda.is_available():
    try:
        sm=10*torch.cuda.get_device_capability(0)[0]+torch.cuda.get_device_capability(0)[1]
        supported=[int(x.split("_")[1]) for x in torch.cuda.get_arch_list() if x.startswith("sm_")]
        cuda_ok=bool(supported) and sm>=min(supported)
    except Exception: cuda_ok=False
DEVICE=torch.device("cuda" if cuda_ok else "cpu")
if DEVICE.type=="cpu": torch.set_num_threads(min(8,os.cpu_count() or 4))
print("device",DEVICE,"cuda_visible",torch.cuda.is_available(),"compatible",cuda_ok,flush=True)

def root():
    for r in ["/kaggle/input/rogii-wellbore-geology-prediction",
              "/kaggle/input/competitions/rogii-wellbore-geology-prediction","data"]:
        if glob.glob(f"{r}/train/*__horizontal_well.csv"): return Path(r)
    hits=glob.glob("/kaggle/input/**/train/*__horizontal_well.csv",recursive=True)
    if hits: return Path(hits[0]).parent.parent
    raise FileNotFoundError("competition data")
DATA=root(); print(DATA,flush=True)

GRID=np.linspace(0,1,L)
def smooth(x,w=11):
    return pd.Series(x).rolling(w,center=True,min_periods=1).mean().to_numpy()
def encode(path):
    w=path.name.split("__")[0]
    h=pd.read_csv(path)
    ps=int(h.TVT_input.notna().sum())
    if ps<60 or ps>=len(h)-3: return None
    n=len(h); q=np.linspace(0,1,n); cut=(ps-1)/max(n-1,1)
    md=h.MD.to_numpy(float); x=h.X.to_numpy(float); y=h.Y.to_numpy(float); z=h.Z.to_numpy(float)
    gr=h.GR.interpolate(limit_direction="both").fillna(h.GR.mean()).to_numpy(float)
    tvi=h.TVT_input.to_numpy(float); anchor=float(tvi[ps-1])
    # Normalize each well at prediction start; preserve the entire future
    # trajectory because steering decisions are downstream evidence of geology.
    def it(v): return np.interp(GRID,q,v)
    dz=z-z[ps-1]; dx=x-x[ps-1]; dy=y-y[ps-1]
    dmd=(md-md[ps-1])/1000.0
    slope=np.gradient(z)/(np.gradient(md)+1e-6)
    az=np.unwrap(np.arctan2(np.gradient(y),np.gradient(x)))
    steer=np.gradient(slope)/(np.gradient(md)+1e-6)*1000
    turn=np.gradient(az)/(np.gradient(md)+1e-6)*1000
    g=(gr-np.nanmedian(gr))/(np.nanstd(gr)+1e-6)
    gm=smooth(g,31); gd=np.gradient(g)
    vis=np.isfinite(tvi).astype(float)
    tvp=np.where(np.isfinite(tvi),tvi-anchor,0.0)
    channels=[dmd,dz/50,dx/5000,dy/5000,slope*20,steer,turn,g,gm,gd,vis,tvp/30]
    X=np.stack([it(np.nan_to_num(v)) for v in channels]).astype(np.float32)
    tail=(GRID>cut).astype(np.float32)
    if "TVT" in h:
        target=it(h.TVT.to_numpy(float)-anchor).astype(np.float32)
    else: target=np.zeros(L,np.float32)
    return {"well":w,"x":X,"y":target,"mask":tail,"q":q,"cut":cut,
            "target_full":h.TVT.to_numpy(float)-anchor if "TVT" in h else None}

paths=sorted((DATA/"train").glob("*__horizontal_well.csv"))
items=[v for v in (encode(p) for p in paths) if v is not None]
print("items",len(items),flush=True)

class Wells(Dataset):
    def __init__(self, ids, augment=False): self.ids=list(ids); self.augment=augment
    def __len__(self): return len(self.ids)
    def __getitem__(self,k):
        v=items[self.ids[k]]; x=v["x"].copy()
        if self.augment:
            x[7:10]+=np.random.normal(0,.025,x[7:10].shape).astype(np.float32)
        return torch.from_numpy(x),torch.from_numpy(v["y"]),torch.from_numpy(v["mask"])

class Block(nn.Module):
    def __init__(self,c,d):
        super().__init__()
        self.n=nn.Sequential(nn.Conv1d(c,c,5,padding=2*d,dilation=d,groups=c),
          nn.Conv1d(c,c,1),nn.GroupNorm(8,c),nn.GELU(),nn.Dropout(.08),
          nn.Conv1d(c,c,5,padding=2*d,dilation=d,groups=c),nn.Conv1d(c,c,1))
    def forward(self,x): return x+self.n(x)
class Net(nn.Module):
    def __init__(self):
        super().__init__(); c=64
        self.net=nn.Sequential(nn.Conv1d(12,c,1),*[Block(c,d) for d in [1,2,4,8,16,32,64,128]],
                               nn.GroupNorm(8,c),nn.GELU(),nn.Conv1d(c,1,1))
    def forward(self,x): return self.net(x).squeeze(1)

def lossfn(p,y,m):
    den=m.sum().clamp_min(1)
    level=(((p-y)**2)*m).sum()/den
    dm=m[:,1:]*m[:,:-1]; pd=p[:,1:]-p[:,:-1]; yd=y[:,1:]-y[:,:-1]
    slope=(((pd-yd)**2)*dm).sum()/dm.sum().clamp_min(1)
    # Anchor continuity prevents an arbitrary level jump at the cut.
    return level+.15*slope

@torch.no_grad()
def predict(model,ids):
    model.eval(); out={}
    dl=DataLoader(Wells(ids),batch_size=BATCH,shuffle=False,num_workers=0)
    for batch_ids,(x,_,_) in zip([ids[i:i+BATCH] for i in range(0,len(ids),BATCH)],dl):
        p=model(x.to(DEVICE)).cpu().numpy()
        for j,i in enumerate(batch_ids): out[i]=p[j]
    return out

groups=np.array([v["well"] for v in items]); oof={}
fold_rows=[]
for fold,(tr,va) in enumerate(GroupKFold(5).split(np.arange(len(items)),groups=groups)):
    torch.manual_seed(SEED+fold); model=Net().to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
    dl=DataLoader(Wells(tr,True),batch_size=BATCH,shuffle=True,num_workers=0,pin_memory=DEVICE.type=="cuda")
    best=1e30; state=None
    for epoch in range(EPOCHS):
        model.train(); losses=[]
        for x,y,m in dl:
            x,y,m=x.to(DEVICE),y.to(DEVICE),m.to(DEVICE)
            opt.zero_grad(set_to_none=True); loss=lossfn(model(x),y,m); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step(); losses.append(float(loss))
        sch.step()
        if epoch%5==0 or epoch==EPOCHS-1: print("fold",fold,"epoch",epoch,"loss",np.mean(losses),flush=True)
    pp=predict(model,va); oof.update(pp)
    sse=nrow=0
    for i in va:
        v=items[i]; pred=np.interp(v["q"],GRID,pp[i])
        sel=v["q"]>v["cut"]; actual=v["target_full"]
        sse+=float(np.sum((actual[sel]-pred[sel])**2)); nrow+=int(sel.sum())
    rmse=float(np.sqrt(sse/nrow)); fold_rows.append({"fold":fold,"rows":nrow,"rmse":rmse,"sse":sse})
    print("FOLD_RMSE",fold,rmse,flush=True)
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

sse=nrow=const_sse=0; well_rows=[]
for i,v in enumerate(items):
    pred=np.interp(v["q"],GRID,oof[i]); sel=v["q"]>v["cut"]; actual=v["target_full"]
    err=actual[sel]-pred[sel]; s=float(np.sum(err**2)); cs=float(np.sum(actual[sel]**2))
    sse+=s; const_sse+=cs; nrow+=int(sel.sum())
    well_rows.append({"well":v["well"],"rows":int(sel.sum()),"rmse":float(np.sqrt(np.mean(err**2)))})
summary={"pooled_rmse":float(np.sqrt(sse/nrow)),"constant_rmse":float(np.sqrt(const_sse/nrow)),
         "rows":nrow,"folds":fold_rows}
print("SUMMARY",json.dumps(summary,indent=2),flush=True)
pd.DataFrame(well_rows).to_csv("/kaggle/working/complete_well_cv_wells.csv",index=False)
pd.DataFrame(fold_rows).to_csv("/kaggle/working/complete_well_cv_folds.csv",index=False)
open("/kaggle/working/complete_well_cv_summary.json","w").write(json.dumps(summary,indent=2))
assert not os.path.exists("/kaggle/working/submission.csv")
'''

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ROGII Complete-Well TCN CV\n",
                "Non-autoregressive sequence-to-sequence model. Five-fold grouped CV only; no submission.\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": CODE.splitlines(keepends=True),
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(OUT / "rogii-complete-well-tcn-cv.ipynb").write_text(
    json.dumps(nb, indent=1), encoding="utf-8"
)
meta = {
    "id": "boltuzamaki/rogii-complete-well-tcn-cv",
    "title": "ROGII Complete Well TCN CV",
    "code_file": "rogii-complete-well-tcn-cv.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "dataset_sources": [],
    "competition_sources": ["rogii-wellbore-geology-prediction"],
    "kernel_sources": [],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
compile(CODE, "<complete-well-tcn>", "exec")
print(OUT)
