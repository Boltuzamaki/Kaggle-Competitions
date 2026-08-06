"""One-complete-well GPU cost-volume -> smooth DCT residual model."""
from pathlib import Path
import json,joblib,sys
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"exp/results/complete_well_costvolume_dct";OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/"exp"))
from whole_well_gr_warp_selector_cv import robust_calibrate
L=384;KCOEF=16;SH=np.arange(-60,61,3,dtype=np.float32)
EPOCHS=int(sys.argv[1]) if len(sys.argv)>1 else 90
FULL=bool(int(sys.argv[2])) if len(sys.argv)>2 else False

d=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
v4=np.asarray(joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb7"],np.float32)
grouped=list(d.groupby("well",sort=False))
wlist=[w for w,_ in grouped]
xgrid=np.linspace(0,1,L,dtype=np.float32)
B=np.stack([np.ones(L)]+[np.cos(np.pi*k*(np.arange(L)+.5)/L)
                         for k in range(1,KCOEF)],1).astype(np.float32)
Bpinv=np.linalg.pinv(B).astype(np.float32)
CV=[];CTX=[];COEF=[];CURVE=[];ORIG=[]
for wi,(w,g) in enumerate(grouped):
 ix=g.index.to_numpy();n=len(g)
 pos=np.linspace(0,n-1,L)
 def rs(a):return np.interp(pos,np.arange(n),np.asarray(a,float)).astype(np.float32)
 hw=pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv")
 tw=pd.read_csv(ROOT/f"data/train/{w}__typewell.csv").sort_values("TVT")
 hs=pd.to_numeric(hw.GR,errors="coerce").interpolate(limit_direction="both")
 hall=hs.fillna(hs.median() if hs.notna().any() else 0).to_numpy(float)
 hr=g.id.astype(str).str.rsplit("_",n=1).str[1].astype(int).to_numpy()
 hgr=rs(hall[hr]);a,b,pfx=robust_calibrate(hw,tw)
 ts=pd.to_numeric(tw.GR,errors="coerce").interpolate(limit_direction="both")
 tg=ts.fillna(ts.median() if ts.notna().any() else 0).to_numpy(float)
 tt=tw.TVT.to_numpy(float);last=float(g.last_known_tvt.iloc[0])
 base=rs(v4[ix]);truth=rs(g.target.to_numpy(float));absbase=last+base
 scale=max(1.4826*np.median(np.abs(hgr-np.median(hgr))),8.)
 cand=a*np.vstack([np.interp(absbase+s,tt,tg) for s in SH])+b
 cv=np.clip((hgr[None]-cand)/scale,-5,5)
 # Add multiscale matching planes; shape is (3, shifts, length).
 cvs=np.stack([cv,gaussian_filter1d(cv,3,axis=1),
               gaussian_filter1d(cv,10,axis=1)]).astype(np.float32)
 ctx=np.stack([
  (hgr-np.median(hgr))/scale,
  gaussian_filter1d(hgr,5)/scale,
  gaussian_filter1d(hgr,20)/scale,
  rs(g.d_z)/max(float(g.d_z.std()),1),
  rs(g.dz_dmd),rs(g.d_xy)/max(float(g.d_xy.max()),1),
  base/30,np.full(L,pfx/30),np.full(L,a),np.full(L,scale/30)
 ]).astype(np.float32)
 residual=(truth-base).astype(np.float32);coef=Bpinv@residual
 CV.append(cvs);CTX.append(ctx);COEF.append(coef);CURVE.append(residual)
 ORIG.append((ix,n))
 if wi%100==0:print("features",wi,flush=True)
CV=np.asarray(CV);CTX=np.asarray(CTX);COEF=np.asarray(COEF);CURVE=np.asarray(CURVE)

class Net(nn.Module):
 def __init__(self):
  super().__init__()
  self.cv=nn.Sequential(
   nn.Conv2d(3,16,(5,9),padding=(2,4)),nn.GELU(),
   nn.Conv2d(16,24,(5,7),padding=(2,3)),nn.GELU(),
   nn.Conv2d(24,24,(3,5),padding=(1,2)),nn.GELU())
  self.seq=nn.Sequential(
   nn.Conv1d(34,64,9,padding=4),nn.GELU(),
   nn.Conv1d(64,64,9,padding=8,dilation=2),nn.GELU(),
   nn.Conv1d(64,64,9,padding=16,dilation=4),nn.GELU())
  self.head=nn.Sequential(nn.AdaptiveAvgPool1d(24),nn.Flatten(),
   nn.Linear(64*24,256),nn.GELU(),nn.Dropout(.15),nn.Linear(256,KCOEF))
 def forward(self,cv,ctx):
  z=self.cv(cv).mean(2)
  return self.head(self.seq(torch.cat([z,ctx],1)))

device="cuda" if torch.cuda.is_available() else "cpu"
splits=list(GroupKFold(5).split(np.arange(len(wlist)),groups=wlist))
folds=range(5) if FULL else range(1)
pred=np.zeros_like(CURVE);fid=np.full(len(wlist),-1)
fold_rows=[]
Bt=torch.tensor(B,device=device)
for fold in folds:
 tr,va=splits[fold]
 # Normalize targets globally on training wells only.
 cscale=np.maximum(np.std(COEF[tr],axis=0),.2).astype(np.float32)
 cscale_t=torch.tensor(cscale,device=device)
 ds=TensorDataset(torch.tensor(CV[tr]),torch.tensor(CTX[tr]),
                  torch.tensor(COEF[tr]/cscale))
 dl=DataLoader(ds,batch_size=24,shuffle=True)
 model=Net().to(device);opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-3)
 best=None;beststate=None
 for ep in range(EPOCHS):
  model.train()
  for cv,ctx,coef in dl:
   cv,ctx,coef=cv.to(device),ctx.to(device),coef.to(device)
   pc=model(cv,ctx); curve=(pc*cscale_t)@Bt.T
   true=(coef*cscale_t)@Bt.T
   loss=((curve-true)/12).square().mean()+.08*(pc-coef).square().mean()
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
  if ep%5==0 or ep==EPOCHS-1:
   model.eval()
   with torch.no_grad():
    pc=model(torch.tensor(CV[va],device=device),
             torch.tensor(CTX[va],device=device)).cpu().numpy()*cscale
   pv=pc@B.T;score=float(np.sqrt(np.mean((pv-CURVE[va])**2)))
   if best is None or score<best:best=score;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 model.load_state_dict(beststate);model.eval()
 with torch.no_grad():
  pc=model(torch.tensor(CV[va],device=device),
           torch.tensor(CTX[va],device=device)).cpu().numpy()*cscale
 pred[va]=pc@B.T;fid[va]=fold
 fold_rows.append({"fold":fold,"wells":len(va),"resampled_residual_rmse":best})
 print(fold_rows[-1],flush=True)

# Exact original-row scoring for completed folds.
yy=[];bb=[];pp=[];oo=[]
for i,(ix,n) in enumerate(ORIG):
 if fid[i]<0:continue
 q=np.interp(np.arange(n),np.linspace(0,n-1,L),pred[i])
 oracle=np.interp(np.arange(n),np.linspace(0,n-1,L),B@(Bpinv@CURVE[i]))
 yy.append(d.target.to_numpy(float)[ix]);bb.append(v4[ix]);pp.append(v4[ix]+q);oo.append(v4[ix]+oracle)
yy,bb,pp,oo=map(np.concatenate,(yy,bb,pp,oo))
def rmse(a):return float(np.sqrt(np.mean((a-yy)**2)))
grid=[{"blend":float(a),"rmse":rmse((1-a)*bb+a*pp)} for a in np.linspace(0,1,21)]
summary={"device":device,"folds_completed":len(fold_rows),"wells":int((fid>=0).sum()),
 "rows":len(yy),"v4":rmse(bb),"network":rmse(pp),"dct_oracle":rmse(oo),
 "best_blend":min(grid,key=lambda q:q["rmse"]),"folds":fold_rows,
 "protocol":"whole-well GKF; full hidden GR + visible prefix TVT; typewell GR; honest V4 center"}
(OUT/("full_summary.json" if FULL else "pilot_summary.json")).write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
