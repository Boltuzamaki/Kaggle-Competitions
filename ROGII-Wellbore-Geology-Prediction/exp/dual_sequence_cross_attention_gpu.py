"""Raw horizontal/typewell cross-attention complete-well DCT pilot."""
from pathlib import Path
import json,joblib,sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"exp/results/dual_sequence_cross_attention";OUT.mkdir(parents=True,exist_ok=True)
H,T,L,K=512,384,384,16
EPOCHS=int(sys.argv[1]) if len(sys.argv)>1 else 80
d=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
v4=np.asarray(joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb7"],np.float32)
grouped=list(d.groupby("well",sort=False));wells=[w for w,_ in grouped]
B=np.stack([np.ones(L)]+[np.cos(np.pi*k*(np.arange(L)+.5)/L)
 for k in range(1,K)],1).astype(np.float32);Bi=np.linalg.pinv(B).astype(np.float32)
HH=[];TT=[];CC=[];RR=[];BASE=[];ORIG=[]
def interp_rows(a,n):
 a=np.asarray(a,float);return np.interp(np.linspace(0,len(a)-1,n),np.arange(len(a)),a)
for wi,(w,g) in enumerate(grouped):
 ix=g.index.to_numpy();hw=pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv")
 tw=pd.read_csv(ROOT/f"data/train/{w}__typewell.csv").sort_values("TVT")
 hs=pd.to_numeric(hw.GR,errors="coerce").interpolate(limit_direction="both")
 hg=hs.fillna(hs.median() if hs.notna().any() else 0).to_numpy(float)
 ps=int(np.flatnonzero(hw.TVT_input.notna().to_numpy())[-1])
 # Prefix 128 + hidden suffix 384 preserves the exact organizer split.
 hp=np.linspace(0,ps,128);hr=g.id.str.rsplit("_",n=1).str[1].astype(int).to_numpy()
 hsuf=np.interp(np.linspace(0,len(hr)-1,384),np.arange(len(hr)),hr)
 ridx=np.r_[hp,hsuf]; ridx_i=np.arange(len(hw))
 med=np.median(hg);scale=max(1.4826*np.median(abs(hg-med)),8)
 gr=np.interp(ridx,ridx_i,hg)
 md=np.interp(ridx,ridx_i,hw.MD);z=np.interp(ridx,ridx_i,hw.Z)
 known=np.zeros(H);known[:128]=1
 vt=np.zeros(H);vt[:128]=(np.interp(hp,ridx_i,
   hw.TVT_input.interpolate(limit_direction="both"))-float(g.last_known_tvt.iloc[0]))/30
 hfeat=np.stack([(gr-med)/scale,(md-md[0])/max(np.ptp(md),1),
  (z-z[0])/max(np.ptp(z),1),np.gradient(z)/np.maximum(np.gradient(md),1e-3),
  vt,known,np.r_[np.zeros(128),np.ones(384)]])
 tgr=pd.to_numeric(tw.GR,errors="coerce").interpolate(limit_direction="both")
 tgr=tgr.fillna(tgr.median() if tgr.notna().any() else 0).to_numpy(float)
 ti=np.linspace(0,len(tw)-1,T);tgv=np.interp(ti,np.arange(len(tw)),tgr)
 tv=np.interp(ti,np.arange(len(tw)),tw.TVT)
 tm=np.median(tgv);ts=max(1.4826*np.median(abs(tgv-tm)),8)
 tfeat=np.stack([(tgv-tm)/ts,(tv-float(g.last_known_tvt.iloc[0]))/100,
                 np.gradient(tgv)/ts,np.linspace(-1,1,T)])
 pos=np.linspace(0,len(g)-1,L)
 base=np.interp(pos,np.arange(len(g)),v4[ix])
 target=np.interp(pos,np.arange(len(g)),g.target)-base
 HH.append(hfeat.astype(np.float32));TT.append(tfeat.astype(np.float32))
 CC.append(Bi@target);RR.append(target.astype(np.float32));BASE.append(base.astype(np.float32));ORIG.append(ix)
 if wi%100==0:print("features",wi,flush=True)
HH=np.asarray(HH,np.float32);TT=np.asarray(TT,np.float32)
CC=np.asarray(CC,np.float32);RR=np.asarray(RR,np.float32)
BASE=np.asarray(BASE,np.float32)

class Model(nn.Module):
 def __init__(self):
  super().__init__();dim=48
  self.he=nn.Sequential(nn.Conv1d(7,dim,9,padding=4),nn.GELU(),
                        nn.Conv1d(dim,dim,7,padding=3),nn.GELU())
  self.te=nn.Sequential(nn.Conv1d(4,dim,9,padding=4),nn.GELU(),
                        nn.Conv1d(dim,dim,7,padding=3),nn.GELU())
  self.hp=nn.Parameter(torch.randn(1,H,dim)*.02)
  self.tp=nn.Parameter(torch.randn(1,T,dim)*.02)
  self.attn=nn.MultiheadAttention(dim,4,batch_first=True,dropout=.1)
  self.norm=nn.LayerNorm(dim)
  self.dec=nn.Sequential(nn.Conv1d(dim*2,64,9,padding=4),nn.GELU(),
   nn.Conv1d(64,64,9,padding=8,dilation=2),nn.GELU())
  self.head=nn.Sequential(nn.AdaptiveAvgPool1d(24),nn.Flatten(),
   nn.Linear(64*24,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,K))
 def forward(self,h,t):
  he=self.he(h).transpose(1,2)+self.hp;te=self.te(t).transpose(1,2)+self.tp
  a,_=self.attn(he,te,te,need_weights=False)
  z=torch.cat([he[:,128:],self.norm(a[:,128:]+he[:,128:])],2).transpose(1,2)
  return self.head(self.dec(z))

device="cuda" if torch.cuda.is_available() else "cpu"
tr,va=list(GroupKFold(5).split(HH,groups=wells))[0]
# Outer-training wells only: simulate later prediction starts. Validation wells
# retain the single exact organizer mask and are never augmented.
fractions=np.r_[0.,np.linspace(.15,.70,14)]
AH=[];AT=[];AC=[]
for i in tr:
 full=HH[i]; suffix=full[:,128:]; truth_delta=BASE[i]+RR[i]
 for frac in fractions:
  k=int(round(frac*(L-1)))
  # Prefix history = original visible prefix plus newly revealed truth rows.
  hist=np.concatenate([full[:,:128],suffix[:,:k]],axis=1)
  hp=np.linspace(0,hist.shape[1]-1,128)
  hh=np.empty_like(full)
  for c in range(4):hh[c,:128]=np.interp(hp,np.arange(hist.shape[1]),hist[c])
  vis_tvt=np.r_[full[4,:128],truth_delta[:k]/30]
  hh[4,:128]=np.interp(hp,np.arange(len(vis_tvt)),vis_tvt)
  hh[5,:128]=1;hh[6,:128]=0
  # Remaining hidden segment is resampled back to the fixed decoder length.
  rem=np.arange(k,L);q=np.linspace(k,L-1,L)
  for c in range(4):hh[c,128:]=np.interp(q,np.arange(L),suffix[c])
  hh[4,128:]=0;hh[5,128:]=0;hh[6,128:]=1
  anchor=float(truth_delta[k]);tt=TT[i].copy();tt[1]-=anchor/100
  residual=np.interp(q,np.arange(L),RR[i]).astype(np.float32)
  AH.append(hh);AT.append(tt);AC.append(Bi@residual)
AH=np.asarray(AH,np.float32);AT=np.asarray(AT,np.float32);AC=np.asarray(AC,np.float32)
print("augmented_train_samples",len(AH),flush=True)
cs=np.maximum(AC.std(0),.2).astype(np.float32);cst=torch.tensor(cs,device=device)
dl=DataLoader(TensorDataset(torch.tensor(AH),torch.tensor(AT),
 torch.tensor(AC/cs)),batch_size=24,shuffle=True)
model=Model().to(device);opt=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=3e-3)
Bt=torch.tensor(B,device=device);best=1e9;state=None
for ep in range(EPOCHS):
 model.train()
 for h,t,c in dl:
  h,t,c=h.to(device),t.to(device),c.to(device)
  # Legal augmentation: GR gain/noise/dropout and mild sequence resampling jitter.
  h=h.clone();t=t.clone()
  h[:,0]*=(.9+.2*torch.rand(len(h),1,device=device));h[:,0]+=torch.randn_like(h[:,0])*.05
  t[:,0]*=(.9+.2*torch.rand(len(t),1,device=device));t[:,0]+=torch.randn_like(t[:,0])*.05
  h[:,0]*=(torch.rand_like(h[:,0])>.03);t[:,0]*=(torch.rand_like(t[:,0])>.03)
  pc=model(h,t);loss=(((pc*cst)@Bt.T-(c*cst)@Bt.T)/12).square().mean()+.05*(pc-c).square().mean()
  opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
 if ep%5==0 or ep==EPOCHS-1:
  model.eval()
  with torch.no_grad():pc=model(torch.tensor(HH[va],device=device),
    torch.tensor(TT[va],device=device)).cpu().numpy()*cs
  score=float(np.sqrt(np.mean((pc@B.T-RR[va])**2)))
  if score<best:best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
model.load_state_dict(state);model.eval()
with torch.no_grad():pc=model(torch.tensor(HH[va],device=device),
 torch.tensor(TT[va],device=device)).cpu().numpy()*cs
pred=pc@B.T
yy=[];bb=[];pp=[];oo=[]
for q,i in enumerate(va):
 ix=ORIG[i];n=len(ix);r=np.interp(np.arange(n),np.linspace(0,n-1,L),pred[q])
 o=np.interp(np.arange(n),np.linspace(0,n-1,L),B@(Bi@RR[i]))
 yy.append(d.target.to_numpy()[ix]);bb.append(v4[ix]);pp.append(v4[ix]+r);oo.append(v4[ix]+o)
yy,bb,pp,oo=map(np.concatenate,(yy,bb,pp,oo))
rm=lambda p:float(np.sqrt(np.mean((p-yy)**2)))
grid=[{"blend":float(a),"rmse":rm((1-a)*bb+a*pp)} for a in np.linspace(0,1,21)]
summary={"device":device,"fold":0,"wells":len(va),"rows":len(yy),"v4":rm(bb),
 "network":rm(pp),"dct_oracle":rm(oo),"best_blend":min(grid,key=lambda q:q["rmse"]),
 "resampled_residual_rmse":best,"protocol":"raw dual sequence cross-attention; exact prefix/suffix; outer whole-well fold"}
(OUT/"pilot_summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
