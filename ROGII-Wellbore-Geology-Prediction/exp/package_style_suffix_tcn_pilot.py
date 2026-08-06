"""Strict pilot of the public package's missing rowwise suffix TCN architecture.

No packaged predictions or weights are read.  Legal raw-derived OOF feature
sequences already generated locally are resampled per well; a six-block
dilated TCN is retrained in each whole-well outer fold to correct Student-t.
"""
from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'exp/results/package_style_suffix_tcn_pilot_v1';OUT.mkdir(parents=True,exist_ok=True)
COLS=['d_md','d_z','d_xy','dz_dmd','gr','gr_m21','gr_s21','frac','slp_all','slp_50',
      'ktvt_rng','ktvt_std','pfx_gr_rmse','pf_d','pf3_d','pf_dis','beam_d','pf_vs_beam',
      'ncc8_d','ncc8_s','ncc15_d','ncc15_s','ncc25_d','ncc25_s',
      'tda-20','tdpf-20','tda-10','tdpf-10','tda0','tdpf0']
L=512
def rs(x):
 x=np.asarray(x,float);ok=np.isfinite(x)
 if not ok.any(): return np.zeros(L,np.float32)
 x=np.interp(np.arange(len(x)),np.flatnonzero(ok),x[ok])
 return np.interp(np.linspace(0,1,L),np.linspace(0,1,len(x)),x).astype(np.float32)

f=pd.read_pickle(ROOT/'r_v4b/train_feats.pkl')
z=np.load(ROOT/'exp/results/heel_calibrated_gr_datum/full_meta/oof.npz');f=f.iloc[z['global_indices'].astype(int)].reset_index(drop=True)
s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);assert np.allclose(s['y'],f.target)
f['_base']=.1*s['accepted']+.9*s['replacement']
groups=list(f.groupby('well',sort=True)); rng=np.random.default_rng(260804)
# Error/target-agnostic fixed sample.
take=np.sort(rng.choice(len(groups),200,replace=False)); groups=[groups[i] for i in take]
W=np.array([w for w,g in groups]); X=np.array([[rs(g[c]) for c in COLS] for w,g in groups])
Y=np.array([rs(g.target-g._base) for w,g in groups]); folds=list(GroupKFold(5).split(X,groups=W));device='cuda' if torch.cuda.is_available() else 'cpu'

class Block(nn.Module):
 def __init__(self,ci,co,d):
  super().__init__();self.c1=nn.Conv1d(ci,co,5,padding=2*d,dilation=d);self.c2=nn.Conv1d(co,co,5,padding=2*d,dilation=d);self.sk=nn.Conv1d(ci,co,1) if ci!=co else nn.Identity();self.dp=nn.Dropout(.15)
 def forward(self,x):
  y=self.dp(torch.relu(self.c1(x)));y=self.dp(self.c2(y));return torch.relu(y+self.sk(x))
class TCN(nn.Module):
 def __init__(self):
  super().__init__();b=[];ci=len(COLS)
  for i in range(6):b.append(Block(ci,64,2**i));ci=64
  self.net=nn.Sequential(*b);self.head=nn.Conv1d(64,1,1)
 def forward(self,x):return self.head(self.net(x)).squeeze(1)

P=np.zeros_like(Y);hist=[]
for k,(tr,va) in enumerate(folds):
 rr=np.random.default_rng(7710+k);p=rr.permutation(tr);cal=p[:max(20,len(tr)//7)];fit=p[max(20,len(tr)//7):]
 mu=X[fit].mean((0,2),keepdims=True);sd=X[fit].std((0,2),keepdims=True)+1e-5
 ym=Y[fit].mean();ys=Y[fit].std()+1e-5
 tx=lambda ix:torch.tensor((X[ix]-mu)/sd,dtype=torch.float32,device=device)
 ty=lambda ix:torch.tensor((Y[ix]-ym)/ys,dtype=torch.float32,device=device)
 torch.manual_seed(7780+k);m=TCN().to(device);opt=torch.optim.AdamW(m.parameters(),1e-3,weight_decay=2e-3)
 xb,yb,xc,yc=tx(fit),ty(fit),tx(cal),ty(cal);best=(1e9,None,0)
 for ep in range(100):
  m.train();order=torch.randperm(len(fit),device=device)
  for q in order.split(8):
   opt.zero_grad();loss=((m(xb[q])-yb[q])**2).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
  m.eval()
  with torch.no_grad():vl=float(((m(xc)-yc)**2).mean())
  if vl<best[0]-1e-4:best=(vl,{a:b.detach().cpu().clone() for a,b in m.state_dict().items()},ep)
  if ep-best[2]>15:break
 m.load_state_dict(best[1]);m.eval()
 with torch.no_grad():P[va]=m(tx(va)).cpu().numpy()*ys+ym
 hist.append({'fold':k,'epoch':best[2],'inner_loss':best[0]});print(hist[-1],flush=True)

row_y=[];row_b=[];row_p=[];row_f=[]
wf={W[j]:k for k,(_,va) in enumerate(folds) for j in va}
for i,(w,g) in enumerate(groups):
 row_y.append(g.target.to_numpy());row_b.append(g._base.to_numpy());row_p.append(np.interp(np.linspace(0,1,len(g)),np.linspace(0,1,L),P[i]));row_f.append(np.full(len(g),wf[w]))
y=np.concatenate(row_y);b=np.concatenate(row_b);p=np.concatenate(row_p);fi=np.concatenate(row_f)
rm=lambda q,m=None:float(np.sqrt(np.mean((y-(q))[m]**2))) if m is not None else float(np.sqrt(np.mean((y-q)**2)))
grid=[]
for a in [0,.025,.05,.1,.2,.35,.5,1]:
 q=b+a*p;grid.append({'alpha':a,'rmse':rm(q),**{f'f{k}':rm(q,fi==k) for k in range(5)}})
d=pd.DataFrame(grid);best=d.iloc[d.rmse.argmin()].to_dict();summary={'wells':len(W),'rows':len(y),'features':COLS,'device':device,'baseline':rm(b),'best':best,'histories':hist,'architecture':{'blocks':6,'channels':64,'kernel':5,'dilations':[1,2,4,8,16,32]},'package_weights_read':False,'package_predictions_read':False}
d.to_csv(OUT/'blend_grid.csv',index=False);(OUT/'summary.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(OUT/'oof.npz',wells=W,curve=P)
print(json.dumps(summary,indent=2))
