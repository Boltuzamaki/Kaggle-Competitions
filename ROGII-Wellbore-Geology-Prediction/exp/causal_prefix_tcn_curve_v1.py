"""GPU TCN: legal visible-prefix sequences -> anchored suffix residual FPCA."""
from pathlib import Path
import hashlib,json,random
import numpy as np,pandas as pd, torch
from torch import nn
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz'
OUT=ROOT/'exp/results/causal_prefix_tcn_curve_v1';OUT.mkdir(parents=True,exist_ok=True)
Z=np.load(SRC,allow_pickle=True);W=Z['wells'].astype(str);C=Z['true'];lens=Z['length']; L=256
def rs(x):return np.interp(np.linspace(0,1,L),np.linspace(0,1,len(x)),np.nan_to_num(x))
seq=[]
for j,w in enumerate(W):
 h=pd.read_csv(ROOT/'data/train'/f'{w}__horizontal_well.csv',usecols=['MD','X','Y','Z','GR','TVT_input'])
 h=h[h.TVT_input.notna()];md=h.MD.to_numpy();tv=h.TVT_input.to_numpy();z=h.Z.to_numpy();gr=h.GR.to_numpy()
 dt=np.r_[0,np.diff(tv)];dz=np.r_[0,np.diff(z)]; er=dt-dz; cum=np.cumsum(er);cum-=cum[-1]
 dx=np.r_[0,np.diff(h.X)];dy=np.r_[0,np.diff(h.Y)]
 tw=pd.read_csv(ROOT/'data/train'/f'{w}__typewell.csv',usecols=['TVT','GR']).dropna()
 chans=[er,cum,dt,dz,gr,dx,dy,np.gradient(tv)/(np.gradient(md)+1e-6),tw.GR.to_numpy(),tw.TVT.to_numpy()-tw.TVT.iloc[-1]]
 seq.append(np.array([rs(x) for x in chans],np.float32))
X=np.asarray(seq); target=C-C[:,[0]] # exact continuous zero correction at boundary
folds=list(GroupKFold(5).split(X,groups=W));device='cuda' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
 def __init__(self,out):
  super().__init__(); self.body=nn.Sequential(
   nn.Conv1d(10,48,7,padding=3),nn.GELU(),nn.BatchNorm1d(48),
   nn.Conv1d(48,64,7,padding=6,dilation=2),nn.GELU(),nn.BatchNorm1d(64),
   nn.Conv1d(64,96,7,padding=12,dilation=4),nn.GELU(),nn.AdaptiveAvgPool1d(1))
  self.head=nn.Sequential(nn.Flatten(),nn.Linear(96,96),nn.GELU(),nn.Dropout(.15),nn.Linear(96,out))
 def forward(self,x):return self.head(self.body(x))

pred=np.zeros_like(C); histories=[]
for f,(tr,va) in enumerate(folds):
 rng=np.random.default_rng(9100+f);perm=rng.permutation(tr);nv=max(60,len(tr)//7);cal=perm[:nv];fit=perm[nv:]
 mu=X[fit].mean((0,2),keepdims=True);sd=X[fit].std((0,2),keepdims=True)+1e-5
 cp=PCA(24,random_state=91).fit(target[fit]); A=cp.transform(target)
 am=A[fit].mean(0);ast=A[fit].std(0)+1e-5
 def tx(ix):return torch.tensor((X[ix]-mu)/sd,device=device)
 def ty(ix):return torch.tensor((A[ix]-am)/ast,dtype=torch.float32,device=device)
 torch.manual_seed(9200+f);m=Net(24).to(device);opt=torch.optim.AdamW(m.parameters(),2e-3,weight_decay=2e-3)
 best=(1e9,None,0); xb=tx(fit);yb=ty(fit);xc=tx(cal);yc=ty(cal)
 for ep in range(180):
  m.train();order=torch.randperm(len(fit),device=device)
  for q in order.split(64):
   opt.zero_grad();loss=((m(xb[q])-yb[q])**2).mean();loss.backward();opt.step()
  m.eval()
  with torch.no_grad():vl=float(((m(xc)-yc)**2).mean())
  if vl<best[0]-1e-4:best=(vl,{k:v.detach().cpu().clone() for k,v in m.state_dict().items()},ep)
  if ep-best[2]>25:break
 m.load_state_dict(best[1]);m.eval()
 with torch.no_grad(): pa=m(tx(va)).cpu().numpy()*ast+am
 pred[va]=cp.inverse_transform(pa);pred[va]-=pred[va][:,[0]]
 histories.append({'fold':f,'best_epoch':best[2],'val_loss':best[0]});print(histories[-1],flush=True)

s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=s['y'];groups=s['groups'].astype(str);base=.1*s['accepted']+.9*s['replacement']
cuts=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1,len(groups)];ixs={groups[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])}
rp=np.empty_like(y)
for w,c in zip(W,pred):
 ix=ixs[w];rp[ix]=np.interp(np.linspace(0,1,len(ix)),np.linspace(0,1,len(c)),c)
wf={w:f for f,(_,v) in enumerate(folds) for w in W[v]};rf=np.array([wf[w] for w in groups])
def rm(p,ix=None):
 if ix is None:ix=np.ones(len(y),bool)
 return float(np.sqrt(np.mean((y[ix]-p[ix])**2)))
rows=[]
for a in [0,.05,.1,.2,.35,.5,.75,1]:
 p=base+a*rp;fs=[rm(p,rf==f) for f in range(5)];rows.append(dict(alpha=a,rmse=rm(p),**{f'f{f}':v for f,v in enumerate(fs)}))
d=pd.DataFrame(rows).sort_values('rmse');d.to_csv(OUT/'blend_grid.csv',index=False);np.savez_compressed(OUT/'predictions.npz',wells=W,curve=pred)
summary={'device':device,'input_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'baseline':rm(base),'best':d.iloc[0].to_dict(),'histories':histories,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
