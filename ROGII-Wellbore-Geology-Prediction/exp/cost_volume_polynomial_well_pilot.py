"""Complete-well CNN: legal GR cost volume -> cubic V4 residual curve."""
from pathlib import Path
import json,joblib,random,time
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"exp/results/cost_volume_polynomial_well";OUT.mkdir(parents=True,exist_ok=True)
random.seed(812);np.random.seed(812);torch.manual_seed(812)
L=256;OFF=np.arange(-40,41,4,dtype=float)
f=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl");base=np.asarray(joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb7"],float)
y=f.target.to_numpy(float);wells=np.array(sorted(f.well.unique()));rng=np.random.RandomState(812)
keep=np.sort(rng.choice(wells,200,replace=False))
def affine(x,y):
 ok=np.isfinite(x)&np.isfinite(y);x,y=x[ok],y[ok];X=np.c_[x,np.ones(len(x))]
 c=np.linalg.lstsq(X,y,rcond=None)[0];sc=20.
 for _ in range(4):
  r=y-X@c;sc=1.4826*np.median(np.abs(r-np.median(r)))+1e-3;ww=1/np.maximum(1,np.abs(r)/(2.5*sc))
  c=np.linalg.lstsq(X*ww[:,None],y*ww,rcond=None)[0]
 return c[0],c[1],sc
items=[]
for wi,w in enumerate(keep):
 ix=np.asarray(f.groupby("well",sort=False).indices[w]);q=f.iloc[ix]
 h=pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv");t=pd.read_csv(ROOT/f"data/train/{w}__typewell.csv").sort_values("TVT")
 tv=t.TVT.to_numpy(float);tg=pd.to_numeric(t.GR,errors="coerce").interpolate(limit_direction="both").to_numpy(float)
 hg=pd.to_numeric(h.GR,errors="coerce").interpolate(limit_direction="both").to_numpy(float);vis=h.TVT_input.notna().to_numpy()
 a,b,sc=affine(np.interp(h.TVT_input[vis],tv,tg),hg[vis])
 rr=np.array([int(z.rsplit("_",1)[1]) for z in q.id],int);path=q.last_known_tvt.to_numpy(float)+base[ix]
 ref=a*np.vstack([np.interp(path+s,tv,tg) for s in OFF]).T+b
 cost=np.log1p((((hg[rr,None]-ref)/max(sc,5.))/2.)**2)
 u=np.linspace(0,len(ix)-1,L);lo=np.floor(u).astype(int);hi=np.minimum(lo+1,len(ix)-1);fr=u-lo
 interp=lambda A:(1-fr[:,None])*A[lo]+fr[:,None]*A[hi] if A.ndim==2 else (1-fr)*A[lo]+fr*A[hi]
 C=interp(cost);C=(C-C.min(1,keepdims=True));C=np.clip(C,0,3)
 zrel=h.Z.to_numpy(float)[rr]-h.Z.to_numpy(float)[rr[0]]
 gr=(hg[rr]-np.mean(hg[vis]))/(np.std(hg[vis])+5)
 ctx=np.column_stack([path-q.last_known_tvt.iloc[0],np.gradient(path),zrel/100,
                      np.gradient(zrel),gr,np.linspace(-1,1,len(ix))])
 inp=np.c_[C,interp(ctx)].T.astype(np.float32)
 rtrue=y[ix]-base[ix];coef=np.polynomial.legendre.legfit(np.linspace(-1,1,len(ix)),rtrue,3)
 items.append({"well":w,"ix":ix,"input":inp,"coef":coef.astype(np.float32),
               "sample_resid":interp(rtrue).astype(np.float32)})
 if (wi+1)%50==0:print("features",wi+1,flush=True)
outer_tr0,outer_va0=next(GroupKFold(5).split(items,groups=keep))
# Internal target-independent early-stop split from outer training.
inner=list(GroupKFold(5).split(outer_tr0,groups=keep[outer_tr0]))[0]
train_idx=outer_tr0[inner[0]];stop_idx=outer_tr0[inner[1]];test_idx=outer_va0
coef=np.stack([z["coef"] for z in items]);cm=coef[train_idx].mean(0);cs=coef[train_idx].std(0)+1e-3
X=torch.tensor(np.stack([z["input"] for z in items]));R=torch.tensor(np.stack([z["sample_resid"] for z in items]))
B=torch.tensor(np.polynomial.legendre.legvander(np.linspace(-1,1,L),3).astype(np.float32))
cm_t=torch.tensor(cm);cs_t=torch.tensor(cs)
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.enc=nn.Sequential(nn.Conv1d(27,64,7,padding=3),nn.GELU(),
   nn.Conv1d(64,96,7,padding=3),nn.GELU(),nn.Conv1d(96,128,7,padding=3),nn.GELU())
  self.head=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,4))
 def forward(self,x):
  z=self.enc(x);return self.head(torch.cat([z.mean(2),z.amax(2)],1))
dev="cuda" if torch.cuda.is_available() else "cpu";net=Net().to(dev);B=B.to(dev);cm_t=cm_t.to(dev);cs_t=cs_t.to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=2e-3);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,300)
def curve(out):return ((out*cs_t+cm_t)@B.T)
best=None
for ep in range(300):
 net.train();order=np.random.permutation(train_idx)
 for st in range(0,len(order),24):
  ii=order[st:st+24];pr=curve(net(X[ii].to(dev)));loss=((pr-R[ii].to(dev))**2).mean()
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
 sched.step()
 if ep%10==0:
  net.eval()
  with torch.no_grad():vl=((curve(net(X[stop_idx].to(dev)))-R[stop_idx].to(dev))**2).mean().item()
  if best is None or vl<best[0]:best=(vl,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},ep)
net.load_state_dict(best[1]);net.eval()
pred=np.zeros(len(y));oracle=np.zeros(len(y))
with torch.no_grad():pc=(net(X[test_idx].to(dev))*cs_t+cm_t).cpu().numpy()
fold_rows=[]
for j,ii0 in enumerate(test_idx):
 z=items[ii0];ix=z["ix"];xx=np.linspace(-1,1,len(ix))
 pred[ix]=base[ix]+np.polynomial.legendre.legval(xx,pc[j])
 oracle[ix]=base[ix]+np.polynomial.legendre.legval(xx,z["coef"])
foldix=np.concatenate([items[i]["ix"] for i in test_idx])
rm=lambda p:float(np.sqrt(np.mean((p[foldix]-y[foldix])**2)))
summary={"pilot_wells":200,"train_wells":len(train_idx),"early_stop_wells":len(stop_idx),
 "outer_holdout_wells":len(test_idx),"device":dev,"best_epoch":best[2],"base_rmse":rm(base),
 "degree3_oracle_rmse":rm(oracle),"cnn_rmse":rm(pred),"cnn_gain":rm(base)-rm(pred),
 "protocol":"target-independent 200-well sample; outer GroupKFold holdout untouched by early stopping; legal GR/typewell TVT+GR/V4/trajectory"}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2));torch.save({"state_dict":best[1],"coef_mean":cm,"coef_std":cs},OUT/"pilot_model.pt")
np.savez_compressed(OUT/"pilot_oof.npz",indices=foldix,prediction=pred[foldix],base=base[foldix],y=y[foldix])
print(json.dumps(summary,indent=2))
