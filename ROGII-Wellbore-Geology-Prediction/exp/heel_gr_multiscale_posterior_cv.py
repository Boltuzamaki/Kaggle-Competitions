"""Legal rolling GR/typewell shift posteriors around Stack-V4 OOF paths."""
from pathlib import Path
import json,joblib
import numpy as np,pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"exp/results/heel_gr_multiscale";OUT.mkdir(parents=True,exist_ok=True)
def robust_affine(x,y):
 ok=np.isfinite(x)&np.isfinite(y);x,y=np.asarray(x)[ok],np.asarray(y)[ok]
 if len(x)>1200:
  take=np.linspace(0,len(x)-1,1200).astype(int);x,y=x[take],y[take]
 X=np.c_[x,np.ones(len(x))];coef=np.linalg.lstsq(X,y,rcond=None)[0]
 for _ in range(5):
  r=y-X@coef;sc=1.4826*np.median(np.abs(r-np.median(r)))+1e-3
  w=1/np.maximum(1,np.abs(r)/(2.5*sc));coef=np.linalg.lstsq(X*w[:,None],y*w,rcond=None)[0]
 return float(coef[0]),float(coef[1]),float(sc)
f=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl");oo=joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
y=f.target.to_numpy(float);base=np.asarray(oo["lgb7"],float); OFF=np.arange(-40,41,4,dtype=float)
windows=(101,301,801,0);temps=(.1,.2,.4)
means={(w,t):np.zeros(len(f),np.float32) for w in windows for t in temps}
for wi,(well,ix0) in enumerate(f.groupby("well",sort=False).indices.items()):
 ix=np.asarray(ix0);h=pd.read_csv(ROOT/f"data/train/{well}__horizontal_well.csv")
 tw=pd.read_csv(ROOT/f"data/train/{well}__typewell.csv").sort_values("TVT")
 tv=tw.TVT.to_numpy(float);tg=pd.to_numeric(tw.GR,errors="coerce").interpolate(limit_direction="both").to_numpy(float)
 hg=pd.to_numeric(h.GR,errors="coerce").interpolate(limit_direction="both").to_numpy(float)
 vis=h.TVT_input.notna().to_numpy();a,b,sc=robust_affine(np.interp(h.TVT_input[vis],tv,tg),hg[vis])
 # V4 IDs provide exact raw suffix rows.
 rr=np.array([int(x.rsplit("_",1)[1]) for x in f.id.iloc[ix]],int)
 path=f.last_known_tvt.to_numpy(float)[ix]+base[ix];obs=hg[rr]
 ref=a*np.vstack([np.interp(path+s,tv,tg) for s in OFF]).T+b
 z=(obs[:,None]-ref)/max(sc,5.);cost=np.log1p((z/2.)**2).astype(np.float32)
 for w in windows:
  C=np.broadcast_to(cost.mean(0),(len(ix),len(OFF))) if w==0 else uniform_filter1d(cost,size=min(w,len(ix)),axis=0,mode="nearest")
  for temp in temps:
   P=np.exp(np.clip(-(C-C.min(1,keepdims=True))/temp,-35,0));P/=P.sum(1,keepdims=True)
   means[w,temp][ix]=(P@OFF).astype(np.float32)
 if (wi+1)%100==0:print("wells",wi+1,flush=True)
global_corr=np.load(ROOT/"exp/results/heel_calibrated_gr_datum/oof.npz")["correction"].astype(float)
global_mean=global_corr/.25
folds=list(GroupKFold(5).split(base,groups=f.well))
def rm(p,ix=None):
 if ix is None:ix=np.arange(len(y))
 return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))
rows=[];pred={}
for (w,t),m in means.items():
 for mix in (.5,.75,1.):
  mm=mix*m+(1-mix)*global_mean
  for hedge in (.1,.15,.2,.25,.3):
   p=base+hedge*mm;fg=[rm(base,va)-rm(p,va) for tr,va in folds]
   key=f"w{w}_t{t}_m{mix}_h{hedge}"
   rows.append({"key":key,"window":w,"temperature":t,"local_mix":mix,"hedge":hedge,
    "rmse":rm(p),"gain":rm(base)-rm(p),"fold_wins":sum(x>0 for x in fg),"fold_gains":fg})
   pred[key]=p
grid=pd.DataFrame(rows).sort_values(["fold_wins","rmse"],ascending=[False,True])
best=grid.iloc[0].to_dict();bp=pred[best["key"]]
assert np.isfinite(bp).all()
grid.to_json(OUT/"grid.json",orient="records",indent=2)
np.savez_compressed(OUT/"oof.npz",prediction=bp,base=base,y=y,groups=f.well.to_numpy(),key=best["key"])
summary={"rows":len(f),"wells":int(f.well.nunique()),"base":rm(base),"best":best,
 "finite":bool(np.isfinite(bp).all()),"protocol":"legal raw GR/typewell TVT+GR; prefix-only affine; V4 grouped OOF path"}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
