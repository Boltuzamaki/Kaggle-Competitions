"""TabICL compact complete-well regret gate with exact expert Gram scoring."""
from pathlib import Path
import contextlib,io,json,runpy,sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tabicl import TabICLRegressor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"exp/results/tabicl_complete_well_gate";OUT.mkdir(parents=True,exist_ok=True)
FULL=bool(int(sys.argv[1])) if len(sys.argv)>1 else False
MODEL=ROOT/"exp/public_artifacts/tabicl/tabicl-regressor-v2-20260212.ckpt"
c=np.load(ROOT/"exp/results/complete_well_moe/well_table.npz",allow_pickle=True)
b=np.load(ROOT/"exp/results/prefix_backtest_moe/backtest_features.npz",allow_pickle=True)
X0,L,wn,nrow=c["X"],c["L"],c["wn"],c["nrow"]
if not np.array_equal(wn.astype(str),b["wells"].astype(str)):raise RuntimeError("BT well identity")
X=np.c_[X0,b["features"]].astype(np.float32)
names=["accepted_heel_meta","base_five_meta","heel_v4","v4_lgb7","har_physics",
 "har_lgb","har_xgb","pil_blend","v4_poly2","v4_poly3"]

# Exact expert paths/Gram matrices, identity-audited in the same common state.
with contextlib.redirect_stdout(io.StringIO()):
 s=runpy.run_path(str(ROOT/"exp/meta_all_honest_oof.py"))
common,y,wells,legs=s["common"],s["y"],s["wells"],s["legs"]
if not np.array_equal(wn.astype(str),pd.Series(wells).drop_duplicates().astype(str).to_numpy()):
 raise RuntimeError("meta well identity")
fm=np.load(ROOT/"exp/results/heel_calibrated_gr_datum/full_meta/oof.npz")
vf=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl").iloc[common].reset_index(drop=True)
hq=np.load(ROOT/"exp/results/heel_calibrated_gr_datum/oof.npz")
heel=legs["v4_lgb7"]+hq["correction"][common]
def proj(p,deg):
 out=p.copy()
 for _,ii in pd.Series(np.arange(len(y))).groupby(wells,sort=False):
  ix=ii.to_numpy();x=vf.d_md.to_numpy(float)[ix];x=2*(x-x.min())/max(np.ptp(x),1e-6)-1
  z=p[ix]+vf.d_z.to_numpy(float)[ix];out[ix]=np.polyval(np.polyfit(x,z,deg),x)-vf.d_z.to_numpy(float)[ix]
 return out
P=np.column_stack([fm["add"],fm["base"],heel,legs["v4_lgb7"],legs["har_physics"],
 legs["har_lgb"],legs["har_xgb"],legs["pil_blend_oof_postprocessed"],
 proj(legs["v4_lgb7"].copy(),2),proj(legs["v4_lgb7"].copy(),3)])
wix=[ii.to_numpy() for _,ii in pd.Series(np.arange(len(y))).groupby(wells,sort=False)]
G=np.empty((len(wn),10,10))
for i,ix in enumerate(wix):
 e=P[ix]-y[ix,None];G[i]=e.T@e/len(ix)

def score(W,ids):
 z=np.einsum("ni,nij,nj->n",W,G[ids],W)
 return float(np.sqrt(np.sum(nrow[ids]*z)/np.sum(nrow[ids])))

splits=list(GroupKFold(5).split(X,groups=wn));folds=range(5) if FULL else range(1)
pred=np.zeros_like(L);fid=np.full(len(wn),-1)
fold_rows=[]
for fold in folds:
 tr,va=splits[fold]
 for j in range(1,10):
  m=TabICLRegressor(n_estimators=4 if FULL else 2,batch_size=4,kv_cache=False,
   model_path=MODEL,allow_auto_download=False,device="cuda",use_amp=True,
   random_state=5200+fold*20+j,verbose=False)
  m.fit(X[tr],L[tr,j]-L[tr,0]);pred[va,j]=m.predict(X[va])
 pred[va,0]=0;fid[va]=fold
 baseW=np.zeros((len(va),10));baseW[:,0]=1
 hard=np.zeros_like(baseW);hard[np.arange(len(va)),np.argmin(pred[va],axis=1)]=1
 fold_rows.append({"fold":fold,"wells":len(va),"base":score(baseW,va),"hard":score(hard,va)})
 print(fold_rows[-1],flush=True)

ids=np.where(fid>=0)[0];baseW=np.zeros((len(ids),10));baseW[:,0]=1
grid=[]
for kind in ("hard","soft"):
 for temp in (.35,.5,1.,2.):
  W=np.zeros_like(baseW);q=pred[ids]
  if kind=="hard":W[np.arange(len(ids)),np.argmin(q,axis=1)]=1
  else:
   W=np.exp(np.clip(-(q-q.min(1,keepdims=True))/temp,-20,0));W/=W.sum(1,keepdims=True)
  for blend in (.1,.2,.3,.5,.7,1):
   Q=blend*W+(1-blend)*baseW
   grid.append({"kind":kind,"temperature":temp,"blend":blend,"rmse":score(Q,ids)})
summary={"full":FULL,"folds_completed":len(fold_rows),"wells":len(ids),"features":X.shape[1],
 "accepted_base":score(baseW,ids),"best":min(grid,key=lambda z:z["rmse"]),
 "folds":fold_rows,"checkpoint":str(MODEL),
 "protocol":"strict outer whole-well GKF; independent TabICL regret targets; exact Gram scoring"}
(OUT/("full_summary.json" if FULL else "smoke_summary.json")).write_text(json.dumps(summary,indent=2))
np.savez_compressed(OUT/("full_oof.npz" if FULL else "smoke_oof.npz"),predicted_regret=pred,fold=fid)
print(json.dumps(summary,indent=2))
