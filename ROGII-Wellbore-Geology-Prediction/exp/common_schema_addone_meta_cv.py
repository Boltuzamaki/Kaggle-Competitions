"""Deployable five-leg meta audit of common-schema raw-input legs."""
from pathlib import Path
import json,joblib,numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
R=Path(__file__).resolve().parents[1];O=R/"exp/results/common_schema_addone_meta";O.mkdir(parents=True,exist_ok=True)
gt=pd.read_parquet(R/"exp/public_artifacts/pilkwang/oof/train_gt.parquet")
d=pd.read_pickle(R/"exp/results/harshini_cached_xgb/rows.pkl"); bw={w:q.index.to_numpy() for w,q in gt.groupby("well_id",sort=False)}
ix=np.empty(len(d),int)
for w,q in d.groupby("well",sort=False):ix[q.index.to_numpy()]=bw[w]
y=d.target.to_numpy(float);g=d.well.to_numpy();warm=1-np.exp(-np.maximum(d.md_since.to_numpy(float),0)/85.)
H=R/"exp/results/harshini_cached_xgb";sv=joblib.load(R/"r_v4b/stack_v4_oofs.joblib")["oofs"];P=R/"exp/public_artifacts/pilkwang/oof"
Z=np.column_stack([warm*d.blend_d,warm*np.load(H/"lgb_fast_oof.npy"),warm*np.load(H/"xgb_oof.npy"),
 np.asarray(sv["lgb7"])[ix],np.load(P/"blend_oof_postprocessed.npy",mmap_mode="r")[ix]])
q=np.load(R/"exp/results/common_schema_raw_input/oof.npz")
cands={"flat_only":q["prefix_flat_only"],"anchor6":q["prefix_anchor6"],
       "reference_position":q["prefix_reference_position"]}
folds=list(GroupKFold(5).split(Z,groups=g))
def cv(A):
 p=np.zeros(len(y));co=[]
 for tr,va in folds:
  m=Ridge(alpha=100,positive=True,fit_intercept=False).fit(A[tr[::8]],y[tr[::8]])
  p[va]=m.predict(A[va]);co.append(m.coef_.tolist())
 return p,co
def rm(p,i=None):
 if i is None:i=np.arange(len(y))
 return float(np.sqrt(np.mean((p[i]-y[i])**2)))
b,_=cv(Z);out={}
sets={k:[k] for k in cands};sets["flat_plus_anchor"]=["flat_only","anchor6"];sets["all_three"]=list(cands)
for name,cs in sets.items():
 A=np.c_[Z,*[cands[x] for x in cs]];p,co=cv(A);fg=[rm(b,va)-rm(p,va) for tr,va in folds]
 out[name]={"rmse":rm(p),"gain":rm(b)-rm(p),"fold_gains":fg,"fold_wins":sum(x>0 for x in fg),
            "added":cs,"weights":co}
s={"base_rmse":rm(b),"variants":out,"acceptance":"positive material gain and 5/5 folds"}
(O/"summary.json").write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
