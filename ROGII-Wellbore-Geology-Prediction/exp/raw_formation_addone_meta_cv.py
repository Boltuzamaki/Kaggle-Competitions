"""Add-one meta audit for the warm-normalized residual LGB OOF leg."""
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"exp/results/raw_formation_addone_meta";OUT.mkdir(parents=True,exist_ok=True)
GT=pd.read_parquet(ROOT/"exp/public_artifacts/pilkwang/oof/train_gt.parquet")
hr=pd.read_pickle(ROOT/"exp/results/harshini_cached_xgb/rows.pkl")
bywell={w:q.index.to_numpy() for w,q in GT.groupby("well_id",sort=False)}
hidx=np.empty(len(hr),int)
for w,q in hr.groupby("well",sort=False): hidx[q.index.to_numpy()]=bywell[w]
y=hr.target.to_numpy(float); wells=hr.well.to_numpy()
warm=1-np.exp(-np.maximum(hr.md_since.to_numpy(float),0)/85.)
legs={"har_physics":warm*hr.blend_d.to_numpy(float),
 "har_lgb":warm*np.load(ROOT/"exp/results/harshini_cached_xgb/lgb_fast_oof.npy"),
 "har_xgb":warm*np.load(ROOT/"exp/results/harshini_cached_xgb/xgb_oof.npy")}
sv=joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
legs["v4_lgb7"]=np.asarray(sv["lgb7"])[hidx]
pp=ROOT/"exp/public_artifacts/pilkwang/oof"
legs["pil_blend_oof_postprocessed"]=np.load(
    pp/"blend_oof_postprocessed.npy",mmap_mode="r")[hidx]
new=np.load(ROOT/"exp/results/raw_official_input_audit/oof.npz")[
    "formation_surface_raw"].astype(float)
Z=np.column_stack(list(legs.values())); Znew=np.c_[Z,new]
split=list(GroupKFold(5).split(Z,groups=wells))
def cv(A):
    p=np.zeros(len(y)); co=[]
    for tr,va in split:
        m=Ridge(alpha=100,positive=True,fit_intercept=False).fit(A[tr[::8]],y[tr[::8]])
        p[va]=m.predict(A[va]);co.append(m.coef_.tolist())
    return p,co
base,bc=cv(Z); add,ac=cv(Znew)
def rmse(p,ix=None):
    if ix is None:ix=np.arange(len(y))
    return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))
fg=[rmse(base,va)-rmse(add,va) for tr,va in split]
summary={"base_rmse":rmse(base),"add_rmse":rmse(add),"gain":rmse(base)-rmse(add),
 "fold_gains":fg,"fold_wins":sum(x>0 for x in fg),"leg_rmse":rmse(new),
 "columns":list(legs)+["raw_formation_surface"],"weights":ac,
 "acceptance":"positive pooled gain and 5/5 fold wins"}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
np.savez_compressed(OUT/"oof.npz",base=base,add=add,y=y,new_leg=new)
print(json.dumps(summary,indent=2))
