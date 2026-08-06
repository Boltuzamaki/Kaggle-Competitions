"""Combine only the common-schema families that won all five folds."""
from pathlib import Path
import json,numpy as np,pandas as pd,lightgbm as lgb
from sklearn.model_selection import GroupKFold
R=Path(__file__).resolve().parents[1];H=R/"exp/results/harshini_cached_xgb"
O=R/"exp/results/common_schema_stable_combo";O.mkdir(parents=True,exist_ok=True)
d=pd.read_pickle(H/"rows.pkl");fc=[c for c in d if c not in ("well","flat","target")]
X=d[fc].replace([np.inf,-np.inf],np.nan).to_numpy(np.float32);y=d.target.to_numpy(np.float32);g=d.well.to_numpy()
w=(1-np.exp(-np.maximum(d.md_since.to_numpy(float),0)/85.)).astype(np.float32)
p=(w*d.blend_d.to_numpy(float)).astype(np.float32);r=y-p
z=np.load(R/"exp/results/raw_official_input_audit/raw_families.npz")
F=np.c_[z["prefix_absolute"][:,:6],z["complete_trajectory_raw"][:,[0,1,2,3,4,5,6,7]]]
b=np.load(R/"exp/results/harshini_target_representation/oof.npz")["direct_physics_residual"]
def rm(q,i):return float(np.sqrt(np.mean((q[i]-y[i])**2)))
folds=list(GroupKFold(5).split(X,groups=g));o=np.zeros(len(y),np.float32)
for f,(tr,va) in enumerate(folds):
 fit=tr[tr%8==f%8]
 m=lgb.LGBMRegressor(objective="huber",n_estimators=420,learning_rate=.035,num_leaves=28,max_depth=8,
 min_child_samples=110,max_bin=127,colsample_bytree=.7,subsample=.85,subsample_freq=1,
 reg_alpha=2.,reg_lambda=16.,verbosity=-1,n_jobs=12,random_state=7700+f)
 m.fit(np.c_[X[fit],F[fit]],r[fit]);o[va]=p[va]+m.predict(np.c_[X[va],F[va]])
fg=[rm(b,va)-rm(o,va) for tr,va in folds]
s={"columns":14,"base_rmse":rm(b,np.arange(len(y))),"rmse":rm(o,np.arange(len(y))),
"gain":rm(b,np.arange(len(y)))-rm(o,np.arange(len(y))),"fold_gains":fg,"fold_wins":sum(x>0 for x in fg)}
(O/"summary.json").write_text(json.dumps(s,indent=2));np.savez_compressed(O/"oof.npz",y=y,prediction=o,base=b);print(s)
