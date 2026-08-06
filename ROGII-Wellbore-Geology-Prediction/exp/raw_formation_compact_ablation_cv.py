"""Compact ablations of the winning 23-column raw formation family."""
from pathlib import Path
import json, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1]; H=ROOT/"exp/results/harshini_cached_xgb"
OUT=ROOT/"exp/results/raw_formation_compact_ablation";OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_pickle(H/"rows.pkl"); feats=[c for c in d if c not in ("well","flat","target")]
X=d[feats].replace([np.inf,-np.inf],np.nan).to_numpy(np.float32)
y=d.target.to_numpy(np.float32);g=d.well.to_numpy()
warm=(1-np.exp(-np.maximum(d.md_since.to_numpy(float),0)/85.)).astype(np.float32)
phy=(warm*d.blend_d.to_numpy(float)).astype(np.float32); r=y-phy
F=np.load(ROOT/"exp/results/raw_official_input_audit/raw_families.npz")["formation_surface_raw"]
base=np.load(ROOT/"exp/results/harshini_target_representation/oof.npz")["direct_physics_residual"]
variants={"raw6":F[:,:6],"raw_plus_relative12":F[:,:12],"full23":F}
def model(seed):return lgb.LGBMRegressor(objective="huber",n_estimators=420,
 learning_rate=.035,num_leaves=28,max_depth=8,min_child_samples=110,max_bin=127,
 colsample_bytree=.7,subsample=.85,subsample_freq=1,reg_alpha=2.,reg_lambda=16.,
 verbosity=-1,n_jobs=12,random_state=seed)
def rmse(p,ix):return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))
folds=list(GroupKFold(5).split(X,groups=g)); out={};preds={}
for vi,(name,V) in enumerate(variants.items()):
 if name=="full23":
  p=np.load(ROOT/"exp/results/raw_official_input_audit/oof.npz")["formation_surface_raw"]
 else:
  p=np.zeros(len(y),np.float32)
  for f,(tr,va) in enumerate(folds):
   fit=tr[tr%8==f%8];m=model(1000*vi+f);m.fit(np.c_[X[fit],V[fit]],r[fit])
   p[va]=phy[va]+m.predict(np.c_[X[va],V[va]])
 fg=[rmse(base,va)-rmse(p,va) for tr,va in folds]
 out[name]={"columns":V.shape[1],"rmse":rmse(p,np.arange(len(y))),
            "gain_vs_base":rmse(base,np.arange(len(y)))-rmse(p,np.arange(len(y))),
            "fold_gains":fg,"fold_wins":sum(x>0 for x in fg)}
 preds[name]=p;print(name,out[name],flush=True)
(OUT/"summary.json").write_text(json.dumps(out,indent=2))
np.savez_compressed(OUT/"oof.npz",y=y,base=base,**preds)
