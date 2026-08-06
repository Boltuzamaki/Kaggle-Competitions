"""Only raw inputs verified present in both official train and test schemas."""
from pathlib import Path
import json,numpy as np,pandas as pd,lightgbm as lgb
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];H=ROOT/"exp/results/harshini_cached_xgb"
OUT=ROOT/"exp/results/common_schema_raw_input";OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_pickle(H/"rows.pkl");cols=[c for c in d if c not in ("well","flat","target")]
X=d[cols].replace([np.inf,-np.inf],np.nan).to_numpy(np.float32)
y=d.target.to_numpy(np.float32);g=d.well.to_numpy()
warm=(1-np.exp(-np.maximum(d.md_since.to_numpy(float),0)/85.)).astype(np.float32)
phy=(warm*d.blend_d.to_numpy(float)).astype(np.float32);res=y-phy
z=np.load(ROOT/"exp/results/raw_official_input_audit/raw_families.npz")
P=z["prefix_absolute"];T=z["complete_trajectory_raw"];W=z["paired_typewell_raw"][:,:10]
# All fields below derive solely from common horizontal MD/X/Y/Z/TVT_input or
# typewell TVT/GR. Geology and train-only formation columns are excluded.
variants={
 "prefix_flat_only":P[:,[0]],
 "prefix_anchor6":P[:,:6],
 "prefix_reference_position":P[:,[0,5,6,7]],
 "trajectory_endpoint_geometry":T[:,[0,1,2,3,4,5,6,7]],
 "typewell_tvt_gr_global":W,
 "all_common_raw":np.c_[P,T,W],
}
base=np.load(ROOT/"exp/results/harshini_target_representation/oof.npz")["direct_physics_residual"]
def mod(seed):return lgb.LGBMRegressor(objective="huber",n_estimators=420,learning_rate=.035,
 num_leaves=28,max_depth=8,min_child_samples=110,max_bin=127,colsample_bytree=.7,
 subsample=.85,subsample_freq=1,reg_alpha=2.,reg_lambda=16.,verbosity=-1,n_jobs=12,random_state=seed)
def rmse(p,ix):return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))
folds=list(GroupKFold(5).split(X,groups=g));out={};pred={}
for vi,(name,F) in enumerate(variants.items()):
 p=np.zeros(len(y),np.float32)
 for f,(tr,va) in enumerate(folds):
  fit=tr[tr%8==f%8];m=mod(5000+100*vi+f);m.fit(np.c_[X[fit],F[fit]],res[fit])
  p[va]=phy[va]+m.predict(np.c_[X[va],F[va]])
 fg=[rmse(base,va)-rmse(p,va) for tr,va in folds]
 out[name]={"columns":F.shape[1],"rmse":rmse(p,np.arange(len(y))),
  "gain":rmse(base,np.arange(len(y)))-rmse(p,np.arange(len(y))),
  "fold_gains":fg,"fold_wins":sum(x>0 for x in fg)}
 pred[name]=p;print(name,out[name],flush=True)
summary={"schema_horizontal":["MD","X","Y","Z","GR","TVT_input"],
 "schema_typewell":["TVT","GR"],"explicitly_excluded":["TVT target","formation columns","Geology"],
 "base_rmse":rmse(base,np.arange(len(y))),"variants":out}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2))
np.savez_compressed(OUT/"oof.npz",y=y,base=base,**pred)
