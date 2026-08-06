"""Fast complementary LightGBM OOF on cached Harshini features."""
from pathlib import Path
import json, os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"exp/results/harshini_cached_xgb"
d=pd.read_pickle(OUT/"rows.pkl")
features=[c for c in d if c not in ("well","flat","target")]
X=d[features].to_numpy(np.float32); y=d.target.to_numpy(np.float32); g=d.well.to_numpy()
sub=np.arange(len(d))%8==0
oof=np.zeros(len(d),np.float32)
for fold,(tr,va) in enumerate(GroupKFold(5).split(X[sub],y[sub],g[sub])):
    vw=set(g[sub][va]); vm=np.fromiter((w in vw for w in g),bool,len(g))
    m=lgb.LGBMRegressor(
        objective="regression_l2",n_estimators=1000,learning_rate=.035,
        num_leaves=127,max_depth=10,min_child_samples=100,
        subsample=.8,subsample_freq=1,colsample_bytree=.65,
        reg_alpha=.2,reg_lambda=8,max_bin=127,verbosity=-1,
        random_state=520+fold,n_jobs=max(1,(os.cpu_count() or 8)//2))
    m.fit(X[sub][tr],y[sub][tr]); oof[vm]=m.predict(X[vm])
    print("fold",fold,"done",flush=True)
warm=1-np.exp(-np.maximum(d.md_since.to_numpy(),0)/85.)
blend=d.blend_d.to_numpy(float); xo=np.load(OUT/"xgb_oof.npy")
def rmse(p): return float(np.sqrt(np.mean((warm*p-y)**2)))
legs=np.c_[blend,oof,xo]
r=Ridge(alpha=1,positive=True,fit_intercept=False).fit(legs,y)
summary={"lgb_rmse":rmse(oof),"blend_rmse":rmse(blend),"xgb_rmse":rmse(xo),
         "ridge_weights":r.coef_.tolist(),"ridge_rmse":rmse(r.predict(legs))}
print(json.dumps(summary,indent=2),flush=True)
np.save(OUT/"lgb_fast_oof.npy",oof)
joblib.dump({"features":features,"ridge":r},OUT/"lgb_fast_meta.joblib")
(OUT/"lgb_fast_summary.json").write_text(json.dumps(summary,indent=2))
