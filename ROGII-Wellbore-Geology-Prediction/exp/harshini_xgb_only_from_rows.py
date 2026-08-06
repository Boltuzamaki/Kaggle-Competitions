"""GPU XGBoost OOF on the cached Harshini row matrix."""
from pathlib import Path
import json, os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"exp/results/harshini_cached_xgb"
d=pd.read_pickle(OUT/"rows.pkl")
features=[c for c in d.columns if c not in ("well","flat","target")]
X=d[features].to_numpy(np.float32)
y=d.target.to_numpy(np.float32)
g=d.well.to_numpy()
sub=np.arange(len(d))%8==0
oof=np.zeros(len(d),np.float32)

for fold,(tr,va) in enumerate(GroupKFold(5).split(X[sub],y[sub],g[sub])):
    vw=set(g[sub][va]); vm=np.fromiter((w in vw for w in g),bool,len(g))
    m=xgb.XGBRegressor(
        objective="reg:squarederror",n_estimators=1400,learning_rate=.025,
        max_depth=8,min_child_weight=25,subsample=.8,colsample_bytree=.7,
        reg_alpha=.25,reg_lambda=12,tree_method="hist",device="cuda",
        max_bin=256,random_state=256+fold,n_jobs=max(1,(os.cpu_count() or 8)//2))
    m.fit(X[sub][tr],y[sub][tr])
    oof[vm]=m.predict(X[vm])
    print("fold",fold,"done",flush=True)

warm=1-np.exp(-np.maximum(d.md_since.to_numpy(),0)/85.)
def rmse(p): return float(np.sqrt(np.mean((warm*p-y)**2)))
blend=d.blend_d.to_numpy(float)
ridge=Ridge(alpha=1,positive=True,fit_intercept=False).fit(
    np.c_[blend,oof],y)
grid=[]
for a in np.linspace(0,1,21):
    grid.append({"xgb_weight":float(a),"rmse":rmse((1-a)*blend+a*oof)})
grid=pd.DataFrame(grid).sort_values("rmse")
summary={"rows":len(d),"wells":int(d.well.nunique()),"features":len(features),
         "blend_rmse":rmse(blend),"xgb_rmse":rmse(oof),
         "best_grid":grid.iloc[0].to_dict(),
         "ridge_weights":ridge.coef_.tolist(),
         "ridge_rmse":rmse(ridge.predict(np.c_[blend,oof]))}
print(json.dumps(summary,indent=2),flush=True)
np.save(OUT/"xgb_oof.npy",oof)
joblib.dump({"features":features,"ridge":ridge},OUT/"xgb_meta.joblib")
grid.to_csv(OUT/"xgb_blend_grid.csv",index=False)
(OUT/"xgb_summary.json").write_text(json.dumps(summary,indent=2))
