"""Full-data counterparts of the honest Harshini fast LGB/XGB OOF legs."""
from pathlib import Path
import hashlib,json,os
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"exp/results/harshini_fast_model_package"
OUT.mkdir(parents=True,exist_ok=True)
d=pd.read_pickle(ROOT/"exp/results/harshini_cached_xgb/rows.pkl")
features=[c for c in d if c not in ("well","flat","target")]
X=d[features].to_numpy(np.float32); y=d.target.to_numpy(np.float32)
sub=np.arange(len(d))%8==0
lp=dict(objective="regression_l2",n_estimators=1000,learning_rate=.035,
        num_leaves=127,max_depth=10,min_child_samples=100,subsample=.8,
        subsample_freq=1,colsample_bytree=.65,reg_alpha=.2,reg_lambda=8,
        max_bin=127,verbosity=-1,random_state=520,n_jobs=max(1,os.cpu_count() or 8))
lm=lgb.LGBMRegressor(**lp).fit(X[sub],y[sub])
lm.booster_.save_model(str(OUT/"harshini_fast_lgb.txt"))
xp=dict(objective="reg:squarederror",n_estimators=1400,learning_rate=.025,
        max_depth=8,min_child_weight=25,subsample=.8,colsample_bytree=.7,
        reg_alpha=.25,reg_lambda=12,tree_method="hist",device="cuda",
        max_bin=256,random_state=256,n_jobs=max(1,(os.cpu_count() or 8)//2))
xm=xgb.XGBRegressor(**xp).fit(X[sub],y[sub])
xm.save_model(OUT/"harshini_fast_xgb.json")
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest={"schema":"rogii_harshini_fast_models_v1","features":features,
          "feature_count":len(features),"training_rows":int(sub.sum()),
          "source_rows":len(d),"stride":8,"target":"delta from last visible TVT",
          "lgb_params":lp,"xgb_params":xp,"prediction_artifacts_used":False,
          "files":{"lgb_sha256":sha(OUT/"harshini_fast_lgb.txt"),
                   "xgb_sha256":sha(OUT/"harshini_fast_xgb.json")}}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
print(json.dumps({"rows":int(sub.sum()),"features":len(features)},indent=2))
