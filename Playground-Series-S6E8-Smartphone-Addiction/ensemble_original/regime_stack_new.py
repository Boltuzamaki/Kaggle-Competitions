#!/usr/bin/env python3
"""Cross-fitted missingness-regime stack over the strongest original models."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]; TARGET="addicted_label"; SEED=20260803
spec=importlib.util.spec_from_file_location("audit",ROOT/"ensemble_original/audit_and_blend.py")
ab=importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
train=pd.read_csv(ROOT/"train.csv"); y=train[TARGET].to_numpy(int); ids=train.id

def read(path,col): return pd.read_csv(ROOT/path).set_index("id").reindex(ids)[col].to_numpy(float)
x=np.column_stack([
 read("artifacts/xgb_te_5fold/oof_nested_te_xgb.csv","pred"),
 read("xgb_nested_hpo/output/oof_nested_hpo_xgb.csv","pred"),
 read("artifacts/nested_te_catboost/oof_nested_te_catboost.csv","pred"),
 read("gpu_lookup_original/output/oof_lookup_transformer.csv","pred"),
])
names=["xgb_inner5","xgb_hpo_inner4","catboost_nested_te","lookup_transformer"]
missing=train.drop(columns=["id",TARGET]).isna().sum(axis=1).to_numpy()
regime=np.where(missing==0,0,np.where(missing<=3,1,2))
cv=StratifiedKFold(5,shuffle=True,random_state=SEED+91)
global_pred=np.zeros(len(y)); regime_pred=np.zeros(len(y)); rows=[]
for fold,(fit,val) in enumerate(cv.split(x,y),1):
 w=ab.fit_weights(x[fit],y[fit],"logit"); global_pred[val]=ab.combine(x[val],w,"logit")
 for r in range(3):
  fi=fit[regime[fit]==r]; vi=val[regime[val]==r]
  wr=ab.fit_weights(x[fi],y[fi],"logit"); regime_pred[vi]=ab.combine(x[vi],wr,"logit")
  rows.append({"meta_fold":fold,"regime":r,"fit_rows":len(fi),"valid_rows":len(vi),**dict(zip(names,wr))})
report={"global_crossfit_auc":roc_auc_score(y,global_pred),"regime_crossfit_auc":roc_auc_score(y,regime_pred)}
report["gain"]=report["regime_crossfit_auc"]-report["global_crossfit_auc"]
out=ROOT/"ensemble_original/reports"; pd.DataFrame(rows).to_csv(out/"new_model_regime_weights.csv",index=False)
pd.DataFrame([report]).to_csv(out/"new_model_regime_auc.csv",index=False)
print(report); print(pd.DataFrame(rows).groupby("regime")[names].mean())
