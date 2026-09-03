"""Original depth-constrained native-NaN XGBoost seed bag, official data only."""
from pathlib import Path
import gc, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

TARGET, ID, SEED, NFOLD = "addicted_label", "id", 20260803, 5
MODEL_SEEDS = (20260803, 20260917)

def locate():
    for root in (Path("/kaggle/input"), Path(".")):
        for p in root.rglob("train.csv"):
            try: cols=pd.read_csv(p,nrows=2).columns
            except Exception: continue
            if TARGET in cols and (p.parent/"test.csv").exists(): return p,p.parent/"test.csv"
    raise FileNotFoundError("official S6E8 train/test not found")

def features(df):
    x=df.drop(columns=[ID,TARGET],errors="ignore").copy(); raw=list(x.columns); miss=x[raw].isna()
    x["missing_count"]=miss.sum(1).astype("int8"); x["complete_row"]=miss.sum(1).eq(0).astype("int8")
    for c in raw: x[c+"__missing"]=miss[c].astype("int8")
    x["leisure_hours"]=x.social_media_hours+x.gaming_hours
    x["accounted_hours"]=x.leisure_hours+x.work_study_hours
    x["unaccounted_screen"]=x.daily_screen_time_hours-x.accounted_hours
    x["weekend_gap"]=x.weekend_screen_time-x.daily_screen_time_hours
    x["notifications_per_open"]=x.notifications_per_day/(x.app_opens_per_day+3.)
    x["opens_per_screen_hour"]=x.app_opens_per_day/(x.daily_screen_time_hours+.5)
    for name,col in (("social","social_media_hours"),("gaming","gaming_hours"),("work","work_study_hours")):
        x[name+"_screen_share"]=x[col]/(x.daily_screen_time_hours+.5)
    # Deterministic label-free coding. Numeric NaNs remain NaN for native missing branches.
    for c in x.select_dtypes(exclude="number"):
        x[c]=pd.factorize(x[c].astype("string").fillna("__NA__"),sort=True)[0]
    return x.replace([np.inf,-np.inf],np.nan).astype("float32")

trp,tep=locate(); train,test=pd.read_csv(trp),pd.read_csv(tep); y=train[TARGET].to_numpy("int8"); x,xt=features(train),features(test)
outer=StratifiedKFold(NFOLD,shuffle=True,random_state=SEED)
oof=np.zeros(len(train)); fold_id=np.zeros(len(train),dtype="int8"); test_folds=[]; rows=[]; used_devices=[]
for fold,(fi,vi) in enumerate(outer.split(x,y),1):
    vals=[]; tests=[]; its=[]
    for model_seed in MODEL_SEEDS:
        common=dict(n_estimators=4200,learning_rate=.018,max_depth=5,min_child_weight=18,
          subsample=.84,colsample_bytree=.82,reg_alpha=.35,reg_lambda=9.,gamma=.015,
          objective="binary:logistic",eval_metric="auc",random_state=model_seed+fold,n_jobs=-1,
          tree_method="hist",early_stopping_rounds=220,max_bin=192)
        try:
            model=XGBClassifier(device="cuda",**common); model.fit(x.iloc[fi],y[fi],eval_set=[(x.iloc[vi],y[vi])],verbose=False); device="cuda"
        except Exception as exc:
            print("GPU fallback:",repr(exc),flush=True); model=XGBClassifier(device="cpu",**common); model.fit(x.iloc[fi],y[fi],eval_set=[(x.iloc[vi],y[vi])],verbose=False); device="cpu"
        vals.append(model.predict_proba(x.iloc[vi])[:,1]); tests.append(model.predict_proba(xt)[:,1]); its.append(int(model.best_iteration+1)); used_devices.append(device)
        del model; gc.collect()
    pred=np.mean(vals,axis=0); oof[vi]=pred; test_folds.append(np.mean(tests,axis=0)); fold_id[vi]=fold
    rows.append({"fold":fold,"auc":float(roc_auc_score(y[vi],pred)),"best_iterations":its}); print(rows[-1],flush=True)
test_pred=np.mean(test_folds,axis=0); score=float(roc_auc_score(y,oof))
pd.DataFrame({ID:train[ID],"fold":fold_id,"y":y,"pred":oof}).to_csv("/kaggle/working/oof_raw_xgb_bag.csv",index=False)
pd.DataFrame({ID:test[ID],"pred":test_pred}).to_csv("/kaggle/working/test_raw_xgb_bag.csv",index=False)
metrics={"provenance":"independently trained; official competition data only","folds":NFOLD,"split_seed":SEED,"model_seeds":MODEL_SEEDS,"view":"raw native NaN plus missingness/composition; no target or frequency encoding","devices":used_devices,"fold_metrics":rows,"oof_auc":score}
Path("/kaggle/working/metrics_raw_xgb_bag.json").write_text(json.dumps(metrics,indent=2)+"\n"); print(json.dumps(metrics,indent=2),flush=True)
