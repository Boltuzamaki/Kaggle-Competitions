"""Original shallow native-NaN LightGBM seed bag, official S6E8 data only."""
from pathlib import Path
import gc, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET, ID, SEED, NFOLD = "addicted_label", "id", 20260803, 5
MODEL_SEEDS = (20260803, 20260881)
CONFIGS = {
    "d4_l15": dict(num_leaves=15, max_depth=4, min_child_samples=220, reg_lambda=8.),
    "d6_l31": dict(num_leaves=31, max_depth=6, min_child_samples=320, reg_lambda=12.),
}

def locate():
    for root in (Path("/kaggle/input"), Path(".")):
        for p in root.rglob("train.csv"):
            try: cols = pd.read_csv(p, nrows=2).columns
            except Exception: continue
            if TARGET in cols and (p.parent/"test.csv").exists():
                return p, p.parent/"test.csv"
    raise FileNotFoundError("official S6E8 train/test not found")

def features(df):
    x = df.drop(columns=[ID, TARGET], errors="ignore").copy()
    raw = list(x.columns)
    miss = x[raw].isna()
    x["missing_count"] = miss.sum(1).astype("int8")
    x["complete_row"] = miss.sum(1).eq(0).astype("int8")
    for c in raw: x[c+"__missing"] = miss[c].astype("int8")
    x["leisure_hours"] = x.social_media_hours + x.gaming_hours
    x["accounted_hours"] = x.leisure_hours + x.work_study_hours
    x["unaccounted_screen"] = x.daily_screen_time_hours - x.accounted_hours
    x["weekend_gap"] = x.weekend_screen_time - x.daily_screen_time_hours
    x["opens_per_screen_hour"] = x.app_opens_per_day / (x.daily_screen_time_hours + .5)
    x["notifications_per_open"] = x.notifications_per_day / (x.app_opens_per_day + 3.)
    for name, col in (("social", "social_media_hours"), ("gaming", "gaming_hours"), ("work", "work_study_hours")):
        x[name+"_screen_share"] = x[col] / (x.daily_screen_time_hours + .5)
    # Native categorical splitting, with missing category explicit. No labels enter this view.
    for c in x.select_dtypes(exclude="number"):
        x[c] = x[c].astype("string").fillna("__NA__").astype("category")
    return x.replace([np.inf, -np.inf], np.nan)

trp, tep = locate(); train, test = pd.read_csv(trp), pd.read_csv(tep)
y = train[TARGET].to_numpy("int8"); x, xt = features(train), features(test)
outer = StratifiedKFold(NFOLD, shuffle=True, random_state=SEED)
oofs={k:np.zeros(len(train)) for k in CONFIGS}; test_folds={k:[] for k in CONFIGS}; fold_id=np.zeros(len(train),dtype="int8"); rows=[]
for fold,(fi,vi) in enumerate(outer.split(x,y),1):
    for config_name,config in CONFIGS.items():
        seed_val=[]; seed_test=[]; its=[]
        for model_seed in MODEL_SEEDS:
            model=lgb.LGBMClassifier(objective="binary",n_estimators=5000,learning_rate=.018,max_bin=127,
              subsample=.84,subsample_freq=1,colsample_bytree=.82,reg_alpha=.35,
              random_state=model_seed+fold,n_jobs=-1,verbosity=-1,**config)
            model.fit(x.iloc[fi],y[fi],eval_set=[(x.iloc[vi],y[vi])],eval_metric="auc",
              callbacks=[lgb.early_stopping(220,verbose=False)])
            seed_val.append(model.predict_proba(x.iloc[vi])[:,1]); seed_test.append(model.predict_proba(xt)[:,1]); its.append(int(model.best_iteration_))
            del model; gc.collect()
        pred=np.mean(seed_val,axis=0); oofs[config_name][vi]=pred; test_folds[config_name].append(np.mean(seed_test,axis=0))
        rows.append({"fold":fold,"config":config_name,"auc":float(roc_auc_score(y[vi],pred)),"best_iterations":its}); print(rows[-1],flush=True)
    fold_id[vi]=fold
scores={k:float(roc_auc_score(y,p)) for k,p in oofs.items()}
fold_stds={k:float(np.std([r["auc"] for r in rows if r["config"]==k],ddof=1)) for k in CONFIGS}
# A tiny stability penalty prevents selecting a noisy configuration on negligible mean gain.
selection_value={k:scores[k]-.10*fold_stds[k] for k in CONFIGS}; selected=max(selection_value,key=selection_value.get)
test_preds={k:np.mean(v,axis=0) for k,v in test_folds.items()}
out=pd.DataFrame({ID:train[ID],"fold":fold_id,"y":y,**{"pred_"+k:v for k,v in oofs.items()}}); out.to_csv("/kaggle/working/oof_raw_lgb_bag.csv",index=False)
pd.DataFrame({ID:test[ID],**{"pred_"+k:v for k,v in test_preds.items()}}).to_csv("/kaggle/working/test_raw_lgb_bag.csv",index=False)
metrics={"provenance":"independently trained; official competition data only","folds":NFOLD,"split_seed":SEED,"model_seeds":MODEL_SEEDS,"configs":CONFIGS,"view":"raw native NaN plus missingness/composition; no target or frequency encoding","fold_metrics":rows,"oof_auc":scores,"fold_auc_std":fold_stds,"selection_value_auc_minus_0.1std":selection_value,"selected":selected,"estimated_lb_from_prior_original_lgb_offset":{"point":scores[selected]+0.00093,"warning":"rough calibration from one prior model; uncertainty is at least several 1e-4 and model-dependent"}}
Path("/kaggle/working/metrics_raw_lgb_bag.json").write_text(json.dumps(metrics,indent=2)+"\n")
print(json.dumps(metrics,indent=2),flush=True)
