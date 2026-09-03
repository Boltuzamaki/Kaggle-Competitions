"""Official pytabkit RealMLP experiment; official competition data only."""
from pathlib import Path
import gc, json, subprocess, sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytabkit==1.7.3"])

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from pytabkit import RealMLP_TD_Classifier

print("torch", torch.__version__, "CPU feasibility run")

TARGET, ID, SEED, FOLDS = "addicted_label", "id", 20260803, 3

def locate():
    for path in Path("/kaggle/input").rglob("train.csv"):
        try: columns = pd.read_csv(path, nrows=1).columns
        except Exception: continue
        if TARGET in columns and (path.parent / "test.csv").exists():
            return path, path.parent / "test.csv"
    raise FileNotFoundError("official competition tables not found")

def make_features(df):
    raw = df.drop(columns=[ID, TARGET], errors="ignore").copy()
    # RealMLP performs its own numerical preprocessing; these additions expose
    # structural missingness and screen-time accounting without target leakage.
    missing = raw.isna()
    for c in raw.select_dtypes(include=np.number).columns:
        raw[c + "__missing"] = missing[c].astype("float32")
    for c in raw.select_dtypes(exclude=np.number).columns:
        raw[c] = raw[c].astype("string").fillna("__MISSING__")
    raw["missing_count"] = missing.sum(axis=1).astype("float32")
    raw["is_complete"] = (~missing.any(axis=1)).astype("float32")
    raw["missing_pattern"] = sum(missing[c].astype("int32") * (2 ** i) for i, c in enumerate(missing.columns))
    parts = ["social_media_hours", "gaming_hours", "work_study_hours"]
    if all(c in raw for c in parts):
        raw["component_sum"] = raw[parts].sum(axis=1, min_count=1)
        if "daily_screen_time_hours" in raw:
            raw["screen_residual"] = raw["daily_screen_time_hours"] - raw["component_sum"]
    return raw

train_path, test_path = locate()
train, test = pd.read_csv(train_path), pd.read_csv(test_path)
X, XT, y = make_features(train), make_features(test), train[TARGET].to_numpy("int64")
folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y))

# Compact RealMLP-TD CPU feasibility run. Selection never touches leaderboard data.
configs = {"td_compact": dict(n_epochs=24, hidden_sizes=[256, 256, 256], p_drop=0.15, lr=0.04)}
all_oof, all_test, report = {}, {}, {}
for config_index, (name, params) in enumerate(configs.items()):
    oof = np.zeros(len(train), dtype="float64"); test_preds=[]; fold_scores=[]
    for fold, (fit_idx, val_idx) in enumerate(folds, 1):
        # PyTabKit 1.7.3 rejects continuous NaNs. Fit completion values only on
        # the outer training partition to preserve honest validation.
        X_fit=X.iloc[fit_idx].copy(); X_val=X.iloc[val_idx].copy(); X_test=XT.copy()
        numeric=X_fit.select_dtypes(include=np.number).columns
        medians=X_fit[numeric].median()
        X_fit[numeric]=X_fit[numeric].replace([np.inf,-np.inf],np.nan).fillna(medians).fillna(0)
        X_val[numeric]=X_val[numeric].replace([np.inf,-np.inf],np.nan).fillna(medians).fillna(0)
        X_test[numeric]=X_test[numeric].replace([np.inf,-np.inf],np.nan).fillna(medians).fillna(0)
        model = RealMLP_TD_Classifier(
            device="cpu", random_state=SEED + 100 * config_index + fold,
            n_cv=1, n_refit=0, val_fraction=0.12, val_metric_name="class_error",
            n_threads=4, batch_size=1024, predict_batch_size=8192, verbosity=1,
            use_early_stopping=True, **params,
        )
        model.fit(X_fit, y[fit_idx])
        vp = model.predict_proba(X_val)[:, 1]
        tp = model.predict_proba(X_test)[:, 1]
        oof[val_idx] = vp; test_preds.append(tp); fold_scores.append(roc_auc_score(y[val_idx], vp))
        del model,X_fit,X_val,X_test; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    all_oof[name] = oof; all_test[name] = np.mean(test_preds, axis=0)
    report[name] = {"params": params, "fold_auc": fold_scores,
                    "fold_mean": float(np.mean(fold_scores)), "fold_std": float(np.std(fold_scores)),
                    "oof_auc": float(roc_auc_score(y, oof))}
    pd.DataFrame({ID:test[ID], TARGET:all_test[name]}).to_csv(f"/kaggle/working/test_realmlp_{name}.csv", index=False)

fold_id=np.zeros(len(train),dtype="int8")
for fold,(_,val_idx) in enumerate(folds,1): fold_id[val_idx]=fold
for name in configs:
    pd.DataFrame({ID:train[ID],"fold":fold_id,"y":y,"pred":all_oof[name]}).to_csv(f"/kaggle/working/oof_realmlp_{name}.csv",index=False)
best=max(report,key=lambda n:report[n]["oof_auc"])
pd.DataFrame({ID:test[ID],TARGET:all_test[best]}).to_csv("/kaggle/working/test_realmlp_best.csv",index=False)
metrics={"implementation":"pytabkit 1.7.3 RealMLP_TD_Classifier","official_data_only":True,
         "seed":SEED,"folds":FOLDS,"torch":torch.__version__,"cuda":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         "device":"cpu","feasibility_protocol":"3 folds; expand to common 5 only if runtime allows",
         "configs":report,"selected_by_oof":best,"submission_created":False}
Path("/kaggle/working/metrics.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
