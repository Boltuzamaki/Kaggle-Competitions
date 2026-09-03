"""Nested-tuned XGBoost DART on official data, producing honest five-fold OOF.

The scored outer fold is never used for configuration or boosting-round selection.
Each outer-training partition runs a bounded inner holdout screen, after which
the winning configuration is refit on the complete outer-training partition.
"""
from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from xgboost import XGBClassifier

TARGET, ID = "addicted_label", "id"
SEED, FOLDS = 20260804, 5
MAX_ROUNDS, EARLY_STOPPING = 1800, 100

# Deliberately small search space: DART behavior is the experiment, not a broad
# depth search duplicating the existing gbtree HPO streams.
CONFIGS = {
    "dart_conservative": dict(max_depth=5, min_child_weight=18, learning_rate=.025,
        subsample=.88, colsample_bytree=.86, reg_alpha=.15, reg_lambda=9.,
        rate_drop=.035, skip_drop=.15),
    "dart_regularized": dict(max_depth=6, min_child_weight=28, learning_rate=.022,
        subsample=.84, colsample_bytree=.80, reg_alpha=.45, reg_lambda=13.,
        rate_drop=.07, skip_drop=.10),
    "dart_shallow": dict(max_depth=4, min_child_weight=12, learning_rate=.030,
        subsample=.90, colsample_bytree=.92, reg_alpha=.08, reg_lambda=7.,
        rate_drop=.05, skip_drop=.20),
}


def locate():
    for root in (Path("/kaggle/input"), Path(".")):
        for p in root.rglob("train.csv"):
            try:
                cols = pd.read_csv(p, nrows=1).columns
            except Exception:
                continue
            if TARGET in cols and (p.parent / "test.csv").exists():
                return p, p.parent / "test.csv"
    raise FileNotFoundError("official competition train/test tables not found")


def features(df):
    """Target-free raw, missingness, ratio, and one-hot features."""
    raw = df.drop(columns=[ID, TARGET], errors="ignore").copy()
    numeric = raw.select_dtypes(include="number").columns.tolist()
    out = raw[numeric].astype("float32").copy()
    miss = raw.isna()
    out["missing_count"] = miss.sum(axis=1).astype("int8")
    for c in raw.columns:
        out["missing__" + c] = miss[c].astype("int8")

    component = raw[["social_media_hours", "gaming_hours", "work_study_hours"]].sum(axis=1, min_count=3)
    out["component_sum"] = component
    out["unaccounted_screen"] = raw.daily_screen_time_hours - component
    out["weekend_gap"] = raw.weekend_screen_time - raw.daily_screen_time_hours
    out["notification_open_ratio"] = raw.notifications_per_day / (raw.app_opens_per_day + 2.)
    out["opens_screen_ratio"] = raw.app_opens_per_day / (raw.daily_screen_time_hours + .5)
    out["screen_sleep_ratio"] = raw.daily_screen_time_hours / (raw.sleep_hours + .5)
    out["social_share"] = raw.social_media_hours / (raw.daily_screen_time_hours + .5)
    out["gaming_share"] = raw.gaming_hours / (raw.daily_screen_time_hours + .5)
    out["work_share"] = raw.work_study_hours / (raw.daily_screen_time_hours + .5)
    cats = raw.select_dtypes(exclude="number").columns
    if len(cats):
        # Fixed, target-free category vocabulary. Calling this once on official
        # train+test is transductive only in feature schema, never in labels.
        dummies = pd.get_dummies(raw[cats].astype("string").fillna("__MISSING__"),
                                 prefix=cats, dtype="int8")
        out = pd.concat([out, dummies], axis=1)
    return out.replace([np.inf, -np.inf], np.nan).astype("float32")


def model(config, seed, n_estimators=MAX_ROUNDS, early_stopping_rounds=None):
    return XGBClassifier(
        booster="dart", objective="binary:logistic", eval_metric="auc",
        tree_method="hist", device="cuda", n_jobs=-1, random_state=seed,
        n_estimators=n_estimators, max_bin=192, normalize_type="tree",
        sample_type="uniform", early_stopping_rounds=early_stopping_rounds,
        **config,
    )


train_path, test_path = locate()
train, test = pd.read_csv(train_path), pd.read_csv(test_path)
y = train[TARGET].to_numpy("int8")
# Build a common target-free schema, then separate the rows again.
both = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
all_x = features(both)
X, XT = all_x.iloc[:len(train)].reset_index(drop=True), all_x.iloc[len(train):].reset_index(drop=True)

outer = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train), dtype="float64")
test_predictions, fold_ids, records = [], np.zeros(len(train), dtype="int8"), []

for fold, (outer_train, outer_valid) in enumerate(outer.split(X, y), 1):
    # One fixed 20% tuning split keeps the search bounded to three screen fits
    # plus one refit per outer fold (20 GPU fits total).
    inner = StratifiedShuffleSplit(n_splits=1, test_size=.20, random_state=SEED + 100 * fold)
    inner_fit_rel, inner_valid_rel = next(inner.split(X.iloc[outer_train], y[outer_train]))
    inner_fit, inner_valid = outer_train[inner_fit_rel], outer_train[inner_valid_rel]
    candidates = []
    for name, config in CONFIGS.items():
        m = model(config, SEED + 1000 * fold, early_stopping_rounds=EARLY_STOPPING)
        m.fit(X.iloc[inner_fit], y[inner_fit],
              eval_set=[(X.iloc[inner_valid], y[inner_valid])], verbose=False)
        pred = m.predict_proba(X.iloc[inner_valid])[:, 1]
        candidates.append({"config": name,
                           "inner_auc": float(roc_auc_score(y[inner_valid], pred)),
                           "selected_rounds": int(m.best_iteration + 1)})
        del m
        gc.collect()

    winner = max(candidates, key=lambda z: z["inner_auc"])
    # Round count is chosen strictly within outer_train; refit uses every allowed row.
    final = model(CONFIGS[winner["config"]], SEED + 10000 + fold,
                  n_estimators=winner["selected_rounds"])
    final.fit(X.iloc[outer_train], y[outer_train], verbose=False)
    valid_pred = final.predict_proba(X.iloc[outer_valid])[:, 1]
    oof[outer_valid] = valid_pred
    test_predictions.append(final.predict_proba(XT)[:, 1])
    fold_ids[outer_valid] = fold
    record = {"fold": fold, "selected_config": winner["config"],
              "selected_rounds": winner["selected_rounds"],
              "outer_auc": float(roc_auc_score(y[outer_valid], valid_pred)),
              "inner_candidates": candidates}
    records.append(record)
    print(json.dumps(record), flush=True)
    del final
    gc.collect()

auc = float(roc_auc_score(y, oof))
pd.DataFrame({ID: train[ID], "fold": fold_ids, "y": y, "pred": oof}).to_csv(
    "/kaggle/working/oof_nested_dart_xgb.csv", index=False)
pd.DataFrame({ID: test[ID], "pred": np.mean(test_predictions, axis=0)}).to_csv(
    "/kaggle/working/test_nested_dart_xgb.csv", index=False)
metrics = {
    "experiment": "nested-tuned raw-feature XGBoost DART",
    "official_data_only": True, "public_predictions_used": False,
    "target_encoding_used": False, "outer_folds": FOLDS,
    "inner_tuning": "one stratified 20% split within each outer-training partition",
    "gpu_fit_count": 20, "seed": SEED, "configs": CONFIGS,
    "selection_rule": "highest inner holdout AUC; its best_iteration",
    "fold_records": records, "oof_auc": auc,
    "fold_auc_mean": float(np.mean([r["outer_auc"] for r in records])),
    "fold_auc_std": float(np.std([r["outer_auc"] for r in records])),
    "submission_created": False,
}
Path("/kaggle/working/metrics_nested_dart_xgb.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2), flush=True)
