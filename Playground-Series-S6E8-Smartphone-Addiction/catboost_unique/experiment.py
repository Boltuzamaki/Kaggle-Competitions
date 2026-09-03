"""Original S6E8 CatBoost experiment: missingness + accounting composition.

Predictions are trained solely from the mounted competition train.csv.  No
public notebook outputs or external prediction files are read.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "addicted_label"
ID = "id"
SEED = 6082026
ROOT = Path("/kaggle/input")
OUT = Path("/kaggle/working")


def locate_data() -> tuple[Path, Path, Path]:
    """Find the competition mount rather than assuming its directory slug."""
    trains = [p for p in ROOT.rglob("train.csv") if p.is_file()]
    candidates = []
    for train_path in trains:
        test_path = train_path.with_name("test.csv")
        sample_path = train_path.with_name("sample_submission.csv")
        if test_path.exists() and sample_path.exists():
            try:
                cols = pd.read_csv(train_path, nrows=2).columns
                if TARGET in cols and ID in cols:
                    candidates.append((train_path, test_path, sample_path))
            except Exception:
                pass
    if len(candidates) != 1:
        inventory = [str(p) for p in ROOT.rglob("*.csv")][:100]
        raise RuntimeError(
            f"Expected exactly one mounted S6E8 source, got {len(candidates)}. "
            f"CSV inventory: {inventory}"
        )
    print("competition_data=", candidates[0][0].parent)
    return candidates[0]


def safe_ratio(a: pd.Series, b: pd.Series, eps: float = 0.25) -> pd.Series:
    return a / (b + eps)


def engineer(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = frame.drop(columns=[ID, TARGET], errors="ignore").copy()
    raw_num = x.select_dtypes(include=np.number).columns.tolist()

    # A missingness fingerprint is a compact generator-regime identifier.
    missing = x[raw_num].isna()
    x["missing_count"] = missing.sum(axis=1).astype("int8")
    weights = (1 << np.arange(len(raw_num), dtype=np.int64))
    x["missing_mask"] = missing.to_numpy(dtype=np.int8).dot(weights).astype(str)
    for c in raw_num:
        x[f"miss__{c}"] = x[c].isna().astype("int8")

    daily = x["daily_screen_time_hours"]
    weekend = x["weekend_screen_time"]
    sleep = x["sleep_hours"]
    social = x["social_media_hours"]
    gaming = x["gaming_hours"]
    work = x["work_study_hours"]
    components = ["social_media_hours", "gaming_hours", "work_study_hours"]
    present = x[components].notna().sum(axis=1)
    component_sum = x[components].sum(axis=1, min_count=1)

    x["component_present_count"] = present.astype("int8")
    x["component_sum_observed"] = component_sum
    x["entertainment_hours"] = social + gaming
    x["daily_minus_components"] = daily - component_sum
    x["daily_minus_components_complete"] = (daily - component_sum).where(present.eq(3))
    x["weekend_minus_daily"] = weekend - daily
    x["awake_hours"] = 24.0 - sleep
    x["daily_plus_sleep"] = daily + sleep
    x["weekend_plus_sleep"] = weekend + sleep
    x["screen_sleep_ratio"] = safe_ratio(daily, sleep)
    x["weekend_daily_ratio"] = safe_ratio(weekend, daily)
    x["entertainment_daily_share"] = safe_ratio(social + gaming, daily)
    x["work_daily_share"] = safe_ratio(work, daily)
    x["social_entertainment_share"] = safe_ratio(social, social + gaming)
    x["notifications_per_open"] = safe_ratio(x["notifications_per_day"], x["app_opens_per_day"], 1.0)
    x["opens_per_screen_hour"] = safe_ratio(x["app_opens_per_day"], daily)

    # Coarse composition bins let CatBoost estimate nonlinear generator cells.
    for c in ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
              "gaming_hours", "work_study_hours", "sleep_hours"]:
        x[f"bin10__{c}"] = x[c].round(1).fillna(-999).astype(str)

    cat_cols = x.select_dtypes(exclude=np.number).columns.tolist()
    for c in cat_cols:
        x[c] = x[c].fillna("__MISSING__").astype(str)
    return x, cat_cols


train_path, test_path, sample_path = locate_data()
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)
y = train[TARGET].astype("int8")
x, cat_cols = engineer(train)
x_test, test_cat_cols = engineer(test)
assert list(x.columns) == list(x_test.columns)
assert cat_cols == test_cat_cols
print(f"train={x.shape} test={x_test.shape} positive_rate={y.mean():.6f}")
print(f"features={len(x.columns)} categoricals={len(cat_cols)}")

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
oof = np.zeros(len(x), dtype=np.float32)
test_pred = np.zeros(len(x_test), dtype=np.float64)
fold_metrics = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(x, y), 1):
    model = CatBoostClassifier(
        iterations=2400,
        depth=8,
        learning_rate=0.055,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=8,
        random_strength=0.4,
        border_count=128,
        random_seed=SEED + fold * 97,
        task_type="GPU",
        devices="0",
        od_type="Iter",
        od_wait=180,
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(
        x.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_cols,
        eval_set=(x.iloc[va_idx], y.iloc[va_idx]), use_best_model=True,
    )
    oof[va_idx] = model.predict_proba(x.iloc[va_idx])[:, 1]
    test_pred += model.predict_proba(x_test)[:, 1] / skf.n_splits
    score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
    fold_metrics.append({"fold": fold, "auc": float(score),
                         "best_iteration": int(model.get_best_iteration() + 1)})
    print("fold_result=", fold_metrics[-1])
    if fold == 1:
        pd.DataFrame(model.get_feature_importance(prettified=True)).to_csv(
            OUT / "feature_importance.csv", index=False
        )
    del model
    gc.collect()

overall = float(roc_auc_score(y, oof))
metrics = {
    "provenance": "trained from scratch using competition train.csv only",
    "metric": "roc_auc", "oof_auc": overall, "folds": fold_metrics,
    "n_features": len(x.columns), "features": list(x.columns),
}
sample[TARGET] = test_pred
sample.to_csv(OUT / "submission_catboost_unique.csv", index=False)
pd.DataFrame({ID: train[ID], TARGET: y, "oof_prediction": oof}).to_csv(
    OUT / "oof_catboost_unique.csv", index=False
)
(OUT / "validation.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
