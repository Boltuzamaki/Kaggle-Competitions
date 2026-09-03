"""Private Kaggle GPU experiment for Playground S6E8.

Trains diverse CatBoost models on one fixed stratified validation split, reports
individual and blended AUC, then fits to the best iteration and writes test
predictions. This script deliberately does not call the Kaggle submission API.
"""

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


DATA = Path("/kaggle/input/playground-series-s6e8")
TARGET = "addicted_label"
ID = "id"
SEED = 2026


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    x = df.drop(columns=[ID], errors="ignore").copy()
    numeric = x.select_dtypes(include="number").columns.tolist()
    # Missingness can be informative in synthetic Playground data.
    x["missing_count"] = x.isna().sum(axis=1).astype("int8")
    x["missing_pattern"] = x[numeric].isna().astype("int8").astype(str).agg("".join, axis=1)

    # Stable behavioral aggregates/contrasts. Guard names so this remains
    # reproducible if Kaggle changes capitalization in a data refresh.
    lower = {c.lower(): c for c in numeric}
    def find(*terms):
        for lc, original in lower.items():
            if all(t in lc for t in terms):
                return original
        return None

    daily = find("daily", "screen") or find("screen", "time")
    weekend = find("weekend", "screen")
    social = find("social")
    gaming = find("gaming")
    work = find("work") or find("study")
    sleep = find("sleep")
    if daily and weekend:
        x["weekend_daily_gap"] = x[weekend] - x[daily]
        x["weekend_daily_ratio"] = x[weekend] / (x[daily] + 0.25)
    usage = [c for c in (social, gaming, work) if c]
    if usage:
        x["component_hours_sum"] = x[usage].sum(axis=1, min_count=1)
    if daily and usage:
        x["unallocated_screen_hours"] = x[daily] - x[usage].sum(axis=1, min_count=1)
    if daily and sleep:
        x["screen_sleep_ratio"] = x[daily] / (x[sleep] + 0.25)

    for c in x.select_dtypes(exclude="number").columns:
        x[c] = x[c].fillna("__MISSING__").astype(str)
    return x


train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
sample = pd.read_csv(DATA / "sample_submission.csv")
y = train.pop(TARGET).astype("int8")
x = engineer(train)
x_test = engineer(test)
cat_cols = x.select_dtypes(exclude="number").columns.tolist()

tr_idx, va_idx = train_test_split(
    np.arange(len(x)), test_size=0.12, random_state=SEED, stratify=y
)

configs = [
    dict(depth=7, learning_rate=0.07, l2_leaf_reg=7, random_seed=2026,
         random_strength=0.35),
    dict(depth=9, learning_rate=0.055, l2_leaf_reg=10, random_seed=913,
         random_strength=0.55),
]

valid_preds, test_preds, scores, iterations = [], [], [], []
for number, config in enumerate(configs, 1):
    model = CatBoostClassifier(
        iterations=2200,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="CPU",
        thread_count=-1,
        border_count=192,
        od_type="Iter",
        od_wait=180,
        verbose=200,
        allow_writing_files=False,
        **config,
    )
    model.fit(
        x.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_cols,
        eval_set=(x.iloc[va_idx], y.iloc[va_idx]), use_best_model=True,
    )
    vp = model.predict_proba(x.iloc[va_idx])[:, 1]
    tp = model.predict_proba(x_test)[:, 1]
    score = roc_auc_score(y.iloc[va_idx], vp)
    best = model.get_best_iteration() + 1
    print(f"model={number} auc={score:.8f} best_iteration={best}")
    valid_preds.append(vp)
    test_preds.append(tp)
    scores.append(float(score))
    iterations.append(int(best))
    if number == 1:
        pd.DataFrame(model.get_feature_importance(prettified=True)).to_csv(
            "/kaggle/working/feature_importance.csv", index=False
        )
    del model, vp, tp
    gc.collect()

# Rank averaging tends to be robust across differently calibrated tree models.
def rank01(a):
    return pd.Series(a).rank(method="average", pct=True).to_numpy()

valid_rank_blend = np.mean([rank01(p) for p in valid_preds], axis=0)
test_rank_blend = np.mean([rank01(p) for p in test_preds], axis=0)
blend_auc = float(roc_auc_score(y.iloc[va_idx], valid_rank_blend))
print(f"rank_blend_auc={blend_auc:.8f}")

sample[TARGET] = test_rank_blend
sample.to_csv("/kaggle/working/submission_gpu_catboost.csv", index=False)
metrics = {
    "metric": "roc_auc",
    "validation_rows": len(va_idx),
    "model_auc": scores,
    "best_iterations": iterations,
    "rank_blend_auc": blend_auc,
    "features": list(x.columns),
}
Path("/kaggle/working/validation.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
