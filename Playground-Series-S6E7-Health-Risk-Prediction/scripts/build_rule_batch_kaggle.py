"""Build independent rule-aware Kaggle experiments for parallel execution."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = {
    "rule_xgb_gpu": ("Health Risk Rule XGB GPU", "health-risk-rule-xgb-gpu", True, "xgb"),
    "rule_cat_gpu": ("Health Risk Rule CatBoost GPU", "health-risk-rule-catboost-gpu", True, "cat"),
    "rule_hgb_cpu": ("Health Risk Rule HGBC CPU", "health-risk-rule-hgbc-cpu", False, "hgb"),
    "rule_extra_cpu": ("Health Risk Rule ExtraTrees CPU", "health-risk-rule-extratrees-cpu", False, "extra"),
    "rule_rf_cpu": ("Health Risk Rule Random Forest CPU", "health-risk-rule-random-forest-cpu", False, "rf"),
    "rule_sgd_cpu": ("Health Risk Rule SGD LogLoss CPU", "health-risk-rule-sgd-logloss-cpu", False, "sgd"),
    "rule_lgb_cpu": ("Health Risk Rule LightGBM CPU", "health-risk-rule-lightgbm-cpu", False, "lgb"),
    "rule_ebm_cpu": ("Health Risk Rule EBM CPU", "health-risk-rule-ebm-cpu", False, "ebm"),
}


def cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "",
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


COMMON = r'''
import gc, json, time, warnings
import subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight
warnings.filterwarnings("ignore")

SEED = 2027
FOLDS = 5
ID = "id"
TARGET = "health_condition"
CLASSES = np.array(["at-risk", "fit", "unhealthy"])
CLASS_TO_INT = {c:i for i,c in enumerate(CLASSES)}
MODEL_KIND = "__MODEL_KIND__"

base = Path("/kaggle/input/competitions/playground-series-s6e7")
train = pd.read_csv(base / "train.csv")
test = pd.read_csv(base / "test.csv")
y = train[TARGET].map(CLASS_TO_INT).to_numpy()

def engineer(df):
    z = df.drop(columns=[ID, TARGET], errors="ignore").copy()
    key = ["sleep_duration", "stress_level", "physical_activity_level"]
    z["key_missing_count"] = z[key].isna().sum(axis=1)
    z["key_missing_pattern"] = (
        z["sleep_duration"].isna().astype(str) + "_" +
        z["stress_level"].isna().astype(str) + "_" +
        z["physical_activity_level"].isna().astype(str)
    )
    s = z["sleep_duration"]
    z["sleep_lt6"] = np.where(s.isna(), np.nan, (s < 6).astype(float))
    z["sleep_lt7"] = np.where(s.isna(), np.nan, (s < 7).astype(float))
    z["sleep_zone"] = pd.cut(
        s, [-np.inf, 6, 7, np.inf], right=False,
        labels=["lt6", "6to7", "ge7"]
    ).astype("object").fillna("__MISSING__")
    z["rule_unhealthy"] = (
        (s < 6) & z["stress_level"].eq("high")
    ).astype(float)
    z["rule_fit"] = (
        (s >= 7) & z["stress_level"].eq("low") &
        z["physical_activity_level"].eq("active")
    ).astype(float)
    z["rule_state"] = (
        z["sleep_zone"].astype(str) + "|" +
        z["stress_level"].fillna("__MISSING__").astype(str) + "|" +
        z["physical_activity_level"].fillna("__MISSING__").astype(str)
    )
    numerics = z.select_dtypes(exclude=["object", "category"]).columns
    for c in numerics:
        if c not in ("sleep_lt6", "sleep_lt7", "rule_unhealthy", "rule_fit"):
            # Generator fingerprints: fractional digits at several resolutions.
            for scale in (10, 100):
                z[f"{c}_frac_{scale}"] = (
                    np.floor((z[c].abs() * scale) + 1e-7) % scale
                )
    if {"calorie_expenditure", "step_count"} <= set(z.columns):
        z["calories_per_step"] = z["calorie_expenditure"] / (z["step_count"] + 1)
    return z

X = engineer(train)
T = engineer(test)
cats = list(X.select_dtypes(include=["object", "category"]).columns)
nums = [c for c in X.columns if c not in cats]
enc = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1,
    encoded_missing_value=-1,
)
both = pd.concat([X[cats].fillna("__MISSING__"), T[cats].fillna("__MISSING__")])
enc.fit(both)
X[cats] = enc.transform(X[cats].fillna("__MISSING__"))
T[cats] = enc.transform(T[cats].fillna("__MISSING__"))
med = X[nums].median()
X[nums] = X[nums].fillna(med)
T[nums] = T[nums].fillna(med)
X = X.astype(np.float32)
T = T.astype(np.float32)
print(MODEL_KIND, X.shape, "categoricals", len(cats))

def make_model():
    if MODEL_KIND == "ebm":
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "interpret"])
            from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(
            interactions=24, max_bins=256, max_interaction_bins=32,
            learning_rate=.035, max_rounds=5000, min_samples_leaf=20,
            outer_bags=8, validation_size=.10, early_stopping_rounds=100,
            random_state=SEED, n_jobs=-1,
        )
    if MODEL_KIND == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=12000, max_depth=7, min_child_weight=25,
            learning_rate=0.0094647877, reg_alpha=1.303154,
            reg_lambda=0.007621, gamma=3.704746, max_delta_step=3,
            colsample_bytree=0.789614, colsample_bylevel=0.821573,
            colsample_bynode=0.948245, subsample=0.755008,
            objective="multi:softprob", eval_metric="mlogloss",
            tree_method="hist", device="cuda", random_state=SEED,
            early_stopping_rounds=100,
        )
    if MODEL_KIND == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=7000, depth=9, learning_rate=0.035,
            loss_function="MultiClass", eval_metric="MultiClass",
            l2_leaf_reg=6, random_seed=SEED, task_type="GPU",
            devices="0", verbose=250, od_type="Iter", od_wait=150,
            allow_writing_files=False,
        )
    if MODEL_KIND == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            learning_rate=.075, max_iter=650, max_leaf_nodes=63,
            min_samples_leaf=35, l2_regularization=2.0,
            early_stopping=True, validation_fraction=.08,
            n_iter_no_change=40, random_state=SEED,
        )
    if MODEL_KIND == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=420, max_features=.8, min_samples_leaf=3,
            class_weight="balanced", n_jobs=-1, random_state=SEED,
            max_samples=.80, bootstrap=True,
        )
    if MODEL_KIND == "sgd":
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(
            loss="log_loss", penalty="elasticnet", alpha=2e-5,
            l1_ratio=.05, max_iter=120, tol=1e-5,
            average=True, random_state=SEED, n_jobs=-1,
        )
    if MODEL_KIND == "lgb":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=3500, learning_rate=.025, num_leaves=63,
            max_depth=-1, min_child_samples=40, subsample=.8,
            colsample_bytree=.85, reg_alpha=.5, reg_lambda=2.0,
            objective="multiclass", random_state=SEED, n_jobs=-1,
            verbosity=-1,
        )
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(
        n_estimators=320, max_features=.75, min_samples_leaf=3,
        class_weight="balanced", n_jobs=-1, random_state=SEED,
        max_samples=.80, bootstrap=True,
    )

oof = np.zeros((len(X), 3), dtype=np.float32)
pred = np.zeros((len(T), 3), dtype=np.float64)
fold_scores = []
t0 = time.time()
skf = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    model = make_model()
    weights = compute_sample_weight("balanced", y[tr])
    fit_kw = {"sample_weight": weights}
    if MODEL_KIND == "xgb":
        fit_kw["eval_set"] = [(X.iloc[va], y[va])]
        fit_kw["verbose"] = False
    elif MODEL_KIND == "cat":
        fit_kw["eval_set"] = (X.iloc[va], y[va])
    model.fit(X.iloc[tr], y[tr], **fit_kw)
    oof[va] = model.predict_proba(X.iloc[va])
    pred += model.predict_proba(T) / FOLDS
    score = balanced_accuracy_score(y[va], oof[va].argmax(1))
    fold_scores.append(float(score))
    print("fold", fold, score)
    del model
    gc.collect()

overall = float(balanced_accuracy_score(y, oof.argmax(1)))
oof_df = pd.DataFrame({ID: train[ID]})
test_df = pd.DataFrame({ID: test[ID]})
for j,c in enumerate(CLASSES):
    oof_df[c] = oof[:,j]
    test_df[c] = pred[:,j]
oof_df.to_csv("oof_preds.csv", index=False)
test_df.to_csv("test_preds.csv", index=False)
sub = pd.DataFrame({ID:test[ID], TARGET:CLASSES[pred.argmax(1)]})
sub.to_csv("submission.csv", index=False)
summary = {
    "experiment": MODEL_KIND, "fold_scores": fold_scores,
    "oof_balanced_accuracy": overall,
    "elapsed_minutes": (time.time()-t0)/60,
    "competition_data_only": True,
}
Path("training_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
'''


def main() -> None:
    for slug, (title, remote_slug, gpu, kind) in EXPERIMENTS.items():
        out = ROOT / "kaggle_kernels" / slug
        out.mkdir(parents=True, exist_ok=True)
        source = COMMON.replace("__MODEL_KIND__", kind)
        nb = {
            "cells": [cell(source)],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        nb["cells"][0]["id"] = f"{kind}-main"
        code_file = f"{slug}.ipynb"
        (out / code_file).write_text(json.dumps(nb, indent=1), encoding="utf-8")
        metadata = {
            "id": f"boltuzamaki/{remote_slug}",
            "title": title,
            "code_file": code_file,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": gpu,
            "enable_internet": kind == "ebm",
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": ["playground-series-s6e7"],
        }
        if gpu:
            metadata["machine_shape"] = "NvidiaTeslaT4"
        (out / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        compile(source, slug, "exec")
        print(out)


if __name__ == "__main__":
    main()
