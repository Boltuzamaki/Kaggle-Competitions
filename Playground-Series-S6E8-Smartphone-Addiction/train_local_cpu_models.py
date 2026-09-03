"""Benchmark CPU tabular models on the canonical seed-42 holdout.

Writes separate artifacts only; it intentionally never replaces submission.csv.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


SEED = 42
TARGET = "addicted_label"
ID = "id"
OUT = Path("artifacts/local_cpu")


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.drop(columns=[ID], errors="ignore").copy()
    numeric = x.select_dtypes(include="number").columns
    x["missing_count"] = x.isna().sum(axis=1)
    # Ratios/totals encode plausible addiction intensity while remaining generic.
    x["leisure_hours"] = x[["social_media_hours", "gaming_hours"]].sum(axis=1, min_count=1)
    x["productive_rest_hours"] = x[["work_study_hours", "sleep_hours"]].sum(axis=1, min_count=1)
    x["digital_events"] = x[["notifications_per_day", "app_opens_per_day"]].sum(axis=1, min_count=1)
    x["screen_weekend_gap"] = x["weekend_screen_time"] - x["daily_screen_time_hours"]
    x["leisure_screen_share"] = x["leisure_hours"] / (x["daily_screen_time_hours"] + 0.25)
    for col in numeric:
        x[f"{col}_missing"] = x[col].isna().astype("int8")
    return x


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")
    y = train[TARGET]
    x = add_features(train.drop(columns=[TARGET]))
    xt = add_features(test)
    categorical = list(x.select_dtypes(exclude="number").columns)
    for col in categorical:
        levels = pd.Index(pd.concat([x[col], xt[col]], ignore_index=True).fillna("Missing").astype(str).unique())
        dtype = pd.CategoricalDtype(categories=levels)
        x[col] = x[col].fillna("Missing").astype(str).astype(dtype)
        xt[col] = xt[col].fillna("Missing").astype(str).astype(dtype)

    idx_train, idx_valid = train_test_split(
        np.arange(len(y)), test_size=0.15, stratify=y, random_state=SEED
    )
    configs = {
        "lgbm_leaf31": dict(num_leaves=31, min_child_samples=50, max_depth=-1),
        "lgbm_leaf63": dict(num_leaves=63, min_child_samples=80, max_depth=-1),
        "lgbm_leaf127": dict(num_leaves=127, min_child_samples=120, max_depth=-1),
    }
    results = []
    valid_predictions = {}
    test_predictions = {}
    for name, extra in configs.items():
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=2500, learning_rate=0.035,
            subsample=0.85, colsample_bytree=0.9, reg_alpha=0.05,
            reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1,
            **extra,
        )
        model.fit(
            x.iloc[idx_train], y.iloc[idx_train],
            eval_set=[(x.iloc[idx_valid], y.iloc[idx_valid])], eval_metric="auc",
            callbacks=[lgb.early_stopping(150, verbose=False)],
            categorical_feature=categorical,
        )
        vp = model.predict_proba(x.iloc[idx_valid])[:, 1]
        tp = model.predict_proba(xt)[:, 1]
        score = roc_auc_score(y.iloc[idx_valid], vp)
        print(f"{name}: AUC={score:.8f}, iteration={model.best_iteration_}", flush=True)
        results.append({"model": name, "auc": score, "iteration": model.best_iteration_})
        valid_predictions[name], test_predictions[name] = vp, tp
        model.booster_.save_model(str(OUT / f"{name}.txt"))
        out = sample.copy(); out[TARGET] = tp
        out.to_csv(OUT / f"submission_{name}.csv", index=False)

    # ExtraTrees adds a structurally different signal for blending.
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-1)
    all_cat = pd.concat([x[categorical], xt[categorical]], ignore_index=True).astype(str)
    enc.fit(all_cat)
    xet = x.copy(); xtet = xt.copy()
    xet[categorical] = enc.transform(x[categorical].astype(str))
    xtet[categorical] = enc.transform(xt[categorical].astype(str))
    imp = SimpleImputer(strategy="median")
    imp.fit(pd.concat([xet, xtet], ignore_index=True))
    xet = imp.transform(xet); xtet = imp.transform(xtet)
    et = ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=8, max_features=0.9,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    et.fit(xet[idx_train], y.iloc[idx_train])
    vp = et.predict_proba(xet[idx_valid])[:, 1]
    tp = et.predict_proba(xtet)[:, 1]
    score = roc_auc_score(y.iloc[idx_valid], vp)
    print(f"extra_trees: AUC={score:.8f}", flush=True)
    results.append({"model": "extra_trees", "auc": score, "iteration": 500})
    valid_predictions["extra_trees"], test_predictions["extra_trees"] = vp, tp
    out = sample.copy(); out[TARGET] = tp
    out.to_csv(OUT / "submission_extra_trees.csv", index=False)

    best = max(results, key=lambda r: r["auc"])["model"]
    # Search simple convex blends using only the untouched holdout.
    for other in valid_predictions:
        if other == best:
            continue
        for w in (0.7, 0.8, 0.9):
            pred = w * valid_predictions[best] + (1-w) * valid_predictions[other]
            results.append({"model": f"blend_{w:.1f}_{best}_{other}",
                            "auc": roc_auc_score(y.iloc[idx_valid], pred), "iteration": np.nan})
    result_frame = pd.DataFrame(results).sort_values("auc", ascending=False)
    result_frame.to_csv(OUT / "validation_results.csv", index=False)
    winner = result_frame.iloc[0]["model"]
    if winner.startswith("blend_"):
        _, weight, primary, other = winner.split("_", 3)
        final_pred = float(weight) * test_predictions[primary] + (1-float(weight)) * test_predictions[other]
    else:
        final_pred = test_predictions[winner]
    out = sample.copy(); out[TARGET] = final_pred
    out.to_csv(OUT / "submission_best_local.csv", index=False)
    print(result_frame.head(12).to_string(index=False), flush=True)
    print(f"Best candidate: {winner}; saved submission_best_local.csv", flush=True)


if __name__ == "__main__":
    main()
