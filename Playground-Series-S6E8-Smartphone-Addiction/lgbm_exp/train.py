"""Original LightGBM experiment for Playground S6E8.

No public predictions or external labels are read. Validation target encodings are
fit on the training partition only; final test encodings are fit on all training rows.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "artifacts"
TARGET, ID, SEED = "addicted_label", "id", 3407


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.drop(columns=[ID, TARGET], errors="ignore").copy()
    nums = list(x.select_dtypes(include="number").columns)
    x["missing_count"] = x.isna().sum(axis=1).astype("int8")
    for c in nums:
        x[c + "__missing"] = x[c].isna().astype("int8")
    # Domain-motivated accounting and intensity features.
    x["leisure_hours"] = x["social_media_hours"] + x["gaming_hours"]
    x["accounted_hours"] = x["social_media_hours"] + x["gaming_hours"] + x["work_study_hours"]
    x["unaccounted_screen"] = x["daily_screen_time_hours"] - x["accounted_hours"]
    x["weekend_gap"] = x["weekend_screen_time"] - x["daily_screen_time_hours"]
    x["weekend_ratio"] = x["weekend_screen_time"] / (x["daily_screen_time_hours"] + .25)
    x["leisure_share"] = x["leisure_hours"] / (x["daily_screen_time_hours"] + .25)
    x["work_share"] = x["work_study_hours"] / (x["daily_screen_time_hours"] + .25)
    x["notif_per_screen"] = x["notifications_per_day"] / (x["daily_screen_time_hours"] + .25)
    x["opens_per_screen"] = x["app_opens_per_day"] / (x["daily_screen_time_hours"] + .25)
    x["notif_per_open"] = x["notifications_per_day"] / (x["app_opens_per_day"] + 2.0)
    x["sleep_screen_balance"] = x["sleep_hours"] - x["daily_screen_time_hours"]
    return x.replace([np.inf, -np.inf], np.nan)


def add_encodings(fit_x, apply_x, y, cols, prior, smoothing=40.0):
    """Exact-value frequency and smoothed target encoding, learned on fit_x only."""
    out = apply_x.copy()
    for c in cols:
        # String sentinel makes missingness an explicit level in the statistic.
        fk = fit_x[c].astype("string").fillna("__NA__")
        ak = apply_x[c].astype("string").fillna("__NA__")
        stat = pd.DataFrame({"key": fk, "y": np.asarray(y)}).groupby("key", observed=True).y.agg(["sum", "count"])
        te = (stat["sum"] + smoothing * prior) / (stat["count"] + smoothing)
        out[c + "__te"] = ak.map(te).fillna(prior).astype("float32")
        out[c + "__logfreq"] = np.log1p(ak.map(stat["count"]).fillna(0)).astype("float32")
    return out


def align_categories(a, b):
    cats = list(a.select_dtypes(exclude="number").columns)
    for c in cats:
        levels = pd.Index(pd.concat([a[c], b[c]], ignore_index=True).astype("string").fillna("Missing").unique())
        dtype = pd.CategoricalDtype(levels)
        a[c] = a[c].astype("string").fillna("Missing").astype(dtype)
        b[c] = b[c].astype("string").fillna("Missing").astype(dtype)
    return cats


def model(seed, n_estimators):
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=n_estimators, learning_rate=.025,
        num_leaves=31, max_depth=-1, min_child_samples=160,
        subsample=.85, subsample_freq=1, colsample_bytree=.86,
        reg_alpha=.15, reg_lambda=2.5, max_bin=255,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    y = train[TARGET].astype("int8")
    raw_cols = [c for c in train.columns if c not in (ID, TARGET)]
    tr0, te0 = base_features(train), base_features(test)
    dev, val = train_test_split(np.arange(len(train)), test_size=.18, stratify=y, random_state=SEED)
    prior = float(y.iloc[dev].mean())
    xdev = add_encodings(tr0.iloc[dev], tr0.iloc[dev], y.iloc[dev], raw_cols, prior)
    xval = add_encodings(tr0.iloc[dev], tr0.iloc[val], y.iloc[dev], raw_cols, prior)
    cats = align_categories(xdev, xval)
    scores, iterations = [], []
    for seed in (3407, 7117):
        m = model(seed, 3000)
        m.fit(xdev, y.iloc[dev], eval_set=[(xval, y.iloc[val])], eval_metric="auc",
              categorical_feature=cats, callbacks=[lgb.early_stopping(180, verbose=False)])
        p = m.predict_proba(xval)[:, 1]
        scores.append(float(roc_auc_score(y.iloc[val], p))); iterations.append(m.best_iteration_)
        print(f"seed={seed} auc={scores[-1]:.8f} iteration={m.best_iteration_}", flush=True)
    # Fit independent full-data models with the validated tree count.
    prior_all = float(y.mean())
    xall = add_encodings(tr0, tr0, y, raw_cols, prior_all)
    xte = add_encodings(tr0, te0, y, raw_cols, prior_all)
    cats = align_categories(xall, xte)
    test_preds = []
    final_n = max(50, int(round(np.mean(iterations) / .82)))  # scale 82% dev fit to full data
    importances = []
    for seed in (3407, 7117, 9919):
        m = model(seed, final_n)
        m.fit(xall, y, categorical_feature=cats)
        test_preds.append(m.predict_proba(xte)[:, 1])
        importances.append(pd.Series(m.feature_importances_, index=xall.columns, name=str(seed)))
        m.booster_.save_model(str(OUT / f"lgbm_seed{seed}.txt"))
    pred = np.mean(test_preds, axis=0)
    submission = sample.copy(); submission[TARGET] = pred
    submission.to_csv(OUT / "submission_lgbm_original.csv", index=False)
    pd.concat(importances, axis=1).assign(mean=lambda z: z.mean(axis=1)).sort_values("mean", ascending=False).to_csv(OUT / "feature_importance.csv")
    report = {"provenance": "trained_from_scratch_train_csv_only", "holdout_seed": SEED,
              "holdout_fraction": .18, "seed_auc": scores, "mean_auc": float(np.mean(scores)),
              "best_auc": float(max(scores)), "best_iterations": iterations, "full_fit_trees": final_n,
              "prediction_min": float(pred.min()), "prediction_max": float(pred.max())}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
