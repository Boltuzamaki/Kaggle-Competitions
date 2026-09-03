"""Ordered CatBoost on fold-local, target-free hierarchical support features.

This deliberately does not construct target encodings.  Counts for a scored row
are learned only from the relevant fit partition; model-training rows use
leave-one-out counts so their own occurrence cannot inflate support.  A bounded
configuration screen and early stopping happen wholly inside each outer-fit
fold.  Outer labels are prediction-only.
"""
from pathlib import Path
import gc
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

TARGET, ID, SEED = "addicted_label", "id", 20260804
SMOKE = os.getenv("S6E8_SMOKE", "0") == "1"
FOLDS = 2 if SMOKE else 5
MAX_ITER = 30 if SMOKE else 2600
PAIR_SPECS = [
    ("daily_screen_time_hours", "weekend_screen_time"),
    ("daily_screen_time_hours", "social_media_hours"),
    ("daily_screen_time_hours", "sleep_hours"),
    ("daily_screen_time_hours", "notifications_per_day"),
    ("social_media_hours", "gaming_hours"),
    ("notifications_per_day", "app_opens_per_day"),
    ("stress_level", "academic_work_impact"),
]
CONFIGS = {
    "ordered_d7": dict(depth=7, learning_rate=.045, l2_leaf_reg=9., random_strength=.35),
    "ordered_d8": dict(depth=8, learning_rate=.035, l2_leaf_reg=12., random_strength=.55),
}
if SMOKE:
    CONFIGS = {"ordered_d7": CONFIGS["ordered_d7"]}


def locate():
    for root in (Path("/kaggle/input"), Path(".")):
        if not root.exists():
            continue
        for p in root.rglob("train.csv"):
            try:
                cols = pd.read_csv(p, nrows=1).columns
            except Exception:
                continue
            if TARGET in cols and (p.parent / "test.csv").exists():
                return p, p.parent / "test.csv"
    raise FileNotFoundError("official S6E8 train/test tables not found")


def key(s, rounded=False):
    if rounded and pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce").round(1)
    return s.astype("string").fillna("__NA__")


def raw_view(df):
    x = df.drop(columns=[ID, TARGET], errors="ignore").copy()
    numeric = list(x.select_dtypes(include="number").columns)
    for c in numeric:
        x[c + "__missing"] = x[c].isna().astype("int8")
    x["missing_count"] = df.drop(columns=[ID, TARGET], errors="ignore").isna().sum(axis=1).astype("int8")
    x["component_sum"] = x[["social_media_hours", "gaming_hours", "work_study_hours"]].sum(axis=1, min_count=3)
    x["screen_residual"] = x["daily_screen_time_hours"] - x["component_sum"]
    x["weekend_gap"] = x["weekend_screen_time"] - x["daily_screen_time_hours"]
    x["leisure_share"] = (x["social_media_hours"] + x["gaming_hours"]) / (x["daily_screen_time_hours"] + .25)
    x["screen_sleep_ratio"] = x["daily_screen_time_hours"] / (x["sleep_hours"] + .25)
    x["notification_open_ratio"] = x["notifications_per_day"] / (x["app_opens_per_day"] + 1.)
    return x.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def key_view(df):
    raw = df.drop(columns=[ID, TARGET], errors="ignore")
    out = {}
    for c in raw.columns:
        out["exact__" + c] = key(raw[c])
        if pd.api.types.is_numeric_dtype(raw[c]):
            out["round1__" + c] = key(raw[c], rounded=True)
    for a, b in PAIR_SPECS:
        out[f"pair_exact__{a}__{b}"] = key(raw[a]) + "|" + key(raw[b])
        if pd.api.types.is_numeric_dtype(raw[a]) and pd.api.types.is_numeric_dtype(raw[b]):
            out[f"pair_round1__{a}__{b}"] = key(raw[a], True) + "|" + key(raw[b], True)
    return pd.DataFrame(out).reset_index(drop=True)


def support_features(fit_keys, apply_keys, leave_one_out=False):
    """Map target-free counts fitted on fit_keys into apply_keys."""
    z = {}
    for c in fit_keys.columns:
        counts = fit_keys[c].value_counts(dropna=False)
        n = apply_keys[c].map(counts).fillna(0).to_numpy(dtype="float32")
        if leave_one_out:
            # Valid only when apply_keys is the same row-aligned fit table.
            n = np.maximum(n - 1., 0.)
        z[c + "__log_support"] = np.log1p(n)
        z[c + "__novel"] = (n == 0).astype("int8")
    out = pd.DataFrame(z)
    # Conditional-support ratios separate dense generator cells from accidental
    # pair matches while remaining wholly target-free.
    for a, b in PAIR_SPECS:
        for tag in ("exact", "round1"):
            pair = f"pair_{tag}__{a}__{b}__log_support"
            ka, kb = f"{tag}__{a}__log_support", f"{tag}__{b}__log_support"
            if pair in out and ka in out and kb in out:
                p, sa, sb = np.expm1(out[pair]), np.expm1(out[ka]), np.expm1(out[kb])
                out[f"pair_{tag}__{a}__{b}__backoff"] = np.log((p + 1.) / np.sqrt((sa + 1.) * (sb + 1.))).astype("float32")
    return out


def design(base_fit, keys_fit, base_apply, keys_apply, fit_rows=False):
    s = support_features(keys_fit, keys_apply, leave_one_out=fit_rows)
    a = base_apply.reset_index(drop=True).copy()
    cats = list(a.select_dtypes(exclude="number").columns)
    for c in cats:
        a[c] = a[c].astype("string").fillna("__NA__")
    med = base_fit.select_dtypes(include="number").median()
    for c in a.select_dtypes(include="number"):
        a[c] = a[c].fillna(med.get(c, 0.)).astype("float32")
    return pd.concat([a, s], axis=1), cats


def model(cfg, seed, iterations, early=False):
    kw = dict(iterations=iterations, loss_function="Logloss", eval_metric="AUC",
              boosting_type="Ordered", grow_policy="SymmetricTree", bootstrap_type="Bernoulli",
              subsample=.82, rsm=.88, random_seed=seed, thread_count=-1,
              allow_writing_files=False, verbose=False, **cfg)
    if early:
        kw.update(od_type="Iter", od_wait=20 if SMOKE else 140)
    return CatBoostClassifier(**kw)


tp, vp = locate()
tr, te = pd.read_csv(tp), pd.read_csv(vp)
if SMOKE:
    tr = tr.groupby(TARGET, group_keys=False).sample(n=2500, random_state=SEED).reset_index(drop=True)
    te = te.head(1200).reset_index(drop=True)
y = tr[TARGET].to_numpy("int8")
btr, bte, ktr, kte = raw_view(tr), raw_view(te), key_view(tr), key_view(te)
outer = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
oof, fold_id, test_pred, records = np.zeros(len(tr)), np.zeros(len(tr), "int8"), [], []
t0 = time.time()
for fold, (fit, val) in enumerate(outer.split(btr, y), 1):
    ss = StratifiedShuffleSplit(1, test_size=.16, random_state=SEED + fold)
    ai, av = next(ss.split(fit, y[fit])); inner_fit, inner_val = fit[ai], fit[av]
    xi, cats = design(btr.iloc[inner_fit], ktr.iloc[inner_fit], btr.iloc[inner_fit], ktr.iloc[inner_fit], True)
    xv, _ = design(btr.iloc[inner_fit], ktr.iloc[inner_fit], btr.iloc[inner_val], ktr.iloc[inner_val])
    trials = []
    for name, cfg in CONFIGS.items():
        m = model(cfg, SEED + 100 * fold, MAX_ITER, True)
        m.fit(xi, y[inner_fit], cat_features=cats, eval_set=(xv, y[inner_val]), use_best_model=True)
        p = m.predict_proba(xv)[:, 1]
        trials.append({"config": name, "inner_auc": float(roc_auc_score(y[inner_val], p)),
                       "iterations": max(1, int(m.get_best_iteration() + 1))})
        del m; gc.collect()
    winner = max(trials, key=lambda d: d["inner_auc"])
    xf, cats = design(btr.iloc[fit], ktr.iloc[fit], btr.iloc[fit], ktr.iloc[fit], True)
    xv, _ = design(btr.iloc[fit], ktr.iloc[fit], btr.iloc[val], ktr.iloc[val])
    xt, _ = design(btr.iloc[fit], ktr.iloc[fit], bte, kte)
    m = model(CONFIGS[winner["config"]], SEED + 1000 + fold, winner["iterations"])
    m.fit(xf, y[fit], cat_features=cats)
    pv = m.predict_proba(xv)[:, 1]
    oof[val], fold_id[val] = pv, fold
    test_pred.append(m.predict_proba(xt)[:, 1])
    records.append({"fold": fold, "outer_auc": float(roc_auc_score(y[val], pv)),
                    "selected": winner, "inner_trials": trials})
    print(json.dumps(records[-1]), flush=True)
    del xi, xv, xf, xt, m; gc.collect()

out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("catboost_ordered_support/output")
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({ID: tr[ID], "fold": fold_id, "y": y, "pred": oof}).to_csv(out / "oof_ordered_support_catboost.csv", index=False)
pd.DataFrame({ID: te[ID], "pred": np.mean(test_pred, axis=0)}).to_csv(out / "test_ordered_support_catboost.csv", index=False)
metrics = {"model": "ordered CatBoost on target-free hierarchical support", "official_data_only": True,
           "public_predictions_used": False, "target_encoding_used": False, "outer_folds": FOLDS,
           "tuning": "bounded 16% holdout wholly inside each outer-fit fold", "configs": CONFIGS,
           "leave_one_out_training_support": True, "fold_records": records,
           "oof_auc": float(roc_auc_score(y, oof)), "runtime_seconds": time.time() - t0,
           "smoke": SMOKE, "submission_created": False}
(out / "metrics_ordered_support_catboost.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2), flush=True)
