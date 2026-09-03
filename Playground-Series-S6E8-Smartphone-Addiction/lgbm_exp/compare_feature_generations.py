#!/usr/bin/env python3
"""Matched honest holdouts for extending the public single-family LightGBM."""
from pathlib import Path
import gc, json, time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "feature_search"
TARGET, ID = "addicted_label", "id"
# Confirmation split. Seed 3407 was screened first; its recorded AUCs were
# exact 0.96742153, coarse singles 0.96731601, singles plus pairs 0.96746127.
SEEDS = [20260804]
PAIR_SPECS = [
    ("daily_screen_time_hours", "weekend_screen_time"),
    ("daily_screen_time_hours", "social_media_hours"),
    ("daily_screen_time_hours", "sleep_hours"),
    ("social_media_hours", "gaming_hours"),
    ("social_media_hours", "work_study_hours"),
    ("notifications_per_day", "app_opens_per_day"),
    ("stress_level", "academic_work_impact"),
]


def base_features(df):
    raw = df.drop(columns=[ID, TARGET], errors="ignore")
    x = raw.select_dtypes(include="number").astype("float32").copy()
    miss = raw.isna()
    x["missing_count"] = miss.sum(axis=1).astype("int8")
    # A compact integer fingerprint lets trees separate common missing regimes.
    bits = np.zeros(len(raw), dtype="int32")
    for j, c in enumerate(raw.columns): bits |= miss[c].to_numpy("int32") << j
    x["missing_pattern"] = bits
    for c in raw.columns: x[c + "__missing"] = miss[c].astype("int8")
    comp = raw[["social_media_hours", "gaming_hours", "work_study_hours"]].sum(axis=1, min_count=3)
    x["component_sum"] = comp
    x["screen_residual"] = raw.daily_screen_time_hours - comp
    x["weekend_residual"] = raw.weekend_screen_time - comp
    x["weekend_gap"] = raw.weekend_screen_time - raw.daily_screen_time_hours
    x["screen_sleep"] = raw.daily_screen_time_hours / (raw.sleep_hours + .25)
    x["social_share"] = raw.social_media_hours / (raw.daily_screen_time_hours + .25)
    x["gaming_share"] = raw.gaming_hours / (raw.daily_screen_time_hours + .25)
    x["work_share"] = raw.work_study_hours / (raw.daily_screen_time_hours + .25)
    x["notif_open"] = raw.notifications_per_day / (raw.app_opens_per_day + 2.)
    return x.replace([np.inf, -np.inf], np.nan)


def as_key(s, mode):
    if mode == "exact": return s.astype("string").fillna("__NA__")
    z = pd.to_numeric(s, errors="coerce")
    if mode == "r1": z = z.round(1)
    elif mode == "floor": z = np.floor(z)
    return z.astype("string").fillna("__NA__")


def keys(df):
    raw = df.drop(columns=[ID, TARGET], errors="ignore")
    out = {}
    for c in raw:
        out[f"single_exact__{c}"] = as_key(raw[c], "exact")
        if pd.api.types.is_numeric_dtype(raw[c]):
            out[f"single_r1__{c}"] = as_key(raw[c], "r1")
            out[f"single_floor__{c}"] = as_key(raw[c], "floor")
    for a, b in PAIR_SPECS:
        for mode in ("exact", "r1", "floor"):
            out[f"pair_{mode}__{a}__{b}"] = as_key(raw[a], mode) + "|" + as_key(raw[b], mode)
    return pd.DataFrame(out)


def map_stats(fit_key, apply_key, y, smooth):
    prior = float(np.mean(y))
    stats = pd.DataFrame({"k":fit_key.to_numpy(), "y":y}).groupby("k", sort=False).y.agg(["sum","count"])
    sums = apply_key.map(stats["sum"]).fillna(0).to_numpy()
    counts = apply_key.map(stats["count"]).fillna(0).to_numpy()
    return ((sums+smooth*prior)/(counts+smooth)).astype("float32"), np.log1p(counts).astype("float32")


def evidence(fit_keys, val_keys, y, seed):
    """Four-fold inner-OOF fit features and complete-fit validation maps."""
    xf, xv = {}, {}
    inner = list(StratifiedKFold(4, shuffle=True, random_state=seed).split(fit_keys, y))
    for c in fit_keys:
        smooth = 60. if c.startswith("pair_") else (35. if "exact" in c else 50.)
        fk, vk = fit_keys[c], val_keys[c]
        te = np.zeros(len(fk), "float32"); fr = np.zeros(len(fk), "float32")
        for ia, ib in inner:
            te[ib], fr[ib] = map_stats(fk.iloc[ia], fk.iloc[ib], y[ia], smooth)
        xf[c + "__te"], xf[c + "__fr"] = te, fr
        xv[c + "__te"], xv[c + "__fr"] = map_stats(fk, vk, y, smooth)
    return pd.DataFrame(xf), pd.DataFrame(xv)


def model(seed):
    # This is a relative feature screen, so a faster matched tree budget is
    # preferable to six publication-sized fits.
    return lgb.LGBMClassifier(objective="binary", n_estimators=900, learning_rate=.06,
        num_leaves=31, min_child_samples=160, subsample=.86, subsample_freq=1,
        colsample_bytree=.88, reg_alpha=.15, reg_lambda=3., max_bin=255,
        random_state=seed, n_jobs=-1, verbosity=-1)


train = pd.read_csv(ROOT / "train.csv"); y = train[TARGET].to_numpy("int8")
base, all_keys = base_features(train), keys(train)
exact_key_names = [c for c in all_keys if c.startswith("single_exact")]
single_key_names = [c for c in all_keys if c.startswith("single_")]
variants = {"exact_single": exact_key_names,
            "coarse_single_plus_pairs": list(all_keys.columns)}
records = []; started = time.time(); OUT.mkdir(parents=True, exist_ok=True)

for split_seed in SEEDS:
    fi, vi = train_test_split(np.arange(len(train)), test_size=.18, stratify=y, random_state=split_seed)
    ef, ev = evidence(all_keys.iloc[fi].reset_index(drop=True), all_keys.iloc[vi].reset_index(drop=True), y[fi], split_seed)
    bfit, bval = base.iloc[fi].reset_index(drop=True), base.iloc[vi].reset_index(drop=True)
    for name, selected_keys in variants.items():
        cols = sum(([c+"__te", c+"__fr"] for c in selected_keys), [])
        xf = pd.concat([bfit, ef[cols]], axis=1); xv = pd.concat([bval, ev[cols]], axis=1)
        m = model(split_seed); m.fit(xf, y[fi], eval_set=[(xv, y[vi])], eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)])
        pred = m.predict_proba(xv)[:, 1]
        row = {"split_seed":split_seed, "variant":name, "auc":float(roc_auc_score(y[vi], pred)),
               "best_iteration":int(m.best_iteration_), "features":xf.shape[1]}
        records.append(row); print(row, flush=True)
        del xf, xv, m, pred; gc.collect()
    del ef, ev; gc.collect()

report = pd.DataFrame(records)
report.to_csv(OUT / "matched_holdout_results.csv", index=False)
summary = report.groupby("variant").auc.agg(["mean", "std", "min", "max"]).sort_values("mean", ascending=False)
summary.to_csv(OUT / "matched_holdout_summary.csv")
(OUT / "manifest.json").write_text(json.dumps({"protocol":"two matched stratified holdouts; four-fold inner-OOF fit evidence",
    "split_seeds":SEEDS,"variants":{k:len(v) for k,v in variants.items()},"runtime_seconds":time.time()-started},indent=2))
print(summary)
