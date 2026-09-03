#!/usr/bin/env python3
"""Screen candidate upgrades for the public no-blend LightGBM notebook.

The published version 3 recipe (public LB 0.96949) builds its exact-value target
encodings by fitting on the same rows it trains on, so every training row sees
its own label through `__te`. The original motivation for this script was that
this looked like a leak worth repairing.

The measurement says otherwise, and the result is recorded here so it is not
re-litigated. Under an identical three-fold screening protocol:

    v0_published (self-including TE)   0.96768759
    v1_oof_te    (inner out-of-fold)   0.96755293   -0.00013
    v2_oof_pair_te                     0.96754486   -0.00014

v0 wins on every fold, by more than the 0.00009 fold standard deviation. At
691k rows an exact-value key holds thousands of rows, so a single row's own
label barely moves its own rate and the leak is negligible. Inner out-of-fold
encoding, meanwhile, builds training features from 80% of the fit rows while
test features are always mapped from all of train. That distribution mismatch
costs more than the leak. The published recipe's virtue is that train and test
features come from the same mapping.

A 30k-row smoke test showed the opposite (+0.0039 for the fix) because at that
scale each key holds few rows and the leak really does dominate. Small-scale
smoke tests are execution checks here, not evidence about full-scale ranking.

The remaining variants therefore keep the published encoding and vary what is
encoded and how much capacity the model has.

Official competition data only. No public predictions and no submission call.
"""
from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "output"
TARGET = "addicted_label"
ID = "id"
SEED = 20260806
INNER_FOLDS = 5

PAIR_COLUMNS = [
    ("daily_screen_time_hours", "weekend_screen_time"),
    ("daily_screen_time_hours", "social_media_hours"),
    ("daily_screen_time_hours", "sleep_hours"),
    ("social_media_hours", "gaming_hours"),
    ("notifications_per_day", "app_opens_per_day"),
]

PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.025, num_leaves=31,
    min_data_in_leaf=160, bagging_fraction=0.85, bagging_freq=1,
    feature_fraction=0.86, lambda_l1=0.15, lambda_l2=2.5, max_bin=255,
    verbosity=-1,
    # Shared box: the stacker and other projects are also on these cores.
    num_threads=int(os.environ.get("LGB_THREADS", "8")),
)
ROUNDS = 3654


def make_base_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.drop(columns=[ID, TARGET], errors="ignore").copy()
    numeric = list(x.select_dtypes(include="number").columns)
    x["missing_count"] = x.isna().sum(axis=1).astype("int8")
    pattern = np.zeros(len(x), dtype="int32")
    for bit, column in enumerate(x.columns):
        pattern |= x[column].isna().to_numpy("int32") << bit
    x["missing_pattern"] = pattern
    for column in numeric:
        x[column + "__missing"] = x[column].isna().astype("int8")
    x["leisure_hours"] = x["social_media_hours"] + x["gaming_hours"]
    x["accounted_hours"] = x["leisure_hours"] + x["work_study_hours"]
    x["unaccounted_screen"] = x["daily_screen_time_hours"] - x["accounted_hours"]
    x["weekend_unaccounted"] = x["weekend_screen_time"] - x["accounted_hours"]
    x["weekend_gap"] = x["weekend_screen_time"] - x["daily_screen_time_hours"]
    x["weekend_ratio"] = x["weekend_screen_time"] / (x["daily_screen_time_hours"] + 0.25)
    x["leisure_share"] = x["leisure_hours"] / (x["daily_screen_time_hours"] + 0.25)
    x["social_share"] = x["social_media_hours"] / (x["daily_screen_time_hours"] + 0.25)
    x["gaming_share"] = x["gaming_hours"] / (x["daily_screen_time_hours"] + 0.25)
    x["work_share"] = x["work_study_hours"] / (x["daily_screen_time_hours"] + 0.25)
    x["notif_per_screen"] = x["notifications_per_day"] / (x["daily_screen_time_hours"] + 0.25)
    x["opens_per_screen"] = x["app_opens_per_day"] / (x["daily_screen_time_hours"] + 0.25)
    x["notif_per_open"] = x["notifications_per_day"] / (x["app_opens_per_day"] + 2.0)
    x["sleep_screen_balance"] = x["sleep_hours"] - x["daily_screen_time_hours"]
    x["screen_sleep_ratio"] = x["daily_screen_time_hours"] / (x["sleep_hours"] + 0.25)
    return x.replace([np.inf, -np.inf], np.nan)


def all_pairs(raw_columns):
    """Every unordered pair of the twelve predictors: 66 keys instead of 5."""
    return [(a, b) for i, a in enumerate(raw_columns) for b in raw_columns[i + 1:]]


def key_of(frame: pd.DataFrame, column: str, digits=None) -> pd.Series:
    s = frame[column] if digits is None else frame[column].round(digits)
    return s.astype("string").fillna("__NA__")


def pair_key(frame: pd.DataFrame, left: str, right: str, digits=None) -> pd.Series:
    return key_of(frame, left, digits) + "|" + key_of(frame, right, digits)


def inner_folds(n: int, seed: int):
    """Label-independent inner folds over `n` rows."""
    order = np.random.default_rng(seed).permutation(n)
    for k in range(INNER_FOLDS):
        va = order[k::INNER_FOLDS]
        tr = np.setdiff1d(order, va, assume_unique=True)
        yield tr, va


def target_rate(keys: pd.Series, y: np.ndarray, smoothing: float, prior: float) -> pd.Series:
    stats = pd.DataFrame({"key": keys.to_numpy(), "y": y}).groupby("key").y.agg(["sum", "count"])
    return (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)


def support_features(fit_x, apply_x, raw_columns, out):
    """Label-free counts. Identical in every variant, so they never explain a
    difference between them."""
    for column in raw_columns:
        counts = key_of(fit_x, column).value_counts()
        out[column + "__logfreq"] = np.log1p(key_of(apply_x, column).map(counts).fillna(0)).astype("float32")
        if pd.api.types.is_numeric_dtype(fit_x[column]):
            rc = key_of(fit_x, column, 1).value_counts()
            out[column + "__rounded_logfreq"] = np.log1p(
                key_of(apply_x, column, 1).map(rc).fillna(0)).astype("float32")
    for left, right in PAIR_COLUMNS:
        for suffix, digits in (("exact", None), ("rounded", 1)):
            counts = pair_key(fit_x, left, right, digits).value_counts()
            out[f"{left}__{right}__{suffix}_logfreq"] = np.log1p(
                pair_key(apply_x, left, right, digits).map(counts).fillna(0)).astype("float32")
    return out


def encode(fit_x, fit_y, apply_x, raw_columns, variant, smoothing=40.0, inner_oof=False, seed=SEED):
    """Build the feature matrix for `apply_x` using only `fit_x`/`fit_y`.

    When `inner_oof` is set, `apply_x` must be `fit_x` itself and each row's
    target encoding is taken from a model fit on the other inner folds, so no
    row ever sees its own label.
    """
    out = apply_x.copy()
    prior = float(np.mean(fit_y))
    out = support_features(fit_x, apply_x, raw_columns, out)

    te_keys = [(c, None) for c in raw_columns]
    if variant in ("pair_te", "pair_te_wide"):
        te_keys += [((l, r), None) for l, r in PAIR_COLUMNS]
    elif variant == "all_pair_te":
        te_keys += [((l, r), None) for l, r in all_pairs(raw_columns)]

    for spec, _ in te_keys:
        if isinstance(spec, tuple):
            name = f"{spec[0]}__{spec[1]}__te"
            fk = pair_key(fit_x, *spec)
            ak = pair_key(apply_x, *spec)
        else:
            name = spec + "__te"
            fk = key_of(fit_x, spec)
            ak = key_of(apply_x, spec)

        if inner_oof:
            values = np.full(len(apply_x), np.nan, dtype="float32")
            # Fold assignment is drawn from a seeded permutation rather than a
            # stratified split so it does not depend on the labels at all. That
            # makes "no row sees its own label" an exact, testable property.
            for tr_i, va_i in inner_folds(len(fk), seed):
                rate = target_rate(fk.iloc[tr_i], fit_y[tr_i], smoothing, prior)
                values[va_i] = fk.iloc[va_i].map(rate).fillna(prior).to_numpy("float32")
            out[name] = values
        else:
            rate = target_rate(fk, fit_y, smoothing, prior)
            out[name] = ak.map(rate).fillna(prior).astype("float32")
    return out


def align_categoricals(a: pd.DataFrame, b: pd.DataFrame):
    cats = list(a.select_dtypes(exclude="number").columns)
    for column in cats:
        levels = pd.Index(pd.concat([a[column], b[column]], ignore_index=True)
                          .astype("string").fillna("Missing").unique())
        dtype = pd.CategoricalDtype(levels)
        a[column] = a[column].astype("string").fillna("Missing").astype(dtype)
        b[column] = b[column].astype("string").fillna("Missing").astype(dtype)
    return a, b, cats


VARIANTS = {
    # Reproduces the published recipe: training rows see their own label in __te.
    "v0_published": dict(inner_oof=False, variant="single_te", rounds=ROUNDS, params={}),
    # Same features, but training-row encodings are inner out-of-fold. Measured
    # WORSE than v0 at full scale: the leak is negligible once keys have large
    # counts, while inner-OOF makes training features differ in distribution
    # from test features, which are always mapped from the whole of train.
    "v1_oof_te": dict(inner_oof=True, variant="single_te", rounds=ROUNDS, params={}),
    "v2_oof_pair_te": dict(inner_oof=True, variant="pair_te", rounds=ROUNDS, params={}),
    "v3_oof_pair_wide": dict(inner_oof=True, variant="pair_te", rounds=5200,
                             params=dict(num_leaves=63, min_data_in_leaf=120, learning_rate=0.02)),
    # --- upgrades that keep the published encoding, since it won the screen ---
    # Five hand-picked pair target rates, encoded the way v0 encodes singles.
    "v4_pair_te": dict(inner_oof=False, variant="pair_te", rounds=ROUNDS, params={}),
    # All 66 pairs rather than the hand-picked five.
    "v5_all_pair_te": dict(inner_oof=False, variant="all_pair_te", rounds=ROUNDS, params={}),
    # More capacity on the published feature view.
    "v6_capacity": dict(inner_oof=False, variant="single_te", rounds=ROUNDS,
                        params=dict(num_leaves=63, min_data_in_leaf=90)),
}


def run_variant(tag: str, cfg: dict, train, test, raw_columns, x_train_base, x_test_base, folds):
    """Honest outer-fold OOF plus a fold-averaged test prediction.

    The test prediction makes each variant usable as an ensemble stream without
    a second training pass, and it is built under the same contract as the
    outer-validation features."""
    y = train[TARGET].to_numpy("int8")
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test)) if test is not None else None
    fold_aucs = []
    started = time.time()
    for fold, (fit_i, val_i) in enumerate(folds, 1):
        fit_base = x_train_base.iloc[fit_i]
        val_base = x_train_base.iloc[val_i]
        fit_y = y[fit_i]
        # Fit-row features may use inner OOF; validation features are always
        # mapped from the complete outer-fit partition, exactly as test will be.
        x_fit = encode(fit_base, fit_y, fit_base, raw_columns, cfg["variant"], inner_oof=cfg["inner_oof"])
        x_val = encode(fit_base, fit_y, val_base, raw_columns, cfg["variant"], inner_oof=False)
        x_fit, x_val, cats = align_categoricals(x_fit, x_val)
        params = {**PARAMS, **cfg["params"], "seed": SEED + fold}
        ds = lgb.Dataset(x_fit, label=fit_y, categorical_feature=cats, free_raw_data=True)
        booster = lgb.train(params, ds, num_boost_round=cfg["rounds"])
        oof[val_i] = booster.predict(x_val)
        auc = roc_auc_score(y[val_i], oof[val_i])
        fold_aucs.append(auc)
        print(f"  [{tag}] fold {fold} AUC {auc:.8f}  ({time.time()-started:.0f}s)", flush=True)
        if test_pred is not None:
            x_test = encode(fit_base, fit_y, x_test_base, raw_columns, cfg["variant"], inner_oof=False)
            x_test = x_test[x_fit.columns.tolist()]
            for column in cats:
                x_test[column] = x_test[column].astype("string").fillna("Missing").astype(x_fit[column].dtype)
            test_pred += booster.predict(x_test) / len(folds)
            del x_test
        del ds, booster, x_fit, x_val
    pooled = roc_auc_score(y, oof)
    print(f"[{tag}] pooled OOF {pooled:.10f}  fold std {np.std(fold_aucs):.8f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: train[ID], "fold": -1, "y": y, "pred": oof}).to_csv(OUT / f"oof_{tag}.csv", index=False)
    if test_pred is not None:
        pd.DataFrame({ID: test[ID], TARGET: test_pred}).to_csv(OUT / f"test_{tag}.csv", index=False)
    return {"variant": tag, "oof_auc": pooled, "fold_aucs": fold_aucs,
            "fold_std": float(np.std(fold_aucs)), "runtime_s": time.time() - started}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--folds", type=int, default=5)
    # Screening mode: the same number of effective trees at a higher learning
    # rate costs far less wall-clock on a shared box. Absolute AUC drops a
    # little, but every variant pays the same price, so the ranking - which is
    # the whole question - is preserved.
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--round-scale", type=float, default=1.0)
    ap.add_argument("--tag-suffix", default="")
    args = ap.parse_args()
    if args.lr is not None:
        PARAMS["learning_rate"] = args.lr
    for cfg in VARIANTS.values():
        cfg["rounds"] = max(200, int(cfg["rounds"] * args.round_scale))
    print(f"protocol: {args.folds} outer folds, lr={PARAMS['learning_rate']}, "
          f"round_scale={args.round_scale}, threads={PARAMS['num_threads']}", flush=True)

    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    raw_columns = [c for c in train.columns if c not in (ID, TARGET)]
    x_train_base = make_base_features(train)
    x_test_base = make_base_features(test)
    y = train[TARGET].to_numpy("int8")
    folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=SEED).split(x_train_base, y))

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    path = OUT / f"experiment_v4_results{args.tag_suffix}.json"
    if path.exists():
        results = json.loads(path.read_text())
    done = {r["variant"] for r in results}
    for tag in args.variants:
        if tag + args.tag_suffix in done:
            print(f"[{tag}] already recorded, skipping")
            continue
        print(f"\n=== {tag}: {VARIANTS[tag]}", flush=True)
        results.append(run_variant(tag + args.tag_suffix, VARIANTS[tag], train, test,
                                   raw_columns, x_train_base, x_test_base, folds))
        path.write_text(json.dumps(results, indent=2) + "\n")

    table = pd.DataFrame(results).sort_values("oof_auc", ascending=False)
    print("\n" + table[["variant", "oof_auc", "fold_std", "runtime_s"]].to_string(index=False))
    print("\nimplied public LB uses the project's measured offset of +0.00100:")
    for _, r in table.iterrows():
        print(f"  {r['variant']:<20} OOF {r['oof_auc']:.8f} -> LB ~{r['oof_auc'] + 0.001:.5f}")


if __name__ == "__main__":
    main()
