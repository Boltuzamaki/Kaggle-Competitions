"""Fold-safe OOF target encoding at multiple smoothing strengths.

The project's existing nested-TE trees use a narrow key set. The public no-blend
LightGBM notebook uses a much richer view (missing pattern, composition shares,
intensity ratios, exact and rounded pair support) but builds its target
encodings on the same rows it trains on, so every training row sees its own
label. This experiment keeps the rich view and repairs the encoding: fit rows
receive inner out-of-fold rates, while outer-validation and test rows are mapped
from the complete outer-fit partition, exactly as at inference time.

This variant differs from the single-smoothing run in what it encodes rather
than how it trains. Each key is encoded at three smoothing strengths (10, 40,
150) instead of one. Low smoothing tracks a rare exact value closely and is
noisy; high smoothing shrinks hard toward the prior and is stable. Giving the
tree all three lets it choose per split how much to trust a key's own rate,
which a single fixed strength cannot express.

Two prespecified XGBoost configurations are screened on a holdout contained
wholly inside each outer-fit partition; the winner is refit on the full outer-fit
partition before the untouched outer fold is scored.

Official competition data only. No public predictions, no external data, and no
submission call.
"""
from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

TARGET, ID = "addicted_label", "id"
SEED, OUTER_FOLDS, INNER_FOLDS = 20260811, 5, 5
SMOOTHINGS = (10.0, 40.0, 150.0)
OUT = Path("/kaggle/working") if Path("/kaggle").exists() else Path(__file__).parent / "output"

PAIR_COLUMNS = [
    ("daily_screen_time_hours", "weekend_screen_time"),
    ("daily_screen_time_hours", "social_media_hours"),
    ("daily_screen_time_hours", "sleep_hours"),
    ("social_media_hours", "gaming_hours"),
    ("notifications_per_day", "app_opens_per_day"),
    ("daily_screen_time_hours", "gaming_hours"),
    ("sleep_hours", "stress_level"),
]

CONFIGS = {
    "d7_regular": dict(max_depth=7, eta=0.022, subsample=0.85, colsample_bytree=0.55,
                       min_child_weight=48, reg_lambda=3.5, reg_alpha=0.25),
    "d9_strong": dict(max_depth=9, eta=0.018, subsample=0.80, colsample_bytree=0.45,
                      min_child_weight=110, reg_lambda=8.0, reg_alpha=0.6),
}
MAX_ROUNDS = 6000


def locate():
    for root in (Path("/kaggle/input"), Path(".."), Path(".")):
        if not root.exists():
            continue
        for p in root.rglob("train.csv"):
            try:
                cols = pd.read_csv(p, nrows=1).columns
            except Exception:
                continue
            if TARGET in cols and (p.parent / "test.csv").exists():
                return p, p.parent / "test.csv"
    raise FileNotFoundError("official train.csv/test.csv not found")


def base_features(frame):
    """Target-free view. Identical construction for train and test."""
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
    for column in x.select_dtypes(exclude="number").columns:
        x[column] = x[column].astype("category").cat.codes.astype("int16")
    return x.replace([np.inf, -np.inf], np.nan)


def key_of(frame, column, digits=None):
    """Stringified lookup key, optionally rounded.

    Rounding only applies to numeric columns; asking for it on a categorical is
    a no-op. pandas 2.x raises on `object.round()` while pandas 3.x silently
    ignores it, so the dtype check keeps behaviour identical on both.
    """
    s = frame[column]
    if digits is not None and pd.api.types.is_numeric_dtype(s):
        s = s.round(digits)
    return s.astype("string").fillna("__NA__")


def pair_key(frame, left, right, digits=None):
    return key_of(frame, left, digits) + "|" + key_of(frame, right, digits)


def rate_stats(keys, y):
    return pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k").y.agg(["sum", "count"])


def rate_map(stats, smoothing, prior):
    """Empirical-Bayes rate at one smoothing strength."""
    return (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)


def inner_folds(n, seed):
    """Label-independent, so 'no row sees its own label' is exact."""
    order = np.random.default_rng(seed).permutation(n)
    for k in range(INNER_FOLDS):
        va = order[k::INNER_FOLDS]
        yield np.setdiff1d(order, va, assume_unique=True), va


def build(fit_raw, fit_y, apply_raw, apply_base, raw_columns, inner_oof):
    """Attach support and target-rate columns to `apply_base`."""
    out = apply_base.copy()
    prior = float(np.mean(fit_y))

    for column in raw_columns:
        counts = key_of(fit_raw, column).value_counts()
        out[column + "__logfreq"] = np.log1p(key_of(apply_raw, column).map(counts).fillna(0)).astype("float32")
        if pd.api.types.is_numeric_dtype(fit_raw[column]):
            rc = key_of(fit_raw, column, 1).value_counts()
            out[column + "__rlogfreq"] = np.log1p(
                key_of(apply_raw, column, 1).map(rc).fillna(0)).astype("float32")
    for left, right in PAIR_COLUMNS:
        for tag, digits in (("exact", None), ("rounded", 1)):
            counts = pair_key(fit_raw, left, right, digits).value_counts()
            out[f"{left}__{right}__{tag}_lf"] = np.log1p(
                pair_key(apply_raw, left, right, digits).map(counts).fillna(0)).astype("float32")

    specs = [(c,) for c in raw_columns] + [p for p in PAIR_COLUMNS]
    for spec in specs:
        stem = "__".join(spec)
        fk = key_of(fit_raw, spec[0]) if len(spec) == 1 else pair_key(fit_raw, *spec)
        if inner_oof:
            values = {s: np.full(len(apply_raw), np.nan, dtype="float32") for s in SMOOTHINGS}
            for tr_i, va_i in inner_folds(len(fk), SEED):
                stats = rate_stats(fk.iloc[tr_i], fit_y[tr_i])
                sub = fk.iloc[va_i]
                for sm in SMOOTHINGS:
                    values[sm][va_i] = sub.map(rate_map(stats, sm, prior)).fillna(prior).to_numpy("float32")
            for sm in SMOOTHINGS:
                out[f"{stem}__te{int(sm)}"] = values[sm]
        else:
            stats = rate_stats(fk, fit_y)
            ak = key_of(apply_raw, spec[0]) if len(spec) == 1 else pair_key(apply_raw, *spec)
            for sm in SMOOTHINGS:
                out[f"{stem}__te{int(sm)}"] = ak.map(rate_map(stats, sm, prior)).fillna(prior).astype("float32")
    return out


def main():
    started = time.time()
    train_path, test_path = locate()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    raw_columns = [c for c in train.columns if c not in (ID, TARGET)]
    y = train[TARGET].to_numpy("int8")
    train_base, test_base = base_features(train), base_features(test)
    train_raw = train[raw_columns]
    test_raw = test[raw_columns]

    gpu = {"device": "cuda", "tree_method": "hist"}
    try:
        xgb.QuantileDMatrix(np.zeros((4, 2), "float32"), label=np.array([0, 1, 0, 1]))
    except Exception:
        pass

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    records = []
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)

    for fold, (fit_i, val_i) in enumerate(outer.split(train_base, y), 1):
        fit_raw, val_raw = train_raw.iloc[fit_i], train_raw.iloc[val_i]
        fit_base, val_base = train_base.iloc[fit_i], train_base.iloc[val_i]
        fit_y, val_y = y[fit_i], y[val_i]

        x_fit = build(fit_raw, fit_y, fit_raw, fit_base, raw_columns, inner_oof=True)
        x_val = build(fit_raw, fit_y, val_raw, val_base, raw_columns, inner_oof=False)
        x_test = build(fit_raw, fit_y, test_raw, test_base, raw_columns, inner_oof=False)
        x_val = x_val[x_fit.columns]
        x_test = x_test[x_fit.columns]

        # Configuration and round count are chosen on a split carved out of the
        # outer-fit rows only; outer-validation labels never take part.
        in_tr, in_va = train_test_split(np.arange(len(fit_y)), test_size=0.18,
                                        random_state=SEED + fold, stratify=fit_y)
        d_in_tr = xgb.DMatrix(x_fit.iloc[in_tr], label=fit_y[in_tr])
        d_in_va = xgb.DMatrix(x_fit.iloc[in_va], label=fit_y[in_va])
        trials = []
        for tag, cfg in CONFIGS.items():
            params = {**cfg, **gpu, "objective": "binary:logistic", "eval_metric": "auc", "seed": SEED + fold}
            booster = xgb.train(params, d_in_tr, MAX_ROUNDS, evals=[(d_in_va, "va")],
                                early_stopping_rounds=150, verbose_eval=False)
            trials.append({"config": tag, "inner_auc": float(booster.best_score),
                           "rounds": int(booster.best_iteration) + 1})
            print(f"fold {fold} inner {tag}: auc {booster.best_score:.7f} @ {booster.best_iteration + 1}", flush=True)
            del booster
            gc.collect()
        del d_in_tr, d_in_va
        gc.collect()

        best = max(trials, key=lambda t: t["inner_auc"])
        params = {**CONFIGS[best["config"]], **gpu, "objective": "binary:logistic",
                  "eval_metric": "auc", "seed": SEED + fold}
        d_fit = xgb.DMatrix(x_fit, label=fit_y)
        booster = xgb.train(params, d_fit, best["rounds"], verbose_eval=False)
        oof[val_i] = booster.predict(xgb.DMatrix(x_val))
        test_pred += booster.predict(xgb.DMatrix(x_test)) / OUTER_FOLDS
        auc = roc_auc_score(val_y, oof[val_i])
        records.append({"fold": fold, "outer_auc": float(auc), "selected": best["config"],
                        "rounds": best["rounds"], "inner_trials": trials})
        print(f"fold {fold} OUTER AUC {auc:.8f} ({best['config']}, {best['rounds']} rounds) "
              f"[{time.time() - started:.0f}s]", flush=True)
        del booster, d_fit, x_fit, x_val, x_test
        gc.collect()

    pooled = float(roc_auc_score(y, oof))
    print(f"pooled OOF AUC {pooled:.10f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: train[ID], "fold": -1, "y": y, "pred": oof}).to_csv(
        OUT / "oof_foldsafe_te_xgb_multi.csv", index=False)
    pd.DataFrame({ID: test[ID], TARGET: test_pred}).to_csv(
        OUT / "test_foldsafe_te_xgb_multi.csv", index=False)
    (OUT / "metrics_foldsafe_te_xgb_multi.json").write_text(json.dumps({
        "model": "fold-safe OOF target encoding XGBoost, multi-smoothing (10/40/150)",
        "official_data_only": True,
        "public_predictions_used": False,
        "outer_folds": OUTER_FOLDS,
        "inner_encoding_folds": INNER_FOLDS,
        "smoothings": list(SMOOTHINGS),
        "own_label_excluded_from_fit_encodings": True,
        "selection": "two configs on an 18% split inside each outer-fit partition",
        "fold_records": records,
        "oof_auc": pooled,
        "runtime_seconds": time.time() - started,
        "submission_created": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
