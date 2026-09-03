"""CPU model families chosen for decorrelation rather than standalone score.

The stream inventory is saturated on boosted trees: every GBDT in the library sits
above 0.99 rank correlation with every other, and the last CatBoost variant added
0.0000059 to the blend for 26 GPU minutes. Meanwhile the biggest recent jump came
from a model that was weaker standalone but wrong in different places.

So these families are picked for having a different inductive bias, not for their
own AUC:

  linear        additive in the encoded space, no interactions at all
  extratrees    bagged randomised trees, variance reduction instead of bias fitting
  randomforest  bagged trees with real split search
  lgb10fold     the more-folds lever, which was worth +0.00018 on XGBoost
  lgbdart       dropout boosting, which fits a different sequence of residuals

They all share the validated fold-safe encoder so any difference in the outputs
comes from the learner and not from the features.

Official competition data only. No public predictions and no submission call.
"""
from pathlib import Path
import argparse
import sys
import gc
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET, ID = "addicted_label", "id"
SEED, INNER_FOLDS = 20260807, 5
SMOOTHING = 40.0

PAIR_COLUMNS = [
    ("daily_screen_time_hours", "weekend_screen_time"),
    ("daily_screen_time_hours", "social_media_hours"),
    ("daily_screen_time_hours", "sleep_hours"),
    ("social_media_hours", "gaming_hours"),
    ("notifications_per_day", "app_opens_per_day"),
    ("daily_screen_time_hours", "gaming_hours"),
    ("sleep_hours", "stress_level"),
]


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
    s = frame[column]
    if digits is not None and pd.api.types.is_numeric_dtype(s):
        s = s.round(digits)
    return s.astype("string").fillna("__NA__")


def pair_key(frame, left, right, digits=None):
    return key_of(frame, left, digits) + "|" + key_of(frame, right, digits)


def rate_map(keys, y, prior):
    stats = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k").y.agg(["sum", "count"])
    return (stats["sum"] + SMOOTHING * prior) / (stats["count"] + SMOOTHING)


def inner_folds(n, seed):
    order = np.random.default_rng(seed).permutation(n)
    for k in range(INNER_FOLDS):
        va = order[k::INNER_FOLDS]
        yield np.setdiff1d(order, va, assume_unique=True), va


def build(fit_raw, fit_y, apply_raw, apply_base, raw_columns, inner_oof):
    """Fold-safe encoder. Columns are collected and joined once to avoid the
    quadratic reallocation that repeated inserts cause at this width."""
    prior = float(np.mean(fit_y))
    cols = {}
    for column in raw_columns:
        counts = key_of(fit_raw, column).value_counts()
        cols[column + "__logfreq"] = np.log1p(
            key_of(apply_raw, column).map(counts).fillna(0)).to_numpy("float32")
        if pd.api.types.is_numeric_dtype(fit_raw[column]):
            rc = key_of(fit_raw, column, 1).value_counts()
            cols[column + "__rlogfreq"] = np.log1p(
                key_of(apply_raw, column, 1).map(rc).fillna(0)).to_numpy("float32")
    for left, right in PAIR_COLUMNS:
        for tag, digits in (("exact", None), ("rounded", 1)):
            counts = pair_key(fit_raw, left, right, digits).value_counts()
            cols[f"{left}__{right}__{tag}_lf"] = np.log1p(
                pair_key(apply_raw, left, right, digits).map(counts).fillna(0)).to_numpy("float32")

    specs = [(c,) for c in raw_columns] + [p for p in PAIR_COLUMNS]
    for spec in specs:
        name = "__".join(spec) + "__te"
        fk = key_of(fit_raw, spec[0]) if len(spec) == 1 else pair_key(fit_raw, *spec)
        if inner_oof:
            values = np.full(len(apply_raw), np.nan, dtype="float32")
            for tr_i, va_i in inner_folds(len(fk), SEED):
                r = rate_map(fk.iloc[tr_i], fit_y[tr_i], prior)
                values[va_i] = fk.iloc[va_i].map(r).fillna(prior).to_numpy("float32")
            cols[name] = values
        else:
            ak = key_of(apply_raw, spec[0]) if len(spec) == 1 else pair_key(apply_raw, *spec)
            cols[name] = ak.map(rate_map(fk, fit_y, prior)).fillna(prior).to_numpy("float32")
    return pd.concat([apply_base, pd.DataFrame(cols, index=apply_base.index)], axis=1)


# --- learners -------------------------------------------------------------

def fit_linear(x_fit, fit_y, x_val, x_test, fold, threads):
    """Additive logistic model. It cannot represent a single interaction, which
    is the entire point: its errors have a different shape from any tree."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, QuantileTransformer
    from sklearn.impute import SimpleImputer

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                            subsample=300000, random_state=SEED + fold),
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=3000, solver="lbfgs"),
    )
    model.fit(x_fit, fit_y)
    return (model.predict_proba(x_val)[:, 1], model.predict_proba(x_test)[:, 1])


def _forest(kind, threads, fold, **over):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    cls = ExtraTreesClassifier if kind == "extratrees" else RandomForestClassifier
    kwargs = dict(n_estimators=400, max_features=0.35, min_samples_leaf=25,
                  n_jobs=threads, random_state=SEED + fold, bootstrap=True)
    kwargs.update(over)
    return cls(**kwargs)


def fit_forest(kind, **over):
    def inner(x_fit, fit_y, x_val, x_test, fold, threads):
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy="median").fit(x_fit)
        model = _forest(kind, threads, fold, **over)
        model.fit(imp.transform(x_fit), fit_y)
        return (model.predict_proba(imp.transform(x_val))[:, 1],
                model.predict_proba(imp.transform(x_test))[:, 1])
    return inner


def fit_lgb(extra):
    def inner(x_fit, fit_y, x_val, x_test, fold, threads):
        import lightgbm as lgb
        params = {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
                  "num_leaves": 31, "min_data_in_leaf": 160, "bagging_fraction": 0.85,
                  "bagging_freq": 1, "feature_fraction": 0.6, "lambda_l1": 0.15,
                  "lambda_l2": 2.5, "verbosity": -1, "seed": SEED + fold,
                  "num_threads": threads, **extra}
        rounds = params.pop("_rounds", 2600)
        booster = lgb.train(params, lgb.Dataset(x_fit, label=fit_y), num_boost_round=rounds)
        return booster.predict(x_val), booster.predict(x_test)
    return inner


MODELS = {
    "linear": (fit_linear, 5),
    "extratrees": (fit_forest("extratrees"), 5),
    "randomforest": (fit_forest("randomforest"), 5),
    "lgb10fold": (fit_lgb({}), 10),
    "lgbdart": (fit_lgb({"boosting": "dart", "drop_rate": 0.1, "skip_drop": 0.5, "_rounds": 1800}), 5),
    # The paired audit settled the question these families were built to ask:
    # extratrees at 0.9617 contributed t=+5.8 while the 10-fold LightGBM at
    # 0.9682 contributed t=+2.8 and was dropped. Randomised trees are the CPU
    # family worth pushing, so this one gets finer leaves, more features per
    # split and the ten-fold treatment that was worth +0.00018 elsewhere.
    "et_deep": (fit_forest("extratrees", n_estimators=400, max_features=0.45,
                           min_samples_leaf=12), 10),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fit_predict, n_folds = MODELS[args.model]
    out_dir = Path(args.out) if args.out else (
        Path("/kaggle/working") if Path("/kaggle").exists()
        else Path(__file__).parent / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    train_path, test_path = locate()
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    raw_columns = [c for c in train.columns if c not in (ID, TARGET)]
    y = train[TARGET].to_numpy("int8")
    train_base, test_base = base_features(train), base_features(test)
    train_raw, test_raw = train[raw_columns], test[raw_columns]

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    records = []
    outer = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)

    for fold, (fit_i, val_i) in enumerate(outer.split(train_base, y), 1):
        fit_raw, fit_y = train_raw.iloc[fit_i], y[fit_i]
        x_fit = build(fit_raw, fit_y, fit_raw, train_base.iloc[fit_i], raw_columns, True)
        x_val = build(fit_raw, fit_y, train_raw.iloc[val_i], train_base.iloc[val_i],
                      raw_columns, False)[x_fit.columns]
        x_test = build(fit_raw, fit_y, test_raw, test_base, raw_columns, False)[x_fit.columns]

        val_pred, test_part = fit_predict(x_fit, fit_y, x_val, x_test, fold, args.threads)
        oof[val_i] = val_pred
        test_pred += test_part / n_folds
        auc = roc_auc_score(y[val_i], val_pred)
        records.append({"fold": fold, "outer_auc": float(auc)})
        print(f"[{args.model}] fold {fold} AUC {auc:.8f} ({time.time()-started:.0f}s)", flush=True)
        del x_fit, x_val, x_test
        gc.collect()

    pooled = float(roc_auc_score(y, oof))
    print(f"[{args.model}] pooled OOF AUC {pooled:.10f}", flush=True)
    tag = args.model
    pd.DataFrame({ID: train[ID], "fold": -1, "y": y, "pred": oof}).to_csv(
        out_dir / f"oof_{tag}.csv", index=False)
    pd.DataFrame({ID: test[ID], TARGET: test_pred}).to_csv(
        out_dir / f"test_{tag}.csv", index=False)
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps({
        "model": tag,
        "official_data_only": True,
        "public_predictions_used": False,
        "outer_folds": n_folds,
        "inner_encoding_folds": INNER_FOLDS,
        "fold_records": records,
        "oof_auc": pooled,
        "runtime_seconds": time.time() - started,
        "submission_created": False,
    }, indent=2) + "\n")



if __name__ == "__main__":
    # Kaggle runs this script with no arguments, so the configuration is fixed
    # here rather than passed on the command line.
    sys.argv = ["experiment", "--model", "extratrees", "--threads", "4"]
    main()
