"""Does any new feature group move a LightGBM on the fold-safe target-encoded view?

The feature view in this project has been stable for days. Every recent gain has
come from adding learners, not from adding columns, and the two most recent
attempts to change the view (E043d capacity, E045 multi-smoothing) both went
backwards. So the honest prior is that the view is finished.

This screen exists to test that prior rather than assume it. Four groups, each
measured against a byte-identical baseline on identical folds, each chosen
because nothing like it is currently in the view:

  peer      deviation of each headline hour column from the mean of its own age
            bucket and its own stress level. Every existing ratio is within-row:
            leisure over screen time, screen over sleep. Nothing tells the model
            whether six hours is ordinary or extreme *for this kind of person*,
            and a tree cannot derive a group mean from raw columns however deep
            it goes.
  rank      the percentile of each numeric column over train and test together.
            No label is read, so this is safe to compute transductively. It gives
            the model a monotone-invariant view of position in the distribution.
  residual  `unaccounted_screen` is currently screen time minus the literal sum
            of its parts, which assumes the parts should add up with unit
            weights. This fits those weights instead, out of fold, and keeps the
            residual. If the generator did not compose the columns additively,
            the fitted residual is the better anomaly signal.
  dup       how many rows share this row's entire raw feature vector, and how
            many share it with one column held out. The lookup transformer's
            whole advantage is that it reads exact values, and it is the only
            family decorrelated from the trees. This asks whether a slice of that
            advantage can be handed to a tree as an ordinary column.

A group is worth building a real stream around only if it clears the noise of
this screen. The 5-fold estimate here is far noisier than the stack's own
cross-fitted number, so the bar is deliberately coarse: anything under about
+0.0002 is not worth a follow-up.

Deliberately not tested: three-way target-encoded keys. E043 measured 55% of
two-column keys already holding a single row, so a three-column key is almost
entirely singletons, and the two attempts to encode sparse keys self-inclusively
scored 0.87 and 0.84. The ledger already answered that one.

Official competition data only. No public predictions and no submission call.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET, ID = "addicted_label", "id"
SEED, INNER_FOLDS = 20260807, 5
SMOOTHING = 40.0
GROUPS = ["peer", "rank", "residual", "dup"]

HOUR_COLUMNS = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
                "gaming_hours", "work_study_hours", "sleep_hours"]
PART_COLUMNS = ["social_media_hours", "gaming_hours", "work_study_hours", "sleep_hours"]

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
    """Fold-safe encoder, identical contract to cpu_kernel_lgb10fold."""
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


# --- the candidate feature groups ----------------------------------------
#
# Each takes the raw fit partition and the raw partition to encode, and returns a
# DataFrame indexed like `apply_raw`. The `fit_raw` argument is what keeps them
# fold-safe: every statistic is computed on the fit rows and only *applied* to
# the rows being encoded, exactly as the target encoder does.


def group_peer(fit_raw, apply_raw):
    """Deviation from the mean of the row's own age bucket and stress level."""
    out = {}
    age_bucket_fit = (fit_raw["age"] // 5).astype("Int64")
    age_bucket_apply = (apply_raw["age"] // 5).astype("Int64")
    for column in HOUR_COLUMNS:
        for tag, fk, ak in (("age", age_bucket_fit, age_bucket_apply),
                            ("stress", fit_raw["stress_level"], apply_raw["stress_level"])):
            means = fit_raw.groupby(fk, observed=True)[column].mean()
            stds = fit_raw.groupby(fk, observed=True)[column].std()
            m = ak.map(means).astype("float32")
            s = ak.map(stds).astype("float32")
            out[f"{column}__{tag}_dev"] = (apply_raw[column] - m).to_numpy("float32")
            # A z-score as well as a raw deviation: the spread of hours differs a
            # lot between cohorts, so the same deviation is not equally unusual.
            out[f"{column}__{tag}_z"] = ((apply_raw[column] - m) / (s + 1e-3)).to_numpy("float32")
    return pd.DataFrame(out, index=apply_raw.index)


def group_rank(fit_raw, apply_raw):
    """Percentile position of each numeric column within the fit distribution."""
    out = {}
    for column in fit_raw.select_dtypes(include="number").columns:
        reference = np.sort(fit_raw[column].dropna().to_numpy("float64"))
        if len(reference) == 0:
            continue
        values = apply_raw[column].to_numpy("float64")
        pos = np.searchsorted(reference, values) / len(reference)
        pos[np.isnan(values)] = np.nan
        out[column + "__pct"] = pos.astype("float32")
    return pd.DataFrame(out, index=apply_raw.index)


def group_residual(fit_raw, apply_raw):
    """Screen time minus a fitted linear combination of its parts.

    `unaccounted_screen` already subtracts the parts with unit weights. Fitting
    the weights on the fit rows asks whether the generator actually composed the
    columns that way; if it did, this is a noisier copy of an existing column and
    the screen will show nothing.
    """
    out = {}
    for target_column in ("daily_screen_time_hours", "weekend_screen_time"):
        fit_x = fit_raw[PART_COLUMNS].to_numpy("float64")
        fit_t = fit_raw[target_column].to_numpy("float64")
        ok = np.isfinite(fit_x).all(axis=1) & np.isfinite(fit_t)
        if ok.sum() < 100:
            continue
        design = np.column_stack([fit_x[ok], np.ones(ok.sum())])
        coef, *_ = np.linalg.lstsq(design, fit_t[ok], rcond=None)
        apply_x = apply_raw[PART_COLUMNS].to_numpy("float64")
        predicted = np.column_stack([apply_x, np.ones(len(apply_x))]) @ coef
        out[target_column + "__fitted_residual"] = (
            apply_raw[target_column].to_numpy("float64") - predicted).astype("float32")
    return pd.DataFrame(out, index=apply_raw.index)


def group_dup(fit_raw, apply_raw):
    """How many fit rows share this row's whole vector, or all but one column."""
    out = {}
    columns = list(fit_raw.columns)
    full_fit = key_of(fit_raw, columns[0])
    full_apply = key_of(apply_raw, columns[0])
    for column in columns[1:]:
        full_fit = full_fit + "|" + key_of(fit_raw, column)
        full_apply = full_apply + "|" + key_of(apply_raw, column)
    counts = full_fit.value_counts()
    out["dup_full"] = np.log1p(full_apply.map(counts).fillna(0)).to_numpy("float32")

    # Leave-one-column-out keys: a row that matches many others on everything
    # except its screen time is a different kind of near-duplicate than one that
    # matches on everything except its age.
    for held in HOUR_COLUMNS:
        rest = [c for c in columns if c != held]
        fk = key_of(fit_raw, rest[0])
        ak = key_of(apply_raw, rest[0])
        for column in rest[1:]:
            fk = fk + "|" + key_of(fit_raw, column)
            ak = ak + "|" + key_of(apply_raw, column)
        counts = fk.value_counts()
        out[f"dup_without__{held}"] = np.log1p(ak.map(counts).fillna(0)).to_numpy("float32")
    return pd.DataFrame(out, index=apply_raw.index)


BUILDERS = {"peer": group_peer, "rank": group_rank,
            "residual": group_residual, "dup": group_dup}


def run_config(group, train_raw, train_base, y, raw_columns, splits, params, rounds):
    """Cross-fitted OOF AUC for the baseline view plus one candidate group."""
    import lightgbm as lgb

    oof = np.zeros(len(y))
    for fold, (fit_i, val_i) in enumerate(splits, 1):
        fit_raw, fit_y = train_raw.iloc[fit_i], y[fit_i]
        x_fit = build(fit_raw, fit_y, fit_raw, train_base.iloc[fit_i], raw_columns, True)
        x_val = build(fit_raw, fit_y, train_raw.iloc[val_i], train_base.iloc[val_i],
                      raw_columns, False)[x_fit.columns]
        if group is not None:
            extra_fit = BUILDERS[group](fit_raw, fit_raw)
            extra_val = BUILDERS[group](fit_raw, train_raw.iloc[val_i])
            x_fit = pd.concat([x_fit, extra_fit.set_index(x_fit.index)], axis=1)
            x_val = pd.concat([x_val, extra_val.set_index(x_val.index)], axis=1)[x_fit.columns]
        model = lgb.train(params, lgb.Dataset(x_fit, label=fit_y), num_boost_round=rounds)
        oof[val_i] = model.predict(x_val)
        print(f"    fold {fold} auc {roc_auc_score(y[val_i], oof[val_i]):.7f}", flush=True)
    return float(roc_auc_score(y, oof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=1200)
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else (
        Path("/kaggle/working") if Path("/kaggle").exists()
        else Path(__file__).parent / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    train_path, _ = locate()
    train = pd.read_csv(train_path)
    if args.rows:
        per_class = max(1, args.rows // 2)
        take = np.concatenate([
            np.flatnonzero(train[TARGET].to_numpy() == c)[:per_class] for c in (0, 1)])
        train = train.iloc[np.sort(take)].reset_index(drop=True)

    raw_columns = [c for c in train.columns if c not in (ID, TARGET)]
    y = train[TARGET].to_numpy("int8")
    train_base = base_features(train)
    train_raw = train[raw_columns]
    splits = list(StratifiedKFold(args.folds, shuffle=True,
                                  random_state=SEED).split(train_base, y))

    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
              "num_leaves": 31, "min_data_in_leaf": 200, "feature_fraction": 0.7,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
              "num_threads": args.threads, "verbosity": -1, "seed": SEED}

    print("[fe_screen] baseline", flush=True)
    baseline = run_config(None, train_raw, train_base, y, raw_columns, splits,
                          params, args.rounds)
    print(f"[fe_screen] baseline OOF {baseline:.7f} ({time.time() - started:.0f}s)", flush=True)

    results = {"baseline": baseline}
    for group in [g for g in args.groups.split(",") if g]:
        print(f"[fe_screen] + {group}", flush=True)
        score = run_config(group, train_raw, train_base, y, raw_columns, splits,
                           params, args.rounds)
        results[group] = score
        print(f"[fe_screen] {group} OOF {score:.7f}  delta {score - baseline:+.7f} "
              f"({time.time() - started:.0f}s)", flush=True)

    ranked = sorted(((k, v) for k, v in results.items() if k != "baseline"),
                    key=lambda kv: -kv[1])
    print("\n[fe_screen] summary")
    for name, score in ranked:
        verdict = "follow up" if score - baseline > 2e-4 else "not worth a stream"
        print(f"  {name:<10} {score:.7f}  {score - baseline:+.7f}  {verdict}")

    (out_dir / "metrics_fe_screen.json").write_text(json.dumps({
        "model": "fe_screen",
        "official_data_only": True,
        "public_predictions_used": False,
        "folds": args.folds,
        "rounds": args.rounds,
        "baseline_oof": baseline,
        "group_oof": {k: v for k, v in results.items() if k != "baseline"},
        "group_delta": {k: v - baseline for k, v in results.items() if k != "baseline"},
        "follow_up_threshold": 2e-4,
        "runtime_seconds": time.time() - started,
        "submission_created": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    if Path("/kaggle").exists():
        sys.argv = ["experiment", "--threads", "4"]
    main()
