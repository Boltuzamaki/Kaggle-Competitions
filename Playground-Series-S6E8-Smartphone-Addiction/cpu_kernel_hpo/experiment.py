"""Hyperparameter search on the fold-safe target-encoded view, then an honest refit.

Every boosted-tree config in this project was hand-picked. `d7_regular` won all
five folds of E042 and the 31-leaf LightGBM was never compared against anything,
so the ledger has no evidence about how much of the remaining gap is model
capacity versus tuning. This kernel measures that.

The awkward part of tuning inside a stacking project is that the stack consumes
OOF predictions, so any hyperparameter chosen by looking at outer-validation rows
quietly inflates the number the stack is later weighted on. The protocol here is
the one the ledger already sanctions (E038): configurations are compared on a
holdout carved wholly inside each outer-fit partition, then the winner is refit
on that complete partition. No outer-validation row is ever scored by a model
whose configuration was chosen with that row's label.

Two phases:

  A. Broad random search on outer fold 1's fit partition against a 15% inner
     holdout. This is a shortlisting device only. Its scores are search scores,
     never reported as OOF, because selection and evaluation share a holdout.
  B. Honest ten-fold refit. Per outer fold, the shortlisted finalists (top two
     from phase A plus the incumbent hand-tuned config) are compared on a fresh
     16% inner holdout drawn from that fold's own fit rows, and the per-fold
     winner is refit on the full fit partition to predict validation and test.

Phase A runs under a wall-clock budget and phase B under a deadline, so the
kernel always reaches its artifact write within the twelve-hour limit.

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
from sklearn.model_selection import StratifiedKFold, train_test_split

LEARNER = "lgb"  # overwritten per-kernel by build_kernels.py

TARGET, ID = "addicted_label", "id"
SEED, INNER_FOLDS = 20260807, 5
SMOOTHING = 40.0
OUTER_FOLDS = 10

SEARCH_BUDGET_S = 3.5 * 3600
TOTAL_BUDGET_S = 10.5 * 3600
EARLY_STOP = 120
MAX_ROUNDS = 6000

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


# --- search space ---------------------------------------------------------

INCUMBENT_LGB = {
    "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 160,
    "bagging_fraction": 0.85, "bagging_freq": 1, "feature_fraction": 0.6,
    "lambda_l1": 0.15, "lambda_l2": 2.5, "max_depth": -1, "min_gain_to_split": 0.0,
}
INCUMBENT_XGB = {
    "eta": 0.03, "max_depth": 7, "min_child_weight": 20.0, "subsample": 0.85,
    "colsample_bytree": 0.6, "reg_lambda": 2.5, "reg_alpha": 0.15, "gamma": 0.0,
}


def sample_lgb(rng):
    return {
        "learning_rate": 0.03,
        "num_leaves": int(rng.integers(16, 256)),
        "min_data_in_leaf": int(np.exp(rng.uniform(np.log(20), np.log(600)))),
        "bagging_fraction": float(rng.uniform(0.6, 1.0)),
        "bagging_freq": 1,
        "feature_fraction": float(rng.uniform(0.35, 1.0)),
        "lambda_l1": float(np.exp(rng.uniform(np.log(1e-3), np.log(10.0)))),
        "lambda_l2": float(np.exp(rng.uniform(np.log(1e-2), np.log(60.0)))),
        "max_depth": int(rng.choice([-1, 6, 8, 10, 12])),
        "min_gain_to_split": float(rng.choice([0.0, 0.0, 0.01, 0.1])),
    }


def sample_xgb(rng):
    return {
        "eta": 0.03,
        "max_depth": int(rng.integers(5, 12)),
        "min_child_weight": float(np.exp(rng.uniform(np.log(1.0), np.log(200.0)))),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.35, 1.0)),
        "reg_lambda": float(np.exp(rng.uniform(np.log(0.1), np.log(60.0)))),
        "reg_alpha": float(np.exp(rng.uniform(np.log(1e-3), np.log(5.0)))),
        "gamma": float(rng.choice([0.0, 0.0, 0.05, 0.2])),
    }


def train_eval(params, x_fit, y_fit, x_val, y_val, threads, rounds=None,
               max_rounds=MAX_ROUNDS):
    """Fit and score. With `rounds=None` the round count is chosen by early
    stopping on `x_val`; with an explicit count nothing is selected here.

    `max_rounds` is tightened in phase B, where phase A has already shown
    roughly where each finalist converges and an open-ended ceiling on a
    255-leaf config would eat the fold budget."""
    if LEARNER == "lgb":
        import lightgbm as lgb
        p = {"objective": "binary", "metric": "auc", "verbosity": -1,
             "seed": SEED, "num_threads": threads, **params}
        ds_fit = lgb.Dataset(x_fit, label=y_fit)
        if rounds is None:
            ds_val = lgb.Dataset(x_val, label=y_val, reference=ds_fit)
            booster = lgb.train(p, ds_fit, num_boost_round=max_rounds,
                                valid_sets=[ds_val],
                                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
            return booster, int(booster.best_iteration)
        booster = lgb.train(p, ds_fit, num_boost_round=rounds)
        return booster, rounds

    import xgboost as xgb
    p = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
         "seed": SEED, "nthread": threads, **params}
    d_fit = xgb.DMatrix(x_fit, label=y_fit)
    if rounds is None:
        d_val = xgb.DMatrix(x_val, label=y_val)
        booster = xgb.train(p, d_fit, num_boost_round=max_rounds,
                            evals=[(d_val, "val")], early_stopping_rounds=EARLY_STOP,
                            verbose_eval=False)
        return booster, int(booster.best_iteration) + 1
    booster = xgb.train(p, d_fit, num_boost_round=rounds)
    return booster, rounds


def predict(booster, x):
    if LEARNER == "lgb":
        return booster.predict(x)
    import xgboost as xgb
    return booster.predict(xgb.DMatrix(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--folds", type=int, default=OUTER_FOLDS)
    ap.add_argument("--rows", type=int, default=0, help="smoke test row cap")
    ap.add_argument("--search-budget", type=float, default=SEARCH_BUDGET_S)
    ap.add_argument("--total-budget", type=float, default=TOTAL_BUDGET_S)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n_folds = args.folds
    out_dir = Path(args.out) if args.out else (
        Path("/kaggle/working") if Path("/kaggle").exists()
        else Path(__file__).parent / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"hpo_{LEARNER}"

    started = time.time()
    train_path, test_path = locate()
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    if args.rows:
        per_class = max(1, args.rows // 2)
        take = np.concatenate([
            np.flatnonzero(train[TARGET].to_numpy() == c)[:per_class] for c in (0, 1)])
        train = train.iloc[np.sort(take)].reset_index(drop=True)
        test = test.head(min(len(test), args.rows // 2)).reset_index(drop=True)

    raw_columns = [c for c in train.columns if c not in (ID, TARGET)]
    y = train[TARGET].to_numpy("int8")
    train_base, test_base = base_features(train), base_features(test)
    train_raw, test_raw = train[raw_columns], test[raw_columns]
    outer = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)
    outer_splits = list(outer.split(train_base, y))

    # --- phase A: shortlist on fold 1's fit partition only -----------------
    print(f"[{tag}] phase A: random search", flush=True)
    fit_i = outer_splits[0][0]
    s_fit, s_val = train_test_split(fit_i, test_size=0.15, random_state=SEED,
                                    stratify=y[fit_i])
    a_fit_raw, a_fit_y = train_raw.iloc[s_fit], y[s_fit]
    xa_fit = build(a_fit_raw, a_fit_y, a_fit_raw, train_base.iloc[s_fit], raw_columns, True)
    xa_val = build(a_fit_raw, a_fit_y, train_raw.iloc[s_val], train_base.iloc[s_val],
                   raw_columns, False)[xa_fit.columns]
    print(f"[{tag}] search view {xa_fit.shape} built ({time.time()-started:.0f}s)", flush=True)

    sampler = sample_lgb if LEARNER == "lgb" else sample_xgb
    incumbent = INCUMBENT_LGB if LEARNER == "lgb" else INCUMBENT_XGB
    rng = np.random.default_rng(SEED)
    trials = []
    trial_id = 0
    while time.time() - started < args.search_budget:
        params = incumbent.copy() if trial_id == 0 else sampler(rng)
        t0 = time.time()
        booster, best_rounds = train_eval(params, xa_fit, a_fit_y, xa_val, y[s_val],
                                          args.threads)
        auc = float(roc_auc_score(y[s_val], predict(booster, xa_val)))
        trials.append({"trial": trial_id, "search_auc": auc, "best_rounds": best_rounds,
                       "seconds": time.time() - t0, "params": params})
        print(f"[{tag}] trial {trial_id:>3} search AUC {auc:.7f} rounds {best_rounds:>4} "
              f"({time.time()-t0:.0f}s, elapsed {time.time()-started:.0f}s)", flush=True)
        trial_id += 1
        del booster
        gc.collect()

    trials.sort(key=lambda r: -r["search_auc"])
    pd.DataFrame([{**{k: v for k, v in t.items() if k != "params"},
                   **t["params"]} for t in trials]).to_csv(
        out_dir / f"search_{LEARNER}.csv", index=False)

    # Finalists: the two best *sampled* configs plus the incumbent, which is the
    # thing any winner has to actually beat. Trial 0 is the incumbent itself and
    # is excluded here, otherwise a strong incumbent occupies two of the three
    # finalist slots and a third of phase B is spent scoring it twice.
    finalists = [{"name": f"search_{t['trial']}", "params": t["params"],
                  "rounds": t["best_rounds"]}
                 for t in trials if t["trial"] != 0][:2]
    incumbent_trial = next((t for t in trials if t["trial"] == 0), None)
    finalists.append({"name": "incumbent", "params": incumbent,
                      "rounds": incumbent_trial["best_rounds"] if incumbent_trial else 2600})
    print(f"[{tag}] finalists: {[f['name'] for f in finalists]}", flush=True)
    del xa_fit, xa_val
    gc.collect()

    # --- phase B: honest ten-fold, per-fold selection inside the fit rows ---
    phase_b_start = time.time()
    print(f"[{tag}] phase B: honest {n_folds}-fold refit", flush=True)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    folds_done = 0
    records = []

    for fold, (fit_i, val_i) in enumerate(outer_splits, 1):
        remaining = args.total_budget - (time.time() - started)
        if folds_done and remaining < (time.time() - phase_b_start) / folds_done:
            print(f"[{tag}] stopping before fold {fold}: {remaining:.0f}s left", flush=True)
            break

        fit_raw, fit_y = train_raw.iloc[fit_i], y[fit_i]
        x_fit = build(fit_raw, fit_y, fit_raw, train_base.iloc[fit_i], raw_columns, True)
        x_val = build(fit_raw, fit_y, train_raw.iloc[val_i], train_base.iloc[val_i],
                      raw_columns, False)[x_fit.columns]
        x_test = build(fit_raw, fit_y, test_raw, test_base, raw_columns, False)[x_fit.columns]

        # Selection holdout is carved from this fold's own fit rows, so no
        # validation label takes any part in choosing the configuration.
        i_fit, i_val = train_test_split(np.arange(len(fit_i)), test_size=0.16,
                                        random_state=SEED + fold, stratify=fit_y)
        scored = []
        for cand in finalists:
            ceiling = min(MAX_ROUNDS, max(200, int(cand["rounds"] * 1.6)))
            booster, rounds = train_eval(cand["params"], x_fit.iloc[i_fit], fit_y[i_fit],
                                         x_fit.iloc[i_val], fit_y[i_val], args.threads,
                                         max_rounds=ceiling)
            auc = float(roc_auc_score(fit_y[i_val], predict(booster, x_fit.iloc[i_val])))
            scored.append({"name": cand["name"], "params": cand["params"],
                           "inner_auc": auc, "rounds": rounds})
            print(f"[{tag}]   fold {fold} {cand['name']:<12} inner {auc:.7f} "
                  f"rounds {rounds}", flush=True)
            del booster
            gc.collect()

        win = max(scored, key=lambda r: r["inner_auc"])
        # Scale the selected round count up to the full fit partition, which is
        # 1/(1-0.16) larger than the sub-train the count was chosen on.
        rounds = max(50, int(win["rounds"] / (1 - 0.16)))
        booster, _ = train_eval(win["params"], x_fit, fit_y, None, None,
                                args.threads, rounds=rounds)
        val_pred = predict(booster, x_val)
        oof[val_i] = val_pred
        test_pred += predict(booster, x_test)
        folds_done += 1

        auc = float(roc_auc_score(y[val_i], val_pred))
        records.append({"fold": fold, "outer_auc": auc, "chosen": win["name"],
                        "rounds": rounds,
                        "inner_scores": {s["name"]: s["inner_auc"] for s in scored}})
        print(f"[{tag}] fold {fold} AUC {auc:.8f} via {win['name']} "
              f"({time.time()-started:.0f}s)", flush=True)
        del x_fit, x_val, x_test, booster
        gc.collect()

    if folds_done == 0:
        raise RuntimeError("no outer fold completed; nothing to write")

    complete = folds_done == n_folds
    test_pred /= folds_done
    scored_rows = np.concatenate([outer_splits[i][1] for i in range(folds_done)])
    pooled = float(roc_auc_score(y[scored_rows], oof[scored_rows]))
    print(f"[{tag}] pooled OOF AUC {pooled:.10f} over {folds_done}/{n_folds} folds",
          flush=True)

    if complete:
        pd.DataFrame({ID: train[ID], "fold": -1, "y": y, "pred": oof}).to_csv(
            out_dir / f"oof_{tag}.csv", index=False)
        pd.DataFrame({ID: test[ID], TARGET: test_pred}).to_csv(
            out_dir / f"test_{tag}.csv", index=False)
    else:
        # A partial OOF cannot enter the stack, so it is written under a name
        # the registry does not pick up.
        pd.DataFrame({ID: train[ID].iloc[scored_rows], "y": y[scored_rows],
                      "pred": oof[scored_rows]}).to_csv(
            out_dir / f"partial_oof_{tag}.csv", index=False)

    (out_dir / f"metrics_{tag}.json").write_text(json.dumps({
        "model": tag,
        "learner": LEARNER,
        "official_data_only": True,
        "public_predictions_used": False,
        "outer_folds": n_folds,
        "outer_folds_completed": folds_done,
        "complete": complete,
        "inner_encoding_folds": INNER_FOLDS,
        "search_trials": len(trials),
        "search_best_auc": trials[0]["search_auc"] if trials else None,
        "search_note": "phase A scores select and evaluate on the same holdout; "
                       "they rank configurations and are not OOF estimates",
        "finalists": finalists,
        "fold_records": records,
        "oof_auc": pooled,
        "runtime_seconds": time.time() - started,
        "submission_created": False,
    }, indent=2, default=float) + "\n")


if __name__ == "__main__":
    if Path("/kaggle").exists():
        sys.argv = ["experiment", "--threads", "4", "--folds", "10"]
    main()
