#!/usr/bin/env python3
"""Second-generation meta-learner search over the validated stream inventory.

`mega_stack.py` established that a plain cross-fitted logistic stack over 40
own streams reaches OOF 0.9696223 and public LB 0.97060, which fixes the
calibration at roughly LB = OOF + 0.00099. This script tries to buy more OOF
by letting the meta-learner condition on the missingness regime, which every
earlier diagnostic flagged as informative but which a global linear blend
cannot express.

Official competition data only. Nothing under `public_outputs/` is loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import ndtri
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import mega_stack as ms

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ensemble_original/reports"
CACHE = OUT / "stream_matrix.npz"
TARGET = ms.TARGET
SEED = 20260806

NUMERIC = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
]
CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]


def build_matrix():
    """Load and gauss-rank every usable stream, caching the result.

    The cache is keyed on the registry fingerprint, so adding or repointing a
    stream forces a rebuild instead of silently reusing the old matrix."""
    fingerprint = ms.registry_fingerprint()
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if str(z.get("fingerprint", "")) == fingerprint:
            return list(z["names"]), z["X"], z["XT"], z["y"]
        print("registry changed since cache was written; rebuilding stream matrix")
    train = pd.read_csv(ROOT / "train.csv", usecols=["id", TARGET])
    test = pd.read_csv(ROOT / "test.csv", usecols=["id"])
    names, X, XT, y, report = ms.load_streams(train, test)
    report.to_csv(OUT / "stack_v2_streams.csv", index=False)
    np.savez_compressed(CACHE, names=np.array(names, dtype=object), X=X, XT=XT, y=y,
                        fingerprint=fingerprint)
    return names, X, XT, y


def context_features():
    """Target-free regime descriptors the meta-learner may gate on.

    These come straight from the raw tables, never from labels, so they add no
    leakage beyond what the base models already saw."""
    cols = ["id"] + NUMERIC + CATEGORICAL
    tr = pd.read_csv(ROOT / "train.csv", usecols=cols)
    te = pd.read_csv(ROOT / "test.csv", usecols=cols)

    def make(df):
        out = pd.DataFrame(index=df.index)
        for c in NUMERIC + CATEGORICAL:
            out[f"miss_{c}"] = df[c].isna().astype(np.int8)
        out["miss_count"] = out.filter(like="miss_").sum(axis=1).astype(np.int8)
        for c in NUMERIC:
            out[c] = df[c].astype(np.float32)
        for c in CATEGORICAL:
            out[c] = df[c].astype("category").cat.codes.astype(np.int8)
        # Screen-time accounting residual: the generator's strongest known
        # constraint, and the axis along which model families disagree most.
        parts = df[["social_media_hours", "gaming_hours", "work_study_hours"]].sum(axis=1, min_count=1)
        out["day_residual"] = (df["daily_screen_time_hours"] - parts).astype(np.float32)
        out["weekend_gap"] = (df["weekend_screen_time"] - df["daily_screen_time_hours"]).astype(np.float32)
        return out

    return make(tr), make(te)


def crossfit(model_fn, X, y, folds):
    pred = np.zeros(len(y))
    for fit, val in folds:
        pred[val] = model_fn(X[fit], y[fit], X[val])
    return pred


def logistic_fn(C):
    def f(Xf, yf, Xv):
        m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(Xf, yf)
        return m.decision_function(Xv)
    return f


def lgb_fn(params, rounds, seed=SEED):
    def f(Xf, yf, Xv):
        # Early stopping uses a split carved out of the fit rows only, so the
        # held-out meta fold never influences the round count.
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(yf))
        cut = int(0.85 * len(idx))
        tr_i, es_i = idx[:cut], idx[cut:]
        ds = lgb.Dataset(Xf[tr_i], label=yf[tr_i])
        dv = lgb.Dataset(Xf[es_i], label=yf[es_i], reference=ds)
        booster = lgb.train(
            {**params, "objective": "binary", "metric": "auc", "verbosity": -1,
             "num_threads": 16, "seed": seed},
            ds, num_boost_round=rounds, valid_sets=[dv],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return booster.predict(Xv, num_iteration=booster.best_iteration)
    return f


def main() -> None:
    names, X, XT, y = build_matrix()
    print(f"streams: {len(names)}")
    ctx_tr, ctx_te = context_features()
    ctx_names = list(ctx_tr.columns)
    XC = np.column_stack([X, ctx_tr.to_numpy(np.float32)])
    XCT = np.column_stack([XT, ctx_te.to_numpy(np.float32)])

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = list(cv.split(X, y))
    results, preds = {}, {}

    preds["logistic"] = crossfit(logistic_fn(0.1), X, y, folds)
    results["logistic_streams"] = roc_auc_score(y, preds["logistic"])
    print(f"logistic (streams only)          {results['logistic_streams']:.10f}")

    grids = {
        "lgb_streams_shallow": (dict(learning_rate=0.02, num_leaves=15, min_data_in_leaf=2000,
                                     feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                                     lambda_l2=20.0), 3000, False),
        "lgb_streams_deep": (dict(learning_rate=0.02, num_leaves=63, min_data_in_leaf=1000,
                                  feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                                  lambda_l2=10.0), 3000, False),
        "lgb_ctx_shallow": (dict(learning_rate=0.02, num_leaves=15, min_data_in_leaf=2000,
                                 feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                                 lambda_l2=20.0), 3000, True),
        "lgb_ctx_deep": (dict(learning_rate=0.02, num_leaves=63, min_data_in_leaf=1000,
                              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                              lambda_l2=10.0), 3000, True),
    }
    for tag, (params, rounds, use_ctx) in grids.items():
        mat = XC if use_ctx else X
        p = crossfit(lgb_fn(params, rounds), mat, y, folds)
        preds[tag] = p
        results[tag] = roc_auc_score(y, p)
        print(f"{tag:<32} {results[tag]:.10f}")

    # Rank-average the linear and non-linear meta-learners: they make very
    # different errors, so a fixed 50/50 blend is worth measuring explicitly.
    def rank(v):
        return pd.Series(v).rank(pct=True).to_numpy()

    for tag in list(grids):
        mix = 0.5 * rank(preds["logistic"]) + 0.5 * rank(preds[tag])
        results[f"mix_logistic_{tag}"] = roc_auc_score(y, mix)
        print(f"mix_logistic+{tag:<20} {results[f'mix_logistic_{tag}']:.10f}")

    pd.Series(results).sort_values(ascending=False).to_csv(OUT / "stack_v2_auc.csv")
    print("\n" + pd.Series(results).sort_values(ascending=False).to_string())

    best = max(results, key=results.get)
    print(f"\nbest meta-learner: {best} at {results[best]:.10f}")
    print(f"implied LB (OOF + 0.00099): {results[best] + 0.00099:.5f}")

    # Refit the winner on all rows and score the test matrix.
    if best == "logistic_streams":
        m = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(X, y)
        test_pred = m.decision_function(XT)
    elif best.startswith("mix_logistic_"):
        tag = best[len("mix_logistic_"):]
        params, rounds, use_ctx = grids[tag]
        m = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(X, y)
        lin_t = m.decision_function(XT)
        mat, mat_t = (XC, XCT) if use_ctx else (X, XT)
        nl_t = lgb_fn(params, rounds)(mat, y, mat_t)
        test_pred = 0.5 * rank(lin_t) + 0.5 * rank(nl_t)
    else:
        params, rounds, use_ctx = grids[best]
        mat, mat_t = (XC, XCT) if use_ctx else (X, XT)
        test_pred = lgb_fn(params, rounds)(mat, y, mat_t)

    test_ids = pd.read_csv(ROOT / "test.csv", usecols=["id"]).id
    sub = pd.DataFrame({"id": test_ids, TARGET: rank(test_pred)})
    assert len(sub) == len(test_ids) and sub[TARGET].between(0, 1).all()
    sub.to_csv(OUT / "stack_v2_submission.csv", index=False)
    manifest = {
        "provenance": "own official-data-only streams; no public predictions",
        "streams": names,
        "context_features": ctx_names,
        "chosen": best,
        "crossfit_auc": float(results[best]),
        "results": {k: float(v) for k, v in results.items()},
        "rows": len(sub),
        "test_id_sha256": hashlib.sha256(np.asarray(test_ids, dtype=np.int64).tobytes()).hexdigest()[:16],
        "submitted": False,
    }
    (OUT / "stack_v2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
