#!/usr/bin/env python3
"""Is the L2 logistic meta-learner leaving anything on the table?

The stack has been an unconstrained L2 logistic regression on gauss-rank streams
since S004. Two alternatives were tried and rejected then, both of which are
*more* restrictive than the incumbent rather than less: non-negative least
squares (0.9689) and hill climbing (0.9694). Nothing with more capacity has ever
been tested against the full inventory.

That is worth checking, because at this point a new stream buys about 0.000005
and the whole meta-learner is one linear map. Two specific reasons to think
capacity might pay:

  - E019 found that missingness regimes carried real signal, but it tested four
    models and the finding never made it into the 50-stream stack, which applies
    one global weight vector to every row regardless of how much of that row is
    actually observed. A model that is excellent on complete rows and poor on
    sparse ones should not get the same weight in both places.
  - Streams disagree most where they are least certain, and a linear map cannot
    express "trust the lookup transformer when the trees disagree with it".

Four meta-learners are cross-fitted on identical folds and compared over eight
meta-fold seeds, because the noise floor of a single cross-fitted number here is
about 0.000002 and any honest comparison has to be paired against it.

Official competition data only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import stack_weights as SW  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ensemble_original/reports"
TARGET = "addicted_label"
C = 0.1
SEEDS = [20260806 + i for i in range(8)]


def regime_of(train: pd.DataFrame) -> np.ndarray:
    """Missingness regime, the split E019 found carried signal."""
    missing = train.drop(columns=["id", TARGET]).isna().sum(axis=1).to_numpy()
    return np.where(missing == 0, 0, np.where(missing <= 3, 1, 2))


def fit_logistic(Xf, yf, Xv, c=C):
    return LogisticRegression(C=c, max_iter=2000, solver="lbfgs").fit(
        Xf, yf).decision_function(Xv)


def build_regime_design(X: np.ndarray, regime: np.ndarray) -> np.ndarray:
    """Interact every stream with every regime.

    The result holds one weight vector per regime rather than one overall, so a
    stream can be trusted on complete rows and discounted on sparse ones. Regime
    indicators are appended so each regime also keeps its own intercept."""
    blocks = [X * (regime == r)[:, None].astype(np.float32) for r in (0, 1, 2)]
    dummies = np.column_stack([(regime == r).astype(np.float32) for r in (1, 2)])
    return np.column_stack(blocks + [dummies])


def fit_lgb(Xf, yf, Xv, seed):
    """Shallow, heavily regularised boosting on the stream matrix.

    Capacity is kept low on purpose: the columns are 0.99 correlated with each
    other, so a deep model would spend itself splitting on noise between
    near-duplicates."""
    import lightgbm as lgb
    params = {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
              "num_leaves": 8, "min_data_in_leaf": 2000, "feature_fraction": 0.5,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 20.0,
              "verbosity": -1, "seed": seed, "num_threads": 4}
    booster = lgb.train(params, lgb.Dataset(Xf, label=yf), num_boost_round=400)
    return booster.predict(Xv)


def main() -> None:
    names, X, XT, y, _ = SW.load()
    train = pd.read_csv(ROOT / "train.csv")
    regime = regime_of(train)
    print(f"\nstreams: {len(names)}   rows: {len(y)}")
    for r in (0, 1, 2):
        print(f"  regime {r}: {int((regime == r).sum()):>7} rows")

    Xr = build_regime_design(X, regime)
    Xm = np.column_stack([X, regime.astype(np.float32)])
    print(f"regime design: {Xr.shape}")

    variants = {
        "logistic (incumbent)": lambda fit, val, s: fit_logistic(
            X[fit], y[fit], X[val]),
        "logistic + regime interactions": lambda fit, val, s: fit_logistic(
            Xr[fit], y[fit], Xr[val]),
        "logistic, C=1.0": lambda fit, val, s: fit_logistic(
            X[fit], y[fit], X[val], c=1.0),
        "shallow lightgbm": lambda fit, val, s: fit_lgb(
            Xm[fit], y[fit], Xm[val], s),
    }

    scores = {k: [] for k in variants}
    for s in SEEDS:
        folds = list(StratifiedKFold(5, shuffle=True, random_state=s).split(X, y))
        for name, fn in variants.items():
            pred = np.zeros(len(y))
            for fit, val in folds:
                pred[val] = fn(fit, val, s)
            scores[name].append(roc_auc_score(y, pred))
        print(f"  seed {s}: " + "  ".join(
            f"{k.split()[0][:9]}={scores[k][-1]:.7f}" for k in variants), flush=True)

    print("\n=== cross-fitted AUC over 8 meta-fold seeds ===")
    base = np.array(scores["logistic (incumbent)"])
    rows = []
    for name, vals in scores.items():
        v = np.array(vals)
        d = v - base
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12) if name != \
            "logistic (incumbent)" else 0.0
        rows.append({"meta_learner": name, "mean_auc": v.mean(), "sd": v.std(ddof=1),
                     "gain_vs_incumbent": d.mean(), "t": t})
        print(f"  {name:<32} {v.mean():.7f}  sd {v.std(ddof=1):.7f}  "
              f"gain {d.mean():+.7f}  t {t:+.1f}")
    pd.DataFrame(rows).to_csv(OUT / "meta_learner_search.csv", index=False)
    print("\nKeep a change only if the paired gain is positive with |t| > 3.")


if __name__ == "__main__":
    main()
