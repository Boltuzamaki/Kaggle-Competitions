#!/usr/bin/env python3
"""How much of the recent stack improvement is real?

Recent stream additions have moved the cross-fitted OOF by 0.00001 to 0.00008.
Those numbers are only meaningful if they are larger than the noise in the
estimate itself, and that noise has never been measured here. Every refit so far
used one fixed meta-fold seed, so a change in the number could be a change in the
model or could be the fold split moving under it.

This script measures three things:

1. The spread of the stack's OOF across meta-fold seeds, which is the noise floor
   for every comparison made in this project.
2. A paired comparison of "with stream" against "without stream" using identical
   folds per seed, which removes fold noise from the contrast and is far more
   sensitive than comparing two independently-run numbers.
3. A held-out check: weights selected on one half of the rows, scored on the
   other half, so the reported figure involves no selection on the rows it scores.

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

OUT = Path(__file__).resolve().parents[1] / "ensemble_original/reports"
C = 0.1
SEEDS = [20260806 + i for i in range(8)]


def crossfit(X, y, seed):
    pred = np.zeros(len(y))
    for fit_i, val_i in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        pred[val_i] = LogisticRegression(C=C, max_iter=2000).fit(
            X[fit_i], y[fit_i]).decision_function(X[val_i])
    return roc_auc_score(y, pred)


def main() -> None:
    names, X, XT, y, _ = SW.load()
    print(f"\nstreams: {len(names)}")

    print("\n=== 1. noise floor of the cross-fitted estimate ===")
    scores = [crossfit(X, y, s) for s in SEEDS]
    scores = np.array(scores)
    print(f"  mean {scores.mean():.7f}  std {scores.std(ddof=1):.7f}")
    print(f"  min  {scores.min():.7f}  max {scores.max():.7f}  range {np.ptp(scores):.7f}")
    print(f"  any single-seed comparison smaller than ~{2*scores.std(ddof=1):.7f} is not evidence")

    print("\n=== 2. paired with/without, identical folds per seed ===")
    print("  (each row is the mean paired difference over 8 seeds)")
    # The full-stack score per seed does not depend on which candidate is being
    # tested, so it is computed once instead of once per candidate.
    full = dict(zip(SEEDS, scores))
    rows = []
    candidates = sys.argv[1:] or [
        "cpu_linear", "foldsafe_te_wide", "foldsafe_te_cat", "lookup_v2_s20260902"]
    for drop in candidates:
        if drop not in names:
            print(f"  {drop:<22} (absent from the stream matrix, skipped)")
            continue
        keep = [i for i, n in enumerate(names) if n != drop]
        diffs = np.array([full[s] - crossfit(X[:, keep], y, s) for s in SEEDS])
        t = diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs)) + 1e-12)
        rows.append({"stream": drop, "mean_gain": diffs.mean(),
                     "sd": diffs.std(ddof=1), "t": t,
                     "verdict": "real" if abs(t) > 3 and diffs.mean() > 0 else "not distinguishable"})
        print(f"  {drop:<22} gain {diffs.mean():+.7f}  sd {diffs.std(ddof=1):.7f}  t {t:+.1f}  {rows[-1]['verdict']}")
    pd.DataFrame(rows).to_csv(OUT / "validation_paired.csv", index=False)

    print("\n=== 3. held-out half, no selection on the scored rows ===")
    rng = np.random.default_rng(12345)
    order = rng.permutation(len(y))
    a, b = order[:len(order) // 2], order[len(order) // 2:]
    m = LogisticRegression(C=C, max_iter=2000).fit(X[a], y[a])
    held = roc_auc_score(y[b], m.decision_function(X[b]))
    best_single = max(roc_auc_score(y[b], X[b, i]) for i in range(X.shape[1]))
    print(f"  weights fit on half A, scored on half B : {held:.7f}")
    print(f"  best single stream on half B            : {best_single:.7f}")
    print(f"  stacking gain on untouched rows         : {held - best_single:+.7f}")


if __name__ == "__main__":
    main()
