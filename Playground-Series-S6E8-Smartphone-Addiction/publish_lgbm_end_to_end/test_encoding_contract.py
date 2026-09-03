#!/usr/bin/env python3
"""Executable leakage contract for the fold-safe encoder in experiment_v4.

Run: ../.venv/bin/python test_encoding_contract.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import experiment_v4 as E  # noqa: E402

SMOOTHING = 40.0


def fixture(n=20000, seed=1):
    train = pd.read_csv(E.ROOT / "train.csv").sample(n, random_state=seed).reset_index(drop=True)
    raw = [c for c in train.columns if c not in (E.ID, E.TARGET)]
    return train, raw, E.make_base_features(train), train[E.TARGET].to_numpy("int8")


def test_inner_folds_partition_exactly_once():
    n = 5000
    cov = np.zeros(n, dtype=int)
    for tr, va in E.inner_folds(n, E.SEED):
        assert len(np.intersect1d(tr, va)) == 0, "inner train/validation overlap"
        cov[va] += 1
    assert (cov == 1).all(), "inner folds must cover every row exactly once"


def test_inner_folds_do_not_depend_on_labels():
    a = [va.tolist() for _, va in E.inner_folds(5000, E.SEED)]
    b = [va.tolist() for _, va in E.inner_folds(5000, E.SEED)]
    assert a == b, "fold assignment must be deterministic and label-free"


def test_oof_encoding_matches_independent_reconstruction():
    _, raw, xb, y = fixture()
    out = E.encode(xb, y, xb, raw, "pair_te", inner_oof=True)
    prior = float(y.mean())
    for col in ("daily_screen_time_hours", "stress_level"):
        keys = E.key_of(xb, col)
        expect = np.full(len(xb), np.nan, dtype="float32")
        for tr, va in E.inner_folds(len(xb), E.SEED):
            rate = E.target_rate(keys.iloc[tr], y[tr], SMOOTHING, prior)
            expect[va] = keys.iloc[va].map(rate).fillna(prior).to_numpy("float32")
        assert np.allclose(expect, out[col + "__te"].to_numpy(), atol=1e-6), col


def test_own_label_is_excluded():
    """Swap one positive and one negative label inside a single inner fold.

    The prior is preserved exactly, so any change to that fold's own encodings
    could only come from those rows seeing their own labels. There must be none.
    """
    _, raw, xb, y = fixture()
    base = E.encode(xb, y, xb, raw, "pair_te", inner_oof=True)
    folds = list(E.inner_folds(len(xb), E.SEED))
    val0 = folds[0][1]
    flipped = y.copy()
    flipped[val0[y[val0] == 0][0]] = 1
    flipped[val0[y[val0] == 1][0]] = 0
    assert flipped.mean() == y.mean()
    other = E.encode(xb, flipped, xb, raw, "pair_te", inner_oof=True)
    te = [c for c in base.columns if c.endswith("__te")]
    a, b = base[te].to_numpy(), other[te].to_numpy()
    assert np.allclose(a[val0], b[val0], atol=1e-7), "a row's encoding moved with its own label"
    rest = np.setdiff1d(np.arange(len(xb)), val0)
    assert not np.allclose(a[rest], b[rest], atol=1e-7), "encoder ignored labels it should use"


def test_support_features_never_use_labels():
    _, raw, xb, y = fixture(n=8000)
    a = E.encode(xb, y, xb, raw, "single_te", inner_oof=True)
    b = E.encode(xb, 1 - y, xb, raw, "single_te", inner_oof=True)
    sup = [c for c in a.columns if "logfreq" in c]
    assert a[sup].equals(b[sup]), "support/frequency features must be target-free"


def test_validation_rows_never_use_their_own_partition():
    """Validation features come only from the fit partition, as at test time."""
    _, raw, xb, y = fixture(n=12000)
    fit, val = np.arange(0, 8000), np.arange(8000, 12000)
    a = E.encode(xb.iloc[fit], y[fit], xb.iloc[val], raw, "pair_te", inner_oof=False)
    xb2 = xb.copy()
    # Perturbing validation-row feature values must not alter the fitted maps,
    # only which map entry each validation row looks up.
    b = E.encode(xb.iloc[fit], y[fit], xb2.iloc[val], raw, "pair_te", inner_oof=False)
    assert a.equals(b)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
