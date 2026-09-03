#!/usr/bin/env python3
"""Compare weight-fitting schemes for the stream stack on the cached matrix.

The deployed stack fits an unconstrained L2 logistic regression on gauss-ranked
stream scores. With 25+ tree streams sitting at 0.99+ mutual correlation, an
unconstrained fit is free to place large offsetting positive and negative
weights on near-duplicate columns. That can fit meta-training noise, so this
script measures constrained and averaged alternatives under the identical
cross-fitting protocol.

Reads the cached stream matrix written by stack_v2.py and appends any newly
completed streams. Official competition data only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import ndtri
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import mega_stack as ms  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ensemble_original/reports"
CACHE = OUT / "stream_matrix.npz"
TARGET = "addicted_label"
SEED = 20260806
# The OOF-to-LB offset shrinks as OOF rises: +0.00100 early, +0.00098 in the
# middle, +0.00096 measured on the 72-stream submission (OOF 0.9698402 -> LB
# 0.97080). Using the old headline +0.00099 here overstated the leaderboard by
# two ten-thousandths, which is larger than several recent gains.
LB_OFFSET = 0.00096

# Streams finished after the cache was written; skipped silently if absent.
EXTRA = {
    # Streams dropped after paired testing: cpu_linear (t=-2.3) and
    # foldsafe_te_cat (t=+0.6). See EXPERIMENT_LOG.md V002.
    "foldsafe_te_xgb": ("artifacts/foldsafe_te_xgb/oof_foldsafe_te_xgb.csv",
                        "artifacts/foldsafe_te_xgb/test_foldsafe_te_xgb.csv", "pred", TARGET),
    "foldsafe_te_xgb_10f": ("gpu_foldsafe_te_xgb/output_10fold/oof_foldsafe_te_xgb_10f.csv",
                            "gpu_foldsafe_te_xgb/output_10fold/test_foldsafe_te_xgb_10f.csv", "pred", TARGET),
    "foldsafe_te_multi": ("artifacts/foldsafe_te_multi/oof_foldsafe_te_xgb_multi.csv",
                          "artifacts/foldsafe_te_multi/test_foldsafe_te_xgb_multi.csv", "pred", TARGET),
    "foldsafe_te_wide": ("artifacts/foldsafe_te_wide/oof_foldsafe_te_wide.csv",
                         "artifacts/foldsafe_te_wide/test_foldsafe_te_wide.csv", "pred", TARGET),
    # CPU families chosen for a different inductive bias rather than for score.
    "cpu_lgbdart": ("cpu_diverse_models/output/oof_lgbdart.csv",
                    "cpu_diverse_models/output/test_lgbdart.csv", "pred", TARGET),
    "cpu_extratrees": ("artifacts/cpu_extratrees/oof_extratrees.csv",
                       "artifacts/cpu_extratrees/test_extratrees.csv", "pred", TARGET),
    "cpu_randomforest": ("artifacts/cpu_randomforest/oof_randomforest.csv",
                         "artifacts/cpu_randomforest/test_randomforest.csv", "pred", TARGET),
    # Dropped after paired testing: cpu_lgb10fold (t=+2.8, below the |t|>3 bar).
    # A GBDT within 0.0002 of the best stream added nothing; see E054.
    "cpu_et_deep": ("artifacts/cpu_et_deep/oof_et_deep.csv",
                    "artifacts/cpu_et_deep/test_et_deep.csv", "pred", TARGET),
    # First network trained on the target-encoded view rather than a target-free
    # one; the four target-free architectures all stalled near 0.939.
    "cpu_mlp_te": ("artifacts/cpu_mlp_te/oof_mlp_te.csv",
                   "artifacts/cpu_mlp_te/test_mlp_te.csv", "pred", TARGET),
    # Tuned boosted trees, from the search kernels.
    "cpu_hpo_lgb": ("artifacts/cpu_hpo_lgb/oof_hpo_lgb.csv",
                    "artifacts/cpu_hpo_lgb/test_hpo_lgb.csv", "pred", TARGET),
    "cpu_hpo_xgb": ("artifacts/cpu_hpo_xgb/oof_hpo_xgb.csv",
                    "artifacts/cpu_hpo_xgb/test_hpo_xgb.csv", "pred", TARGET),
    # Nystroem-approximated RBF kernel machine: local similarity to landmark
    # rows, which is neither the axis-aligned boxes of a tree nor the smooth
    # global surface of a network. Dropped 2026-08-07: standalone 0.9515181 is
    # far below the band floor, and adding it moved the 63-stream stack from
    # 0.9698134 to 0.9698083. A novel inductive bias does not buy a place in the
    # stack on its own; it has to clear the accuracy band first. See E058.
    # "cpu_kernel_rbf": ("artifacts/cpu_kernel_rbf/oof_kernel_rbf.csv",
    #                    "artifacts/cpu_kernel_rbf/test_kernel_rbf.csv", "pred", TARGET),
    "cpu_rf_deep": ("artifacts/cpu_rf_deep/oof_rf_deep.csv",
                    "artifacts/cpu_rf_deep/test_rf_deep.csv", "pred", TARGET),
    # Ten folds and three seeds of the network that works, plus a wider and
    # deeper one run for a differently-shaped error rather than a better score.
    "cpu_mlp_te_10f": ("artifacts/cpu_mlp_te_10f/oof_mlp_te_10f.csv",
                       "artifacts/cpu_mlp_te_10f/test_mlp_te_10f.csv", "pred", TARGET),
    "cpu_mlp_te_wide": ("artifacts/cpu_mlp_te_wide/oof_mlp_te_wide.csv",
                        "artifacts/cpu_mlp_te_wide/test_mlp_te_wide.csv", "pred", TARGET),
    # Architectures other than a fully-connected stack, all on the same fold-safe
    # TE view. The four target-free neural rejections in the ledger (FT-Transformer
    # 0.9401, DCNv2 0.9385, GANDALF 0.9387, TabR 0.9380) were rejections of the
    # view, not of the architectures: the plain MLP scored 0.9661 once it was
    # given the target statistics. All three land inside the accuracy band. See E060.
    "cpu_resnet_te": ("artifacts/cpu_resnet_te/oof_resnet_te.csv",
                      "artifacts/cpu_resnet_te/test_resnet_te.csv", "pred", TARGET),
    "cpu_cnn1d_te": ("artifacts/cpu_cnn1d_te/oof_cnn1d_te.csv",
                     "artifacts/cpu_cnn1d_te/test_cnn1d_te.csv", "pred", TARGET),
    "cpu_dae_te": ("artifacts/cpu_dae_te/oof_dae_te.csv",
                   "artifacts/cpu_dae_te/test_dae_te.csv", "pred", TARGET),
    # FT-Transformer, run on the local 4060 after Kaggle's P100 turned out to
    # have no torch kernel. At 0.9671574 it is the strongest network in the
    # inventory and beats the plain MLP by 0.0010. The ledger recorded this same
    # architecture at 0.9401 target-free, so the view was worth +0.027 to it.
    "cpu_ftt_te": ("artifacts/cpu_ftt_te/oof_ftt_te.csv",
                   "artifacts/cpu_ftt_te/test_ftt_te.csv", "pred", TARGET),
    # Ten folds of the same transformer: 0.9671574 -> 0.9673175, the fold lever
    # holding for a fourth family. Note the paired test says this family pays far
    # less than its standalone rank suggests (t=+4.3 against resnet's t=+9.9), so
    # the bag is not widened past these two.
    "cpu_ftt_te_10f": ("artifacts/cpu_ftt_te_10f/oof_ftt_te_10f.csv",
                       "artifacts/cpu_ftt_te_10f/test_ftt_te_10f.csv", "pred", TARGET),
    "cpu_ftt_te_wide": ("artifacts/cpu_ftt_te_wide/oof_ftt_te_wide.csv",
                        "artifacts/cpu_ftt_te_wide/test_ftt_te_wide.csv", "pred", TARGET),
    "cpu_gated_te": ("artifacts/cpu_gated_te/oof_gated_te.csv",
                     "artifacts/cpu_gated_te/test_gated_te.csv", "pred", TARGET),
    "cpu_dae_te_10f": ("artifacts/cpu_dae_te_10f/oof_dae_te_10f.csv",
                       "artifacts/cpu_dae_te_10f/test_dae_te_10f.csv", "pred", TARGET),
    # A third autoencoder, wider bottleneck and heavier swap noise. Carried on
    # sufferance: the family's two existing members both failed a paired test
    # (t=+1.9 and t=+1.1), so this one is expected to fail too and is included
    # only because it is complete and in band. It goes if it does not clear |t|>3.
    "cpu_dae_te_big": ("artifacts/cpu_dae_te_big/oof_dae_te_big.csv",
                       "artifacts/cpu_dae_te_big/test_dae_te_big.csv", "pred", TARGET),
    "cpu_resnet_te_deep": ("artifacts/cpu_resnet_te_deep/oof_resnet_te_deep.csv",
                           "artifacts/cpu_resnet_te_deep/test_resnet_te_deep.csv", "pred", TARGET),
    # dcnv2_te is deliberately absent. At 0.9483968 it sits below every stream
    # the band floor has ever admitted, and the two nearest comparisons both cost
    # the stack score: the RBF machine at 0.9515 (E059) and cpu_linear at 0.9578
    # (E052). Bounded-degree crossing is a genuinely different mechanism, which
    # is exactly the argument that already failed twice. See E065.
    # Tuned member of the MLP family. The search found nothing the hand-set shape
    # did not already have (0.96500 against 0.96497 on the search split, and
    # 0.9660810 against 0.9661121 cross-fitted), so this is carried as a second
    # seed of a known-good configuration rather than as an improvement. See E062.
    "cpu_hpo_mlp_te": ("artifacts/cpu_hpo_mlp_te/oof_hpo_mlp_te.csv",
                       "artifacts/cpu_hpo_mlp_te/test_hpo_mlp_te.csv", "pred", TARGET),
    # Recovered 2026-08-07 from a filename mismatch in mega_stack, then dropped
    # again: paired t of -0.2, +1.1, -2.0 and +0.1 for v0/v1/v2/v6. Four more
    # LightGBMs on the published view are four more things the stack already
    # knows. tabm_lattice (0.9592) also failed at t=+0.6, which puts the band
    # floor above it. See E057.
}
# The lookup transformer is the only family materially decorrelated from the
# trees, so its seed bag is widened rather than adding more boosted variants.
for _seed in (20260901, 20260902, 20260903, 20260904,
              20260905, 20260906, 20260907, 20260908):
    EXTRA[f"lookup_v2_s{_seed}"] = (
        f"artifacts/local_lookup_v2_seed{_seed}/oof_lookup_v2.csv",
        f"artifacts/local_lookup_v2_seed{_seed}/test_lookup_v2.csv", "pred", TARGET)
# The ten-fold run of the same family, chained behind the seed queue. Ten folds
# was worth +0.00018 on XGBoost, and this applies that lever to the family that
# carries the most stack weight.
EXTRA["lookup_v2_10fold"] = ("artifacts/local_lookup_v2_10fold/oof_lookup_v2.csv",
                             "artifacts/local_lookup_v2_10fold/test_lookup_v2.csv",
                             "pred", TARGET)
# That single ten-fold run is the strongest stream in the library, so the seed
# bag is widened at ten folds rather than five. Each of these outscores every
# five-fold seed of the same family and every boosted tree.
for _seed in (20260911, 20260912, 20260913, 20260914,
              20260915, 20260916, 20260917, 20260918):
    EXTRA[f"lookup_v2_10fold_s{_seed}"] = (
        f"artifacts/local_lookup_v2_10fold_s{_seed}/oof_lookup_v2.csv",
        f"artifacts/local_lookup_v2_10fold_s{_seed}/test_lookup_v2.csv", "pred", TARGET)


def load() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    z = np.load(CACHE, allow_pickle=True)
    names, X, XT, y = list(z["names"]), z["X"], z["XT"], z["y"]
    train_ids = pd.read_csv(ROOT / "train.csv", usecols=["id"]).id
    test_ids = pd.read_csv(ROOT / "test.csv", usecols=["id"]).id
    for name, (op, tp, oc, tc) in EXTRA.items():
        if name in names:
            continue
        opath, tpath = ROOT / op, ROOT / tp
        if not (opath.exists() and tpath.exists()):
            print(f"  (pending) {name}")
            continue
        o = pd.read_csv(opath).set_index("id")[oc].reindex(train_ids)
        t = pd.read_csv(tpath).set_index("id")[tc].reindex(test_ids)
        if o.isna().any() or t.isna().any():
            print(f"  (incomplete) {name}")
            continue
        auc = roc_auc_score(y, o.to_numpy(float))
        print(f"  + {name}: standalone {auc:.7f}")
        names.append(name)
        X = np.column_stack([X, ms.gauss_rank(o.to_numpy(float))])
        XT = np.column_stack([XT, ms.gauss_rank(t.to_numpy(float))])
    return names, X, XT, y, test_ids


def fit_nnls(Xf, yf):
    """Non-negative weights against the centred label.

    Constraining weights to be non-negative removes the offsetting
    positive/negative pairs that near-duplicate streams invite."""
    w, _ = nnls(Xf, (yf - yf.mean()).astype(np.float64))
    return w / (w.sum() or 1.0)


def main() -> None:
    print("loading cached matrix")
    names, X, XT, y, test_ids = load()
    print(f"streams: {len(names)}")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = list(cv.split(X, y))
    results, test_preds = {}, {}

    for C in (0.03, 0.1, 0.3):
        pred = np.zeros(len(y))
        for fit, val in folds:
            m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(X[fit], y[fit])
            pred[val] = m.decision_function(X[val])
        results[f"logistic_C{C}"] = roc_auc_score(y, pred)
        print(f"logistic C={C:<5} {results[f'logistic_C{C}']:.10f}", flush=True)
    best_C = float(max((k for k in results), key=lambda k: results[k]).split("C")[1])
    m = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs").fit(X, y)
    test_preds[f"logistic_C{best_C}"] = m.decision_function(XT)

    pred = np.zeros(len(y))
    ws = []
    for fit, val in folds:
        w = fit_nnls(X[fit], y[fit])
        ws.append(w)
        pred[val] = X[val] @ w
    results["nnls"] = roc_auc_score(y, pred)
    print(f"nnls              {results['nnls']:.10f}", flush=True)
    test_preds["nnls"] = XT @ np.mean(ws, axis=0)

    # Bagging the linear fit over resampled meta-training sets reduces variance
    # in the weights themselves without changing the hypothesis class.
    pred = np.zeros(len(y))
    bag_models = []
    for fit, val in folds:
        acc = np.zeros(len(val))
        for b in range(5):
            rng = np.random.default_rng(SEED + b)
            sub = rng.choice(fit, int(0.7 * len(fit)), replace=False)
            mb = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs").fit(X[sub], y[sub])
            acc += mb.decision_function(X[val]) / 5
            if len(bag_models) < 5:
                bag_models.append(mb)
        pred[val] = acc
    results["logistic_bagged"] = roc_auc_score(y, pred)
    print(f"logistic_bagged   {results['logistic_bagged']:.10f}", flush=True)
    test_preds["logistic_bagged"] = np.mean([mb.decision_function(XT) for mb in bag_models], axis=0)

    def rank(v):
        return pd.Series(v).rank(pct=True).to_numpy()

    lin = rank(test_preds[f"logistic_C{best_C}"])
    results_series = pd.Series(results).sort_values(ascending=False)
    print("\n" + results_series.to_string())
    best = results_series.index[0]
    print(f"\nbest: {best} {results[best]:.10f} -> implied LB {results[best] + LB_OFFSET:.5f}")

    sub = pd.DataFrame({"id": test_ids, TARGET: rank(test_preds[best])})
    assert len(sub) == len(test_ids) and sub[TARGET].between(0, 1).all()
    sub.to_csv(OUT / "stack_weights_submission.csv", index=False)
    (OUT / "stack_weights_manifest.json").write_text(json.dumps({
        "provenance": "own official-data-only streams; no public predictions",
        "streams": names,
        "chosen": best,
        "results": {k: float(v) for k, v in results.items()},
        "implied_lb": float(results[best] + LB_OFFSET),
        "rows": len(sub),
        "test_id_sha256": hashlib.sha256(np.asarray(test_ids, dtype=np.int64).tobytes()).hexdigest()[:16],
        "submitted": False,
    }, indent=2) + "\n")
    print("wrote", OUT / "stack_weights_submission.csv")


if __name__ == "__main__":
    main()
