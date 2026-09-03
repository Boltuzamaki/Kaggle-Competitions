#!/usr/bin/env python3
"""Cross-fitted stack over every completed own-model stream.

Previous deployments hand-tuned a rank average over about ten streams. This
script instead validates the whole inventory, drops partial/broken artifacts,
and cross-fits several meta-learners so their honest OOF AUCs are comparable.

Official competition data only: nothing under `public_outputs/` may be loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ensemble_original/reports"
TARGET = "addicted_label"
SEED = 20260806

# name -> (oof path, test path, oof column, test column)
REGISTRY: dict[str, tuple[str, str, str, str]] = {
    # --- nested target-encoding tree family (historically the strongest) ---
    "xgb_hpo_d7": ("xgb_nested_hpo/output/oof_nested_hpo_xgb.csv", "xgb_nested_hpo/output/test_nested_hpo_xgb.csv", "pred", "pred"),
    "xgb_te_5fold": ("artifacts/xgb_te_5fold/oof_nested_te_xgb.csv", "artifacts/xgb_te_5fold/submission_nested_te_xgb.csv", "pred", TARGET),
    "xgb_te_4fold": ("artifacts/kaggle_te_xgb/oof_nested_te_xgb.csv", "artifacts/kaggle_te_xgb/submission_nested_te_xgb.csv", TARGET, TARGET),
    "xgb_d7_alt1": ("xgb_d7_altseed/output/oof_d7_altseed_xgb.csv", "xgb_d7_altseed/output/test_d7_altseed_xgb.csv", "pred_d7_altseed", "pred_d7_altseed"),
    "xgb_d7_alt2": ("xgb_d7_altseed2/output/oof_d7_altseed_xgb.csv", "xgb_d7_altseed2/output/test_d7_altseed_xgb.csv", "pred_d7_altseed", "pred_d7_altseed"),
    "xgb_dd_d4": ("xgb_depth_diversity/output/oof_depth_diversity_xgb.csv", "xgb_depth_diversity/output/test_depth_diversity_xgb.csv", "pred_d4_regular", "pred_d4_regular"),
    "xgb_dd_d5": ("xgb_depth_diversity/output/oof_depth_diversity_xgb.csv", "xgb_depth_diversity/output/test_depth_diversity_xgb.csv", "pred_d5_balanced", "pred_d5_balanced"),
    "xgb_dd_d6": ("xgb_depth_diversity/output/oof_depth_diversity_xgb.csv", "xgb_depth_diversity/output/test_depth_diversity_xgb.csv", "pred_d6_strong", "pred_d6_strong"),
    # --- CatBoost family ---
    "cat_nested_te": ("gpu_catboost_te/output/oof_nested_te_catboost.csv", "gpu_catboost_te/output/test_nested_te_catboost.csv", "pred", TARGET),
    "cat_dual_view": ("gpu_catboost_dual/output/oof_dual_view_catboost.csv", "gpu_catboost_dual/output/test_dual_view_catboost.csv", "pred", TARGET),
    "cat_dual_seed81": ("gpu_catboost_dual_seed81/output/oof_dual_view_catboost.csv", "gpu_catboost_dual_seed81/output/test_dual_view_catboost.csv", "pred", TARGET),
    "cat_unique": ("artifacts/catboost_unique/oof_catboost_unique.csv", "artifacts/catboost_unique/submission_catboost_unique.csv", "oof_prediction", TARGET),
    "cat_cpu5": ("artifacts/cpu_cat5/oof.csv", "artifacts/cpu_cat5/test_predictions.csv", "prediction", TARGET),
    "cat_pair_evidence": ("artifacts/pending_dl/s6e8-nested-pair-evidence-catboost-private/oof_nested_pair_evidence_catboost.csv", "artifacts/pending_dl/s6e8-nested-pair-evidence-catboost-private/test_nested_pair_evidence_catboost.csv", "pred", "pred"),
    # --- LightGBM / HistGB family ---
    "lgb_te_5fold": ("artifacts/cpu_lgb_te_5fold/oof.csv", "artifacts/cpu_lgb_te_5fold/test_predictions.csv", "prediction", TARGET),
    "lgb_pair_lattice": ("lgb_pair_lattice/output/oof_pair_lattice_lgb.csv", "lgb_pair_lattice/output/test_pair_lattice_lgb.csv", "pred", "pred"),
    "lgb_driver_recon": ("lgb_driver_reconstruction/output/oof_driver_reconstruction_lgb.csv", "lgb_driver_reconstruction/output/test_driver_reconstruction_lgb.csv", "pred", "pred"),
    "lgb_missing_global": ("missing_count_specialists/output/oof_missing_count_specialists.csv", "missing_count_specialists/output/test_missing_count_specialists.csv", "pred_global", "pred_global"),
    "lgb_raw_d4": ("artifacts/raw_lgb_bag/oof_raw_lgb_bag.csv", "artifacts/raw_lgb_bag/test_raw_lgb_bag.csv", "pred_d4_l15", "pred_d4_l15"),
    "lgb_raw_d6": ("artifacts/raw_lgb_bag/oof_raw_lgb_bag.csv", "artifacts/raw_lgb_bag/test_raw_lgb_bag.csv", "pred_d6_l31", "pred_d6_l31"),
    "histgb_5fold": ("artifacts/cpu_histgb_5fold/oof.csv", "artifacts/cpu_histgb_5fold/test_predictions.csv", "prediction", TARGET),
    "xgb_raw_bag": ("artifacts/raw_xgb_bag/oof_raw_xgb_bag.csv", "artifacts/raw_xgb_bag/test_raw_xgb_bag.csv", "pred", "pred"),
    "repr_lgb_global": ("artifacts/representation_lgbm/oof_representation_lgbm.csv", "artifacts/representation_lgbm/test_representation_lgbm.csv", "pred_global", "pred_global"),
    # --- exact-value lookup transformer family (strongest own single models) ---
    "lookup_v1": ("gpu_lookup_original/output/oof_lookup_transformer.csv", "gpu_lookup_original/output/test_lookup_transformer.csv", "pred", TARGET),
    "lookup_v2_s03": ("artifacts/local_lookup_v2/oof_lookup_v2.csv", "artifacts/local_lookup_v2/test_lookup_v2.csv", "pred", TARGET),
    "lookup_v2_s81": ("artifacts/local_lookup_v2_seed81/oof_lookup_v2.csv", "artifacts/local_lookup_v2_seed81/test_lookup_v2.csv", "pred", TARGET),
    "lookup_v2_s1037": ("artifacts/local_lookup_v2_seed1037/oof_lookup_v2.csv", "artifacts/local_lookup_v2_seed1037/test_lookup_v2.csv", "pred", TARGET),
    "lookup_v2_s42": ("gpu_lookup_v2_seed42/output/oof_lookup_v2.csv", "gpu_lookup_v2_seed42/output/test_lookup_v2.csv", "pred", TARGET),
    "lookup_v2_s959": ("gpu_lookup_v2_seed959/output/oof_lookup_v2.csv", "gpu_lookup_v2_seed959/output/test_lookup_v2.csv", "pred", TARGET),
    "lookup_v3_evidence": ("artifacts/local_lookup_v3_evidence/oof_lookup_v3_evidence.csv", "artifacts/local_lookup_v3_evidence/test_lookup_v3_evidence.csv", "pred", TARGET),
    # --- other neural families ---
    "neural_5fold": ("artifacts/neural_tabular/oof_neural.csv", "artifacts/neural_tabular/test_neural.csv", "pred", TARGET),
    "tabm_rank1": ("artifacts/tabm_rank1_v2/oof_tabm_rank1.csv", "artifacts/tabm_rank1_v2/test_tabm_rank1.csv", "pred", TARGET),
    "tabm_missing": ("artifacts/tabm_missing_v2/oof_tabm_missing.csv", "artifacts/tabm_missing_v2/test_tabm_missing.csv", "pred", TARGET),
    "realmlp_lattice": ("realmlp_lattice_honest/output/oof_realmlp_lattice.csv", "realmlp_lattice_honest/output/test_realmlp_lattice.csv", "pred", TARGET),
    "deepfm_exact": ("artifacts/local_deepfm_exact/oof_exact_deepfm.csv", "artifacts/local_deepfm_exact/test_exact_deepfm.csv", "pred", TARGET),
    # --- the six streams launched most recently (weak standalone, target-free) ---
    "fttransformer": ("artifacts/pending_dl/s6e8-nested-ft-transformer-private/oof_fttransformer.csv", "artifacts/pending_dl/s6e8-nested-ft-transformer-private/test_fttransformer.csv", "pred", TARGET),
    "dcnv2_cross": ("artifacts/pending_dl/s6e8-dcnv2-cross-network-private/oof_dcnv2_cross.csv", "artifacts/pending_dl/s6e8-dcnv2-cross-network-private/test_dcnv2_cross.csv", "pred", TARGET),
    "gandalf_gflu": ("artifacts/pending_dl/s6e8-gandalf-gflu-private/oof_gandalf_gflu.csv", "artifacts/pending_dl/s6e8-gandalf-gflu-private/test_gandalf_gflu.csv", "pred", TARGET),
    "tabr_retrieval": ("artifacts/pending_dl/s6e8-nested-tabr-retrieval-private/oof_tabr.csv", "artifacts/pending_dl/s6e8-nested-tabr-retrieval-private/test_tabr.csv", "pred", TARGET),
    "extratrees_support": ("artifacts/pending_dl/s6e8-extratrees-support-private/oof_extratrees_support.csv", "artifacts/pending_dl/s6e8-extratrees-support-private/test_extratrees_support.csv", "pred", "pred"),
    "ebm_exact": ("cpu_ebm_exact/output/oof_exact_ebm.csv", "cpu_ebm_exact/output/test_exact_ebm.csv", "pred", TARGET),
    # --- fold-safe out-of-fold target encoding on the rich composition view ---
    "foldsafe_te_xgb": ("artifacts/foldsafe_te_xgb/oof_foldsafe_te_xgb.csv", "artifacts/foldsafe_te_xgb/test_foldsafe_te_xgb.csv", "pred", TARGET),
    "foldsafe_te_xgb_10f": ("gpu_foldsafe_te_xgb/output_10fold/oof_foldsafe_te_xgb_10f.csv", "gpu_foldsafe_te_xgb/output_10fold/test_foldsafe_te_xgb_10f.csv", "pred", TARGET),
    "foldsafe_te_cat": ("artifacts/foldsafe_te_cat/oof_foldsafe_te_cat.csv", "artifacts/foldsafe_te_cat/test_foldsafe_te_cat.csv", "pred", TARGET),
    # These three were registered without the `_screen` suffix the experiment
    # actually writes, so `load_streams` recorded them as "missing" and four
    # complete LightGBM streams at 0.9675 sat unused. There is no v3 artifact on
    # disk at all; v4 and v5 exist but are the catastrophic self-including pair
    # encodings from E043b/E043c (AUC 0.871 and 0.835) and stay out.
    "nb_v0_published": ("publish_lgbm_end_to_end/output/oof_v0_published_screen.csv", "publish_lgbm_end_to_end/output/test_v0_published_screen.csv", "pred", TARGET),
    "nb_v1_oof_te": ("publish_lgbm_end_to_end/output/oof_v1_oof_te_screen.csv", "publish_lgbm_end_to_end/output/test_v1_oof_te_screen.csv", "pred", TARGET),
    "nb_v2_oof_pair_te": ("publish_lgbm_end_to_end/output/oof_v2_oof_pair_te_screen.csv", "publish_lgbm_end_to_end/output/test_v2_oof_pair_te_screen.csv", "pred", TARGET),
    "nb_v6_capacity": ("publish_lgbm_end_to_end/output/oof_v6_capacity_screen.csv", "publish_lgbm_end_to_end/output/test_v6_capacity_screen.csv", "pred", TARGET),
}


def registry_fingerprint() -> str:
    """Changes whenever the stream set changes, so caches cannot go stale."""
    payload = json.dumps(sorted((k, *v) for k, v in REGISTRY.items()), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def gauss_rank(x: np.ndarray) -> np.ndarray:
    """Monotone map to standard-normal scores so differently calibrated
    streams become directly comparable to a linear meta-learner."""
    r = pd.Series(x).rank(method="average").to_numpy()
    return ndtri(r / (len(r) + 1.0)).astype(np.float32)


def load_streams(train: pd.DataFrame, test: pd.DataFrame):
    y = train[TARGET].to_numpy(np.int8)
    names, oof_cols, test_cols, report = [], [], [], []
    for name, (op, tp, oc, tc) in REGISTRY.items():
        if "public_outputs" in op or "public_outputs" in tp:
            raise ValueError(f"public artifact prohibited: {name}")
        opath, tpath = ROOT / op, ROOT / tp
        if not opath.exists() or not tpath.exists():
            report.append({"stream": name, "status": "missing"})
            continue
        o = pd.read_csv(opath).set_index("id")
        t = pd.read_csv(tpath).set_index("id")
        if oc not in o.columns or tc not in t.columns:
            report.append({"stream": name, "status": "bad_column"})
            continue
        o = o[oc].reindex(train.id)
        t = t[tc].reindex(test.id)
        if o.isna().any() or t.isna().any():
            report.append({"stream": name, "status": "incomplete_coverage"})
            continue
        ov, tv = o.to_numpy(float), t.to_numpy(float)
        if not (np.isfinite(ov).all() and np.isfinite(tv).all()):
            report.append({"stream": name, "status": "nonfinite"})
            continue
        auc = roc_auc_score(y, ov)
        if auc < 0.90:
            report.append({"stream": name, "status": "rejected_low_auc", "auc": auc})
            continue
        names.append(name)
        oof_cols.append(gauss_rank(ov))
        test_cols.append(gauss_rank(tv))
        report.append({"stream": name, "status": "ok", "auc": auc})
    return names, np.column_stack(oof_cols), np.column_stack(test_cols), y, pd.DataFrame(report)


def crossfit_logistic(X, y, C, folds):
    """Honest OOF for an L2 logistic meta-learner."""
    pred = np.zeros(len(y))
    for fit, val in folds:
        m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        m.fit(X[fit], y[fit])
        pred[val] = m.decision_function(X[val])
    return pred


def hill_climb(X, y, names, fit_idx, val_idx, rounds=60, seed=SEED):
    """Greedy with-replacement rank blend selected only on `fit_idx`."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(fit_idx, min(150_000, len(fit_idx)), replace=False)
    Xs, ys = X[sub], y[sub]
    counts = np.zeros(X.shape[1])
    current = np.zeros(len(sub))
    best_auc = 0.0
    for _ in range(rounds):
        gains = []
        n = counts.sum()
        for j in range(X.shape[1]):
            cand = (current * n + Xs[:, j]) / (n + 1)
            gains.append(roc_auc_score(ys, cand))
        j = int(np.argmax(gains))
        if gains[j] <= best_auc + 1e-7 and n > 0:
            break
        best_auc = gains[j]
        current = (current * n + Xs[:, j]) / (n + 1)
        counts[j] += 1
    w = counts / counts.sum()
    return w, best_auc


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv", usecols=["id", TARGET])
    test = pd.read_csv(ROOT / "test.csv", usecols=["id"])
    names, X, XT, y, report = load_streams(train, test)
    report.to_csv(OUT / "mega_stack_streams.csv", index=False)
    print(report.to_string(index=False))
    print(f"\nusable streams: {len(names)}")

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = list(cv.split(X, y))
    results = {}

    # Best single stream, as the reference point every stack must beat.
    single = {n: roc_auc_score(y, X[:, i]) for i, n in enumerate(names)}
    best_single = max(single, key=single.get)
    results["best_single_" + best_single] = single[best_single]

    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        auc = roc_auc_score(y, crossfit_logistic(X, y, C, folds))
        results[f"logistic_C{C}"] = auc
        print(f"logistic C={C:<6} crossfit AUC {auc:.10f}")

    # Hill-climb weights are chosen inside each fit fold only.
    hc_pred = np.zeros(len(y))
    hc_weights = []
    for fit, val in folds:
        w, _ = hill_climb(X, y, names, fit, val)
        hc_pred[val] = X[val] @ w
        hc_weights.append(w)
    results["hill_climb"] = roc_auc_score(y, hc_pred)
    print(f"hill_climb crossfit AUC {results['hill_climb']:.10f}")

    pd.DataFrame(np.array(hc_weights), columns=names).to_csv(OUT / "mega_stack_hc_weights.csv", index=False)
    pd.Series(results).sort_values(ascending=False).to_csv(OUT / "mega_stack_auc.csv")
    print("\n" + pd.Series(results).sort_values(ascending=False).to_string())

    # Deploy the better of the two families, refit on all rows.
    best_C = max((k for k in results if k.startswith("logistic_")), key=lambda k: results[k])
    if results[best_C] >= results["hill_climb"]:
        C = float(best_C.split("C")[1])
        model = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(X, y)
        test_pred = model.decision_function(XT)
        chosen, weights = best_C, dict(zip(names, model.coef_[0].astype(float)))
    else:
        w = np.mean(hc_weights, axis=0)
        test_pred = XT @ w
        chosen, weights = "hill_climb", dict(zip(names, w.astype(float)))

    sub = pd.DataFrame({"id": test.id, TARGET: pd.Series(test_pred).rank(pct=True).to_numpy()})
    assert len(sub) == len(test) and sub[TARGET].between(0, 1).all()
    sub.to_csv(OUT / "mega_stack_submission.csv", index=False)
    manifest = {
        "provenance": "own official-data-only streams; no public predictions",
        "streams": names,
        "chosen": chosen,
        "crossfit_auc": results[chosen] if chosen in results else results["hill_climb"],
        "results": {k: float(v) for k, v in results.items()},
        "weights": weights,
        "rows": len(sub),
        "test_id_sha256": hashlib.sha256(np.asarray(test.id, dtype=np.int64).tobytes()).hexdigest()[:16],
        "submitted": False,
    }
    (OUT / "mega_stack_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("chosen", "crossfit_auc", "rows")}, indent=2))


if __name__ == "__main__":
    main()
