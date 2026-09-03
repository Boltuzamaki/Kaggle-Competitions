#!/usr/bin/env python3
"""Audit and blend only locally trained S6E8 OOF/test prediction pairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "reports"
TARGET = "addicted_label"
SEED = 20260803

# Deliberate allow-list: every entry must be a from-scratch competition-data model.
MODELS = {
    "xgb_te_5fold": ("artifacts/xgb_te_5fold/oof_nested_te_xgb.csv", "artifacts/xgb_te_5fold/submission_nested_te_xgb.csv"),
    "xgb_te_4fold": ("artifacts/kaggle_te_xgb/oof_nested_te_xgb.csv", "artifacts/kaggle_te_xgb/submission_nested_te_xgb.csv"),
    "catboost_unique": ("artifacts/catboost_unique/oof_catboost_unique.csv", "artifacts/catboost_unique/submission_catboost_unique.csv"),
    "histgb_5fold": ("artifacts/cpu_histgb_5fold/oof.csv", "artifacts/cpu_histgb_5fold/test_predictions.csv"),
    "neural_5fold": ("artifacts/neural_tabular/oof_neural.csv", "artifacts/neural_tabular/test_neural.csv"),
    "raw_xgb_bag": ("artifacts/raw_xgb_bag/oof_raw_xgb_bag.csv", "artifacts/raw_xgb_bag/test_raw_xgb_bag.csv"),
    "raw_lgb_bag_d4": ("artifacts/raw_lgb_bag/oof_raw_lgb_bag.csv", "artifacts/raw_lgb_bag/test_raw_lgb_bag.csv", "pred_d4_l15", "pred_d4_l15"),
    "raw_lgb_bag_d6": ("artifacts/raw_lgb_bag/oof_raw_lgb_bag.csv", "artifacts/raw_lgb_bag/test_raw_lgb_bag.csv", "pred_d6_l31", "pred_d6_l31"),
    "lgb_te_5fold": ("artifacts/cpu_lgb_te_5fold/oof.csv", "artifacts/cpu_lgb_te_5fold/test_predictions.csv"),
    "tabm_rank1_v2": ("artifacts/tabm_rank1_v2/oof_tabm_rank1.csv", "artifacts/tabm_rank1_v2/test_tabm_rank1.csv"),
    "repr_lgb_global": ("artifacts/representation_lgbm/oof_representation_lgbm.csv", "artifacts/representation_lgbm/test_representation_lgbm.csv", "pred_global", "pred_global"),
    "repr_lgb_experts": ("artifacts/representation_lgbm/oof_representation_lgbm.csv", "artifacts/representation_lgbm/test_representation_lgbm.csv", "pred_regime_experts", "pred_regime_experts"),
    "repr_lgb_blend_e02": ("artifacts/representation_lgbm/oof_representation_lgbm.csv", "artifacts/representation_lgbm/test_representation_lgbm.csv", "blend_expert_0.2", ("blend", 0.2)),
    "repr_lgb_blend_e04": ("artifacts/representation_lgbm/oof_representation_lgbm.csv", "artifacts/representation_lgbm/test_representation_lgbm.csv", "blend_expert_0.4", ("blend", 0.4)),
    "repr_lgb_blend_e06": ("artifacts/representation_lgbm/oof_representation_lgbm.csv", "artifacts/representation_lgbm/test_representation_lgbm.csv", "blend_expert_0.6", ("blend", 0.6)),
    "repr_lgb_blend_e08": ("artifacts/representation_lgbm/oof_representation_lgbm.csv", "artifacts/representation_lgbm/test_representation_lgbm.csv", "blend_expert_0.8", ("blend", 0.8)),
    "catboost_cpu_5fold": ("artifacts/cpu_cat5/oof.csv", "artifacts/cpu_cat5/test_predictions.csv"),
    "catboost_nested_te_5fold": ("artifacts/nested_te_catboost/oof_nested_te_catboost.csv", "artifacts/nested_te_catboost/test_nested_te_catboost.csv", "pred", TARGET),
    "realmlp_cpu_3fold": ("artifacts/realmlp_cpu/oof_realmlp_td_compact.csv", "artifacts/realmlp_cpu/test_realmlp_best.csv"),
}


def sha_ids(s: pd.Series) -> str:
    return hashlib.sha256(np.asarray(s, dtype=np.int64).tobytes()).hexdigest()[:16]


def prediction_column(df: pd.DataFrame, oof: bool) -> str:
    preferred = ["pred", "prediction", "oof_prediction", TARGET]
    blocked = {"id", "fold", "y"} | ({TARGET} if oof and len(df.columns) > 2 else set())
    hits = [c for c in preferred if c in df and c not in blocked]
    if len(hits) != 1:
        raise ValueError(f"Cannot identify prediction column from {df.columns.tolist()}")
    return hits[0]


def load_pair(name: str, paths: tuple, train: pd.DataFrame, test: pd.DataFrame):
    op, tp = (ROOT / p for p in paths[:2])
    if "public_outputs" in str(op) or "public_outputs" in str(tp):
        raise ValueError(f"Public artifact prohibited: {name}")
    o, t = pd.read_csv(op), pd.read_csv(tp)
    for frame, reference, kind in [(o, train, "OOF"), (t, test, "test")]:
        if "id" not in frame or not frame.id.is_unique or len(frame) != len(reference):
            raise ValueError(f"{name} {kind}: invalid IDs/row count")
        if set(frame.id) != set(reference.id):
            raise ValueError(f"{name} {kind}: ID set mismatch")
    o = o.set_index("id").reindex(train.id).reset_index()
    t = t.set_index("id").reindex(test.id).reset_index()
    oc = paths[2] if len(paths) > 2 else prediction_column(o, True)
    tc = paths[3] if len(paths) > 3 else prediction_column(t, False)
    if oc not in o:
        raise ValueError(f"{name}: requested OOF stream {oc!r} is absent")
    opred = o[oc].to_numpy(float)
    if isinstance(tc, tuple) and tc[0] == "blend":
        weight = float(tc[1])
        tpred = ((1 - weight) * t["pred_global"] + weight * t["pred_regime_experts"]).to_numpy(float)
    else:
        if tc not in t:
            raise ValueError(f"{name}: requested test stream {tc!r} is absent")
        tpred = t[tc].to_numpy(float)
    if not np.isfinite(opred).all() or not np.isfinite(tpred).all():
        raise ValueError(f"{name}: non-finite prediction")
    # If labels/folds are saved, verify them rather than trusting row order.
    label_col = "y" if "y" in o else TARGET if TARGET in o and oc != TARGET else None
    if label_col and not np.array_equal(o[label_col].to_numpy(int), train[TARGET].to_numpy(int)):
        raise ValueError(f"{name}: saved OOF labels do not align")
    if "fold" in o and (o.fold.isna().any() or (o.fold < 0).any()):
        raise ValueError(f"{name}: some rows lack an OOF fold")
    return opred, tpred, {"oof": str(op.relative_to(ROOT)), "test": str(tp.relative_to(ROOT))}


def transform(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "raw":
        return x
    if kind == "rank":
        return np.column_stack([pd.Series(x[:, j]).rank(pct=True).to_numpy() for j in range(x.shape[1])])
    if kind == "logit":
        return logit(np.clip(x, 1e-6, 1 - 1e-6))
    raise ValueError(kind)


def combine(x: np.ndarray, w: np.ndarray, kind: str) -> np.ndarray:
    z = transform(x, kind) @ w
    return expit(z) if kind == "logit" else z


def fit_weights(x: np.ndarray, y: np.ndarray, kind: str) -> np.ndarray:
    z = transform(x, kind)
    n = z.shape[1]
    # Smooth pairwise ranking surrogate, subsampled deterministically for speed.
    rng = np.random.default_rng(SEED)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    # A fixed pair sample keeps repeated meta-fold/LOO optimization lightweight
    # enough for a local CPU while retaining a stable ranking signal.
    m = min(30_000, len(pos), len(neg))
    pi, ni = rng.choice(pos, m), rng.choice(neg, m)
    d = z[pi] - z[ni]
    def objective(w):
        margin = np.clip(d @ w, -30, 30)
        return np.logaddexp(0, -margin).mean() + 2e-4 * np.square(w).sum()
    def gradient(w):
        margin = np.clip(d @ w, -30, 30)
        return -(d.T @ expit(-margin)) / len(d) + 4e-4 * w
    result = minimize(objective, np.full(n, 1 / n), jac=gradient, method="SLSQP",
                      bounds=[(0, 1)] * n,
                      constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                      options={"maxiter": 150, "ftol": 1e-10})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def slice_name(missing: np.ndarray) -> np.ndarray:
    return np.where(missing == 0, "complete", np.where(missing <= 3, "missing_1_3", "missing_4_plus"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "test.csv")
    assert train.id.is_unique and test.id.is_unique and set(train[TARGET].unique()) <= {0, 1}
    y = train[TARGET].to_numpy(int)
    names, oofs, tests, provenance = [], [], [], {}
    for name, paths in MODELS.items():
        op, tp, prov = load_pair(name, paths, train, test)
        names.append(name); oofs.append(op); tests.append(tp); provenance[name] = prov
    x, xt = np.column_stack(oofs), np.column_stack(tests)

    missing = train.drop(columns=["id", TARGET]).isna().sum(axis=1).to_numpy()
    slices = slice_name(missing)
    rows = []
    for j, name in enumerate(names):
        row = {"model": name, "auc_all": roc_auc_score(y, x[:, j])}
        for s in ["complete", "missing_1_3", "missing_4_plus"]:
            mask = slices == s
            row[f"auc_{s}"] = roc_auc_score(y[mask], x[mask, j])
            row[f"n_{s}"] = int(mask.sum())
        rows.append(row)
    pd.DataFrame(rows).sort_values("auc_all", ascending=False).to_csv(OUT / "model_auc.csv", index=False)
    pd.DataFrame(x, columns=names).corr(method="spearman").to_csv(OUT / "oof_spearman.csv")

    # Nested meta-validation: weights are learned only on four meta-folds and
    # evaluated on the untouched fifth. This guards weight-selection optimism.
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED + 91)
    blend_rows, weight_rows = [], []
    for kind in ["raw", "rank", "logit"]:
        pred = np.zeros(len(y))
        for fold, (tr, va) in enumerate(cv.split(x, y)):
            w = fit_weights(x[tr], y[tr], kind)
            pred[va] = combine(x[va], w, kind)
            weight_rows.append({"kind": kind, "meta_fold": fold, **dict(zip(names, w))})
        blend_rows.append({"blend": f"crossfit_{kind}", "auc": roc_auc_score(y, pred)})
    pd.DataFrame(weight_rows).to_csv(OUT / "crossfit_weights.csv", index=False)

    # Full-OOF weights are deployment weights only; their apparent AUC is marked optimistic.
    candidates = {}
    for kind in ["raw", "rank", "logit"]:
        w = fit_weights(x, y, kind)
        p = combine(x, w, kind)
        candidates[kind] = (w, combine(xt, w, kind))
        blend_rows.append({"blend": f"fullfit_{kind}_optimistic", "auc": roc_auc_score(y, p)})
    pd.DataFrame(blend_rows).to_csv(OUT / "blend_auc.csv", index=False)

    # Leave-one-model-out cross-fitted blend value.
    loo = []
    for omitted in range(len(names)):
        keep = [j for j in range(len(names)) if j != omitted]
        for kind in ["rank", "logit"]:
            pred = np.zeros(len(y))
            for tr, va in cv.split(x, y):
                w = fit_weights(x[tr][:, keep], y[tr], kind)
                pred[va] = combine(x[va][:, keep], w, kind)
            loo.append({"omitted": names[omitted], "kind": kind, "auc": roc_auc_score(y, pred)})
    pd.DataFrame(loo).to_csv(OUT / "leave_one_out.csv", index=False)

    best_kind = max([r for r in blend_rows if r["blend"].startswith("crossfit")], key=lambda r: r["auc"])["blend"].split("_")[-1]
    # The representation blends are deterministic mixtures of two streams.
    # Compare additions to the pre-existing core instead of letting redundant
    # columns silently degrade the deployment artifact.
    optional_names = {"catboost_cpu_5fold", "realmlp_cpu_3fold"}
    core = [j for j, name in enumerate(names) if not name.startswith("repr_lgb_") and name not in optional_names]
    optional_cols = [j for j, name in enumerate(names) if name.startswith("repr_lgb_") or name in optional_names]
    repr_cols = [j for j in optional_cols if names[j].startswith("repr_lgb_")]
    configurations = [("core", core), ("core_plus_all_repr", core + repr_cols),
                      ("core_plus_all_optional", core + optional_cols)]
    configurations += [(f"core_plus_{names[j]}", core + [j]) for j in optional_cols]
    incremental = []
    for config_name, keep in configurations:
        pred = np.zeros(len(y))
        for tr, va in cv.split(x, y):
            cw = fit_weights(x[tr][:, keep], y[tr], "logit")
            pred[va] = combine(x[va][:, keep], cw, "logit")
        incremental.append({"configuration": config_name, "crossfit_logit_auc": roc_auc_score(y, pred)})
    pd.DataFrame(incremental).sort_values("crossfit_logit_auc", ascending=False).to_csv(OUT / "representation_incremental.csv", index=False)
    robust_gain_threshold = 0.00030
    core_auc = next(row["crossfit_logit_auc"] for row in incremental if row["configuration"] == "core")
    best_incremental = max(incremental, key=lambda row: row["crossfit_logit_auc"])
    selected_config = (best_incremental["configuration"]
                       if best_incremental["crossfit_logit_auc"] >= core_auc + robust_gain_threshold
                       else "core")
    selected_keep = dict(configurations)[selected_config]
    selected_w = fit_weights(x[:, selected_keep], y, "logit")
    test_pred = combine(xt[:, selected_keep], selected_w, "logit")
    w = np.zeros(len(names)); w[selected_keep] = selected_w
    pd.DataFrame({"id": test.id, "prediction": test_pred}).to_csv(OUT / "recommended_test_predictions.csv", index=False)
    manifest = {
        "provenance": "original competition-data-only model artifacts; public predictions prohibited",
        "train_id_fingerprint": sha_ids(train.id), "test_id_fingerprint": sha_ids(test.id),
        "models": provenance, "selection": "highest 5-fold cross-fitted logit AUC across core and representation-stream additions",
        "selected_configuration": selected_config,
        "robust_gain_threshold": robust_gain_threshold,
        "best_incremental_gain_vs_core": best_incremental["crossfit_logit_auc"] - core_auc,
        "selected_transform": "logit", "deployment_weights": dict(zip(names, map(float, w))),
        "warning": "recommended_test_predictions.csv is not a Kaggle submission and has not been submitted",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(rows).sort_values("auc_all", ascending=False).to_string(index=False))
    print("\n", pd.DataFrame(blend_rows).to_string(index=False))
    print("\nselected", best_kind, dict(zip(names, np.round(w, 5))))


if __name__ == "__main__":
    main()
