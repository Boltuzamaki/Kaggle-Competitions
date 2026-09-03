#!/usr/bin/env python3
"""Large-slice error audit for leakage-safe original OOF streams."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "reports"
MIN_ROWS = 5_000

spec = importlib.util.spec_from_file_location("audit_blend", HERE / "audit_and_blend.py")
ab = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ab)


def auc(y, p, mask):
    return roc_auc_score(y[mask], p[mask]) if np.unique(y[mask]).size == 2 else np.nan


def main():
    train, test = pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "test.csv")
    y = train[ab.TARGET].to_numpy(int)
    # Analyze the guarded core plus plausible alternative tree representations.
    core_names = [n for n in ab.MODELS if not n.startswith("repr_lgb_") and n not in {"catboost_cpu_5fold", "realmlp_cpu_3fold"}]
    alt_names = ["catboost_cpu_5fold", "repr_lgb_global", "repr_lgb_experts",
                 "lgb_te_5fold", "histgb_5fold", "raw_xgb_bag"]
    names = list(dict.fromkeys(core_names + alt_names))
    oofs = []
    for name in names:
        op, _, _ = ab.load_pair(name, ab.MODELS[name], train, test)
        oofs.append(op)
    x = np.column_stack(oofs)
    core_idx = [names.index(n) for n in core_names]

    cv = StratifiedKFold(5, shuffle=True, random_state=ab.SEED + 91)
    core_pred = np.zeros(len(y))
    meta_fold = np.full(len(y), -1, dtype=np.int8)
    for fold, (fit, val) in enumerate(cv.split(x, y)):
        weights = ab.fit_weights(x[fit][:, core_idx], y[fit], "logit")
        core_pred[val] = ab.combine(x[val][:, core_idx], weights, "logit")
        meta_fold[val] = fold

    features = [c for c in test.columns if c != "id"]
    missing = train[features].isna()
    missing_count = missing.sum(axis=1).to_numpy()
    bits = (missing.to_numpy(np.uint16) * (1 << np.arange(len(features), dtype=np.uint16))).sum(axis=1)
    pattern_counts = pd.Series(bits).value_counts()
    slices = {
        "complete": missing_count == 0,
        "missing_1_3": (missing_count >= 1) & (missing_count <= 3),
        "missing_4_plus": missing_count >= 4,
    }
    for feature in features:
        slices[f"missing::{feature}"] = missing[feature].to_numpy()
        slices[f"available::{feature}"] = ~missing[feature].to_numpy()
    for pattern, count in pattern_counts.items():
        if count >= MIN_ROWS:
            absent = [features[j] for j in range(len(features)) if int(pattern) & (1 << j)]
            label = "pattern::complete" if not absent else "pattern::" + "+".join(absent)
            slices[label] = bits == pattern

    rows = []
    xgb = x[:, names.index("xgb_te_5fold")]
    for label, mask in slices.items():
        n = int(mask.sum())
        if n < MIN_ROWS or np.unique(y[mask]).size < 2:
            continue
        base = auc(y, core_pred, mask)
        row = {"slice": label, "rows": n, "positive_rate": y[mask].mean(),
               "xgb5_auc": auc(y, xgb, mask), "core_auc": base}
        for name in alt_names:
            score = auc(y, x[:, names.index(name)], mask)
            row[f"{name}_auc"] = score
            row[f"{name}_minus_core"] = score - base
        rows.append(row)
    report = pd.DataFrame(rows).sort_values(["rows", "slice"], ascending=[False, True])
    report.to_csv(OUT / "missing_slice_auc.csv", index=False)

    # Row-level residual summaries diagnose ranking failures without pretending
    # that threshold classification accuracy is the competition objective.
    residual = y - core_pred
    residual_rows = []
    for count in sorted(np.unique(missing_count)):
        mask = missing_count == count
        residual_rows.append({"missing_count": int(count), "rows": int(mask.sum()),
                              "core_auc": auc(y, core_pred, mask),
                              "mean_abs_residual": np.abs(residual[mask]).mean(),
                              "mean_signed_residual": residual[mask].mean()})
    pd.DataFrame(residual_rows).to_csv(OUT / "missing_count_residuals.csv", index=False)
    pd.DataFrame({"id": train.id, "meta_fold": meta_fold, "y": y,
                  "core_crossfit_pred": core_pred, "missing_pattern": bits,
                  "missing_count": missing_count}).to_csv(OUT / "core_crossfit_diagnostics.csv", index=False)

    # Hypothesis-generating exact-pattern gates. Pattern definitions are fixed;
    # alternative weights are selected on four meta-folds and applied to the
    # untouched fifth. Because patterns were discovered in this audit, the final
    # estimate still requires confirmation on a new model run or frozen split.
    proposals = [
        ("daily_only__catboost", ["daily_screen_time_hours"], "catboost_cpu_5fold"),
        ("social_gaming__raw_xgb", ["social_media_hours", "gaming_hours"], "raw_xgb_bag"),
        ("notifications_opens__repr_lgb", ["notifications_per_day", "app_opens_per_day"], "repr_lgb_global"),
    ]
    weight_grid = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
    gated = core_pred.copy()
    gate_rows = []
    for rule, absent, alternative in proposals:
        pattern = sum(1 << features.index(feature) for feature in absent)
        slice_mask = bits == pattern
        alt = x[:, names.index(alternative)]
        for fold in range(5):
            fit_mask = slice_mask & (meta_fold != fold)
            val_mask = slice_mask & (meta_fold == fold)
            fit_core = logit(np.clip(core_pred[fit_mask], 1e-6, 1 - 1e-6))
            fit_alt = logit(np.clip(alt[fit_mask], 1e-6, 1 - 1e-6))
            scores = []
            for weight in weight_grid:
                candidate = expit((1 - weight) * fit_core + weight * fit_alt)
                scores.append(roc_auc_score(y[fit_mask], candidate))
            selected_weight = float(weight_grid[int(np.argmax(scores))])
            val_candidate = expit((1 - selected_weight) * logit(np.clip(core_pred[val_mask], 1e-6, 1 - 1e-6))
                                  + selected_weight * logit(np.clip(alt[val_mask], 1e-6, 1 - 1e-6)))
            gated[val_mask] = val_candidate
            gate_rows.append({"rule": rule, "absent_features": "+".join(absent),
                              "alternative": alternative, "meta_fold": fold,
                              "fit_rows": int(fit_mask.sum()), "validation_rows": int(val_mask.sum()),
                              "selected_weight": selected_weight,
                              "validation_core_auc": auc(y, core_pred, val_mask),
                              "validation_gated_auc": auc(y, gated, val_mask)})
    pd.DataFrame(gate_rows).to_csv(OUT / "conditional_gate_folds.csv", index=False)
    summary = pd.DataFrame([
        {"prediction": "core", "auc": roc_auc_score(y, core_pred)},
        {"prediction": "three_pattern_nested_weight_gate", "auc": roc_auc_score(y, gated)},
    ])
    summary["gain_vs_core"] = summary.auc - summary.loc[summary.prediction.eq("core"), "auc"].iloc[0]
    summary.to_csv(OUT / "conditional_gate_summary.csv", index=False)


if __name__ == "__main__":
    main()
