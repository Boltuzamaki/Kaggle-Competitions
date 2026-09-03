#!/usr/bin/env python3
"""Evaluate new independently trained streams against the current S003 stack."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
TARGET = "addicted_label"
train = pd.read_csv(ROOT / "train.csv")
y = train[TARGET].to_numpy()


def values(path, column="pred"):
    frame = pd.read_csv(ROOT / path).set_index("id").reindex(train.id)
    assert frame[column].notna().all(), path
    return frame[column].to_numpy(float)


def rank(x):
    return pd.Series(x).rank(method="average", pct=True).to_numpy()


# Reproduce the fixed, previously selected S003 OOF formula.
new_xgb = rank(values("xgb_nested_hpo/output/oof_nested_hpo_xgb.csv"))
old_xgb = rank(values("artifacts/xgb_te_5fold/oof_nested_te_xgb.csv"))
nested_cat = rank(values("gpu_catboost_te/output/oof_nested_te_catboost.csv"))
lookup_v1 = rank(values("gpu_lookup_original/output/oof_lookup_transformer.csv"))
dual_cat = rank(values("gpu_catboost_dual/output/oof_dual_view_catboost.csv"))
driver = rank(values("lgb_driver_reconstruction/output/oof_driver_reconstruction_lgb.csv"))
pre = .83 * (.4 * new_xgb + .3 * old_xgb + .05 * nested_cat + .25 * lookup_v1) + .15 * dual_cat + .02 * driver

v2_paths_3 = [
    "artifacts/local_lookup_v2/oof_lookup_v2.csv",
    "artifacts/local_lookup_v2_seed81/oof_lookup_v2.csv",
    "artifacts/local_lookup_v2_seed1037/oof_lookup_v2.csv",
]
v2_paths_5 = v2_paths_3 + [
    "gpu_lookup_v2_seed42/output/oof_lookup_v2.csv",
    "gpu_lookup_v2_seed959/output/oof_lookup_v2.csv",
]
v2_3 = np.mean([rank(values(p)) for p in v2_paths_3], axis=0)
v2_5 = np.mean([rank(values(p)) for p in v2_paths_5], axis=0)
alt_xgb = rank(values("xgb_d7_altseed/output/oof_d7_altseed_xgb.csv", "pred_d7_altseed"))
alt_cat = rank(values("gpu_catboost_dual_seed81/output/oof_dual_view_catboost.csv"))
deepfm = rank(values("artifacts/local_deepfm_exact/oof_exact_deepfm.csv"))

rows = []
for bag_name, v2 in (("three_seed", v2_3), ("five_seed", v2_5)):
 for vw in np.arange(.50, .801, .025):
    core = (1 - vw) * pre + vw * v2
    for dw in np.arange(0, .101, .01):
        # Preserve the selected alternative-tree weights and only allocate residual
        # mass from the core to the new stream.
        pred = (1 - .075 - .10 - dw) * core + .075 * alt_xgb + .10 * alt_cat + dw * deepfm
        rows.append({"bag": bag_name, "v2_weight": vw, "deepfm_weight": dw, "auc": roc_auc_score(y, pred)})

report = pd.DataFrame(rows).sort_values("auc", ascending=False)
out = ROOT / "ensemble_original/reports/new_candidate_grid.csv"
report.to_csv(out, index=False)
print("standalone_deepfm", roc_auc_score(y, deepfm))
print(report.head(20).to_string(index=False))

# Nested meta-validation for the only new decision: whether the two completed
# Transformer streams improve the existing three-seed bag. All mixture weights
# are selected without looking at the held-out meta fold.
grid = [(vw, xw, cw) for vw in np.arange(.50, .801, .025)
        for xw in np.arange(0, .151, .025) for cw in np.arange(0, .151, .025)
        if vw + xw + cw <= .95]
meta = StratifiedKFold(5, shuffle=True, random_state=20260804)
cf_rows, cf_pred = [], {"three_seed": np.zeros(len(y)), "five_seed": np.zeros(len(y))}
for bag_name, v2 in (("three_seed", v2_3), ("five_seed", v2_5)):
    for fold, (fi, vi) in enumerate(meta.split(pre, y), 1):
        # A fixed stratified 120k-row subset makes the discrete weight search
        # practical; the selected formula is still scored on the entire held-out
        # meta fold. This sample is used for selection only, never evaluation.
        rng = np.random.default_rng(20260804 + fold)
        si = np.concatenate([rng.choice(fi[y[fi] == label], 60000, replace=False) for label in (0, 1)])
        best = max(grid, key=lambda z: roc_auc_score(
            y[si], (1-z[0]-z[1]-z[2])*pre[si] + z[0]*v2[si] + z[1]*alt_xgb[si] + z[2]*alt_cat[si]))
        vw, xw, cw = best
        cf_pred[bag_name][vi] = (1-vw-xw-cw)*pre[vi] + vw*v2[vi] + xw*alt_xgb[vi] + cw*alt_cat[vi]
        cf_rows.append({"bag": bag_name, "meta_fold": fold, "v2_weight": vw,
                        "alt_xgb_weight": xw, "alt_cat_weight": cw})
cf_summary = pd.DataFrame([
    {"bag": name, "crossfit_auc": roc_auc_score(y, pred)}
    for name, pred in cf_pred.items()
]).sort_values("crossfit_auc", ascending=False)
pd.DataFrame(cf_rows).to_csv(ROOT / "ensemble_original/reports/transformer_bag_crossfit_weights.csv", index=False)
cf_summary.to_csv(ROOT / "ensemble_original/reports/transformer_bag_crossfit_auc.csv", index=False)
print("\nnested transformer-bag comparison")
print(cf_summary.to_string(index=False))
