#!/usr/bin/env python3
"""Nested rank-blend audit for newly completed original model streams."""
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ensemble_original/reports"
train = pd.read_csv(ROOT / "train.csv")
y = train.addicted_label.to_numpy("int8")


def load(path, col="pred"):
    x = pd.read_csv(ROOT / path).set_index("id").reindex(train.id)[col]
    assert x.notna().all() and np.isfinite(x).all(), path
    return x.rank(method="average", pct=True).to_numpy(float)


new_xgb = load("xgb_nested_hpo/output/oof_nested_hpo_xgb.csv")
old_xgb = load("artifacts/xgb_te_5fold/oof_nested_te_xgb.csv")
nested_cat = load("gpu_catboost_te/output/oof_nested_te_catboost.csv")
lookup_v1 = load("gpu_lookup_original/output/oof_lookup_transformer.csv")
dual_cat = load("gpu_catboost_dual/output/oof_dual_view_catboost.csv")
driver = load("lgb_driver_reconstruction/output/oof_driver_reconstruction_lgb.csv")
pre = .83*(.4*new_xgb + .3*old_xgb + .05*nested_cat + .25*lookup_v1) + .15*dual_cat + .02*driver

v2 = np.mean([load(p) for p in [
    "artifacts/local_lookup_v2/oof_lookup_v2.csv",
    "artifacts/local_lookup_v2_seed81/oof_lookup_v2.csv",
    "artifacts/local_lookup_v2_seed1037/oof_lookup_v2.csv",
    "gpu_lookup_v2_seed42/output/oof_lookup_v2.csv",
    "gpu_lookup_v2_seed959/output/oof_lookup_v2.csv",
]], axis=0)
alt_xgb = load("xgb_d7_altseed/output/oof_d7_altseed_xgb.csv", "pred_d7_altseed")
alt_cat = load("gpu_catboost_dual_seed81/output/oof_dual_view_catboost.csv")
extras = {
    "realmlp_lattice": load("realmlp_lattice_honest/output/oof_realmlp_lattice.csv"),
    "lookup_v3_evidence": load("artifacts/local_lookup_v3_evidence/oof_lookup_v3_evidence.csv"),
    "xgb_dart": load("xgb_dart_nested/output/oof_nested_dart_xgb.csv"),
    "pair_evidence_catboost": load("catboost_pair_evidence_nested/output/oof_nested_pair_evidence_catboost.csv"),
    "fttransformer": load("gpu_fttransformer_nested/output/oof_fttransformer.csv"),
}

# Search is conducted on a fixed balanced subset of each meta-training fold;
# every reported score uses the complete untouched meta-validation fold.
base_grid = list(product(np.arange(.50, .701, .025), np.arange(.05, .151, .025),
                         np.arange(.05, .151, .025)))
extra_grid = np.arange(0, .151, .025)
configs = {"base": [], **{f"plus_{k}": [k] for k in extras},
           "plus_realmlp_and_evidence": ["realmlp_lattice", "lookup_v3_evidence"]}
cv = StratifiedKFold(5, shuffle=True, random_state=20260804)
preds = {k: np.zeros(len(y)) for k in configs}
weight_rows = []

for fold, (fi, vi) in enumerate(cv.split(pre, y), 1):
    rng = np.random.default_rng(6082026 + fold)
    si = np.concatenate([rng.choice(fi[y[fi] == c], 10000, replace=False) for c in (0, 1)])
    for config, names in configs.items():
        best_auc, best = -1., None
        weights = [()] if not names else [z for z in product(extra_grid, repeat=len(names)) if sum(z) <= .15]
        for ew in weights:
            esum = sum(ew)
            for vw, xw, cw in base_grid:
                if vw+xw+cw+esum >= .98:
                    continue
                p = (1-vw-xw-cw-esum)*pre[si] + vw*v2[si] + xw*alt_xgb[si] + cw*alt_cat[si]
                for name, w in zip(names, ew): p += w*extras[name][si]
                auc = roc_auc_score(y[si], p)
                if auc > best_auc: best_auc, best = auc, (vw, xw, cw, *ew)
        vw, xw, cw, *ew = best; esum = sum(ew)
        p = (1-vw-xw-cw-esum)*pre[vi] + vw*v2[vi] + xw*alt_xgb[vi] + cw*alt_cat[vi]
        row = {"configuration": config, "meta_fold": fold, "v2": vw,
               "alt_xgb": xw, "alt_cat": cw, "selection_auc": best_auc}
        for name, w in zip(names, ew): p += w*extras[name][vi]; row[name] = w
        preds[config][vi] = p; weight_rows.append(row)

summary = pd.DataFrame([{"configuration": k, "crossfit_auc": roc_auc_score(y, p)}
                        for k, p in preds.items()]).sort_values("crossfit_auc", ascending=False)
OUT.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT / "completed_candidate_crossfit_auc.csv", index=False)
pd.DataFrame(weight_rows).to_csv(OUT / "completed_candidate_crossfit_weights.csv", index=False)
print(summary.to_string(index=False))
