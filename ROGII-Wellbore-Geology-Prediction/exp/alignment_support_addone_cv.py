"""Strict add-one audit of the full support-aware alignment OOF leg."""
from pathlib import Path
import contextlib
import io
import json
import runpy

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/alignment_unet_support"

with contextlib.redirect_stdout(io.StringIO()):
    state = runpy.run_path(str(ROOT / "exp/meta_all_honest_oof.py"))
legs, y, wells, common = (state[k] for k in ("legs", "y", "wells", "common"))
base_names = ["har_physics", "har_lgb", "har_xgb",
              "pil_blend_oof_postprocessed", "v4_lgb7"]

heel = np.load(ROOT / "exp/results/heel_calibrated_gr_datum/oof.npz")
heel_path = legs["v4_lgb7"] + heel["correction"][common].astype(float)
alignment_all = np.load(OUT / "oof_delta.npz")["prediction"].astype(float)
alignment = alignment_all[common]
if not np.isfinite(alignment).all():
    raise RuntimeError("alignment OOF is incomplete on common rows")

splits = list(GroupKFold(5).split(y, groups=wells))


def crossfit(matrix):
    pred = np.zeros(len(y))
    coefs = []
    for train, valid in splits:
        model = Ridge(alpha=100, positive=True, fit_intercept=False)
        model.fit(matrix[train[::8]], y[train[::8]])
        pred[valid] = model.predict(matrix[valid])
        coefs.append(model.coef_.tolist())
    return pred, coefs


def rmse(pred, rows=None):
    if rows is None:
        rows = np.arange(len(y))
    return float(np.sqrt(np.mean((pred[rows] - y[rows]) ** 2)))


base = np.column_stack([legs[k] for k in base_names] + [heel_path])
base_pred, base_coef = crossfit(base)
add_pred, add_coef = crossfit(np.column_stack([base, alignment]))
fold_gains = [rmse(base_pred, va) - rmse(add_pred, va) for _, va in splits]
grid = []
for weight in np.linspace(0, .3, 13):
    pred = (1-weight)*base_pred + weight*alignment
    grid.append({"weight": float(weight), "rmse": rmse(pred)})
summary = {
    "accepted_base": rmse(base_pred),
    "alignment_leg": rmse(alignment),
    "add_alignment": rmse(add_pred),
    "gain": rmse(base_pred)-rmse(add_pred),
    "fold_gains": fold_gains,
    "fold_wins": int(sum(g > 0 for g in fold_gains)),
    "best_fixed_blend": min(grid, key=lambda row: row["rmse"]),
    "base_coefficients": base_coef,
    "add_coefficients": add_coef,
}
(OUT / "addone_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
