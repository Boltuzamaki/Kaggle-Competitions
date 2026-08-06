"""Independent whole-well resplit audit for the support-alignment add-one leg."""
from pathlib import Path
import contextlib
import io
import json
import runpy

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/alignment_unet_support"
with contextlib.redirect_stdout(io.StringIO()):
    state = runpy.run_path(str(ROOT / "exp/meta_all_honest_oof.py"))
legs, y, wells, common = (state[k] for k in ("legs", "y", "wells", "common"))
names = ["har_physics", "har_lgb", "har_xgb",
         "pil_blend_oof_postprocessed", "v4_lgb7"]
heel = np.load(ROOT / "exp/results/heel_calibrated_gr_datum/oof.npz")
heel_path = legs["v4_lgb7"] + heel["correction"][common].astype(float)
alignment = np.load(OUT / "oof_delta.npz")["prediction"][common].astype(float)
base = np.column_stack([legs[k] for k in names] + [heel_path])
added = np.column_stack([base, alignment])
unique = np.unique(wells)


def rmse(pred, rows=None):
    if rows is None:
        rows = np.arange(len(y))
    return float(np.sqrt(np.mean((pred[rows]-y[rows])**2)))


def audit(seed):
    folds = KFold(5, shuffle=True, random_state=seed).split(unique)
    bp = np.zeros(len(y)); ap = np.zeros(len(y)); gains = []
    for _, valid_well_index in folds:
        valid_wells = unique[valid_well_index]
        valid = np.flatnonzero(np.isin(wells, valid_wells))
        train = np.flatnonzero(~np.isin(wells, valid_wells))
        mb = Ridge(alpha=100, positive=True, fit_intercept=False).fit(
            base[train[::8]], y[train[::8]])
        ma = Ridge(alpha=100, positive=True, fit_intercept=False).fit(
            added[train[::8]], y[train[::8]])
        bp[valid] = mb.predict(base[valid]); ap[valid] = ma.predict(added[valid])
        gains.append(rmse(bp, valid)-rmse(ap, valid))
    return {"seed": seed, "base": rmse(bp), "added": rmse(ap),
            "gain": rmse(bp)-rmse(ap), "fold_gains": gains,
            "fold_wins": int(sum(g > 0 for g in gains))}


results = [audit(seed) for seed in (17, 71, 271, 2026, 9917)]
summary = {
    "resplits": results,
    "mean_gain": float(np.mean([r["gain"] for r in results])),
    "min_gain": float(np.min([r["gain"] for r in results])),
    "total_fold_wins": int(sum(r["fold_wins"] for r in results)),
    "total_folds": 25,
}
(OUT / "resplit_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
