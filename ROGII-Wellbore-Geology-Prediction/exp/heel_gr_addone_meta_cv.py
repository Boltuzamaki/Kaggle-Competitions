"""Strict grouped add-one audit for heel-calibrated GR datum posterior."""
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"exp/results/heel_calibrated_gr_datum"
f = pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
sv = joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
q = np.load(OUT/"oof.npz")
y = f.target.to_numpy(float); groups = f.well.to_numpy()
names = ["lgb123", "lgb7", "xgb", "cat"]
Z = np.column_stack([np.asarray(sv[k], float) for k in names])
gr_path = np.asarray(sv["lgb7"], float) + q["correction"].astype(float)
splits = list(GroupKFold(5).split(Z, groups=groups))

def crossfit(A):
    pred = np.zeros(len(y)); coefs = []
    for tr, va in splits:
        m = Ridge(alpha=100, positive=True, fit_intercept=False)
        m.fit(A[tr[::8]], y[tr[::8]])
        pred[va] = m.predict(A[va])
        coefs.append(m.coef_.tolist())
    return pred, coefs

def rmse(p, ix=None):
    if ix is None: ix = np.arange(len(y))
    return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))

bp, bc = crossfit(Z)
ap, ac = crossfit(np.c_[Z, gr_path])
fold_gains = [rmse(bp, va)-rmse(ap, va) for _, va in splits]
grid = []
for w in np.linspace(0, .3, 13):
    grid.append({"weight": float(w), "rmse": rmse((1-w)*bp+w*gr_path)})
summary = {
    "rows": len(y), "wells": int(pd.Series(groups).nunique()),
    "base_crossfit": rmse(bp), "add_gr_crossfit": rmse(ap),
    "gain": rmse(bp)-rmse(ap), "fold_gains": fold_gains,
    "fold_wins": int(sum(x > 0 for x in fold_gains)),
    "base_columns": names, "base_weights": bc, "add_weights": ac,
    "gr_leg_rmse": rmse(gr_path),
    "best_fixed_blend": min(grid, key=lambda d: d["rmse"]),
}
assert np.isfinite(ap).all()
pd.DataFrame(grid).to_csv(OUT/"meta_fixed_blend.csv", index=False)
np.savez_compressed(OUT/"meta_oof.npz", base=bp, add=ap, y=y, gr_path=gr_path)
(OUT/"meta_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
