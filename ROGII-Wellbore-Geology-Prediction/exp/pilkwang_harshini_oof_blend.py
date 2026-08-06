"""Strict common-row OOF complementarity test: Pilkwang vs Harshini."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
H = ROOT/"exp/results/harshini_cached_xgb"
P = ROOT/"exp/public_artifacts/pilkwang/oof"
OUT = ROOT/"exp/results/pilkwang_harshini_oof"
OUT.mkdir(parents=True, exist_ok=True)

d = pd.read_pickle(H/"rows.pkl")
gt = pd.read_parquet(P/"train_gt.parquet")
pp = np.load(P/"blend_oof_postprocessed.npy")

# Align on well and within-well official row order. Verify exact target equality;
# this makes an accidental ordering match impossible.
pil = np.empty(len(d), np.float32)
gt_groups = gt.groupby(gt.well_id.astype(str), sort=False).groups
for w, ix in d.groupby("well", sort=False).groups.items():
    hi = np.asarray(ix, int)
    pi = np.asarray(gt_groups[str(w)], int)
    pi = pi[np.argsort(gt.loc[pi, "row_index"].to_numpy())]
    if len(hi) != len(pi):
        raise RuntimeError(f"row mismatch {w}: {len(hi)} != {len(pi)}")
    if not np.allclose(d.target.to_numpy()[hi],
                       gt.target_delta_from_last_known.to_numpy()[pi],
                       atol=2e-3):
        raise RuntimeError(f"target/order mismatch {w}")
    pil[hi] = pp[pi]

y = d.target.to_numpy(float)
warm = 1-np.exp(-np.maximum(d.md_since.to_numpy(float), 0)/85.)
physics = d.blend_d.to_numpy(float)
xgb = np.load(H/"xgb_oof.npy")
lgb = np.load(H/"lgb_fast_oof.npy")
hcoef = np.asarray(json.loads((H/"lgb_fast_summary.json").read_text())
                   ["ridge_weights"])
har = warm*(np.c_[physics, lgb, xgb]@hcoef)

def rmse(p, mask=None):
    if mask is None:
        return float(np.sqrt(np.mean((y-p)**2)))
    return float(np.sqrt(np.mean((y[mask]-p[mask])**2)))

grid = []
for pw in np.linspace(0, .5, 51):
    q = (1-pw)*har+pw*pil
    grid.append({"pilkwang_weight": pw, "rmse": rmse(q)})
grid = pd.DataFrame(grid).sort_values("rmse")
ridge = Ridge(alpha=1, positive=True, fit_intercept=False).fit(
    np.c_[har, pil], y)
rp = ridge.predict(np.c_[har, pil])
# Cross-fit the two meta weights by complete well to quantify weight-selection
# optimism independently of both supplied base OOF systems.
meta = np.zeros(len(y))
meta_coef = []
groups = d.well.to_numpy()
for tr, va in GroupKFold(5).split(np.c_[har, pil], groups=groups):
    r = Ridge(alpha=1, positive=True, fit_intercept=False).fit(
        np.c_[har[tr], pil[tr]], y[tr])
    meta[va] = r.predict(np.c_[har[va], pil[va]])
    meta_coef.append(r.coef_.tolist())

well_rows = []
for w, ix in d.groupby("well", sort=False).groups.items():
    ii = np.asarray(ix, int)
    well_rows.append({"well": w, "rows": len(ii), "harshini": rmse(har, ii),
                      "pilkwang": rmse(pil, ii), "ridge": rmse(rp, ii)})
wr = pd.DataFrame(well_rows)
worst = wr.nlargest(max(1, int(np.ceil(.1*len(wr)))), "harshini")
summary = {
    "rows": len(d), "wells": int(d.well.nunique()),
    "harshini_rmse": rmse(har), "pilkwang_rmse": rmse(pil),
    "residual_correlation": float(np.corrcoef(y-har, y-pil)[0, 1]),
    "best_grid": grid.iloc[0].to_dict(),
    "ridge_weights": ridge.coef_.tolist(), "ridge_rmse": rmse(rp),
    "crossfit_meta_weights": meta_coef, "crossfit_meta_rmse": rmse(meta),
    "well_win_rate_ridge": float((wr.ridge < wr.harshini).mean()),
    "p90_harshini": float(wr.harshini.quantile(.9)),
    "p90_ridge": float(wr.ridge.quantile(.9)),
    "worst_decile_harshini": float(np.sqrt(np.average(
        worst.harshini**2, weights=worst.rows))),
    "worst_decile_ridge": float(np.sqrt(np.average(
        worst.ridge**2, weights=worst.rows))),
}
print(json.dumps(summary, indent=2))
grid.to_csv(OUT/"blend_grid.csv", index=False)
wr.to_csv(OUT/"well_metrics.csv", index=False)
(OUT/"summary.json").write_text(json.dumps(summary, indent=2))
