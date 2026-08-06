"""Cross-fitted meta blend of all independent Harshini and Pilkwang OOF legs."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
H = ROOT/"exp/results/harshini_cached_xgb"
P = ROOT/"exp/public_artifacts/pilkwang/oof"
OUT = ROOT/"exp/results/harshini_alllegs_meta"
OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_pickle(H/"rows.pkl")
y = d.target.to_numpy(float)
g = d.well.to_numpy()
warm = 1-np.exp(-np.maximum(d.md_since.to_numpy(float), 0)/85.)

# Align Pilkwang OOF by verified well + official row order.
gt = pd.read_parquet(P/"train_gt.parquet")
pp = np.load(P/"blend_oof_postprocessed.npy")
pil = np.empty(len(d), np.float32)
gg = gt.groupby(gt.well_id.astype(str), sort=False).groups
for w, ix in d.groupby("well", sort=False).groups.items():
    hi = np.asarray(ix, int); pi = np.asarray(gg[str(w)], int)
    pi = pi[np.argsort(gt.loc[pi, "row_index"].to_numpy())]
    assert len(hi) == len(pi)
    assert np.allclose(y[hi],
        gt.target_delta_from_last_known.to_numpy()[pi], atol=2e-3)
    pil[hi] = pp[pi]

names = ["physics", "lgb", "xgb", "cat_d8", "cat_d10", "pilkwang"]
Z = np.column_stack([
    warm*d.blend_d.to_numpy(float),
    warm*np.load(H/"lgb_fast_oof.npy"),
    warm*np.load(H/"xgb_oof.npy"),
    warm*np.load(H/"cat_d8_oof.npy"),
    warm*np.load(H/"cat_d10_oof.npy"),
    pil])

def rmse(p, ix=None):
    if ix is None: ix = np.arange(len(y))
    return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))

# Honest second-level CV: meta weights for each validation well are learned
# only from other wells.
meta = np.zeros(len(y)); coefs = []
for k, (tr, va) in enumerate(GroupKFold(5).split(Z, groups=g)):
    r = Ridge(alpha=10., positive=True, fit_intercept=False).fit(Z[tr], y[tr])
    meta[va] = r.predict(Z[va]); coefs.append(r.coef_.tolist())
global_r = Ridge(alpha=10., positive=True, fit_intercept=False).fit(Z, y)
global_p = global_r.predict(Z)

wr = []
base = Z[:, 0]
for w, ix in d.groupby("well", sort=False).groups.items():
    ii = np.asarray(ix, int)
    wr.append({"well": w, "rows": len(ii), "physics": rmse(base, ii),
               "crossfit": rmse(meta, ii), "global": rmse(global_p, ii)})
wr = pd.DataFrame(wr)
worst = wr.nlargest(max(1, int(np.ceil(.1*len(wr)))), "physics")
summary = {
    "individual": {n: rmse(Z[:, i]) for i, n in enumerate(names)},
    "global_weights": dict(zip(names, global_r.coef_.tolist())),
    "global_rmse": rmse(global_p),
    "crossfit_weights": [dict(zip(names, c)) for c in coefs],
    "crossfit_rmse": rmse(meta),
    "crossfit_well_win_rate": float((wr.crossfit < wr.physics).mean()),
    "p90_physics": float(wr.physics.quantile(.9)),
    "p90_crossfit": float(wr.crossfit.quantile(.9)),
    "worst_decile_physics": float(np.sqrt(np.average(
        worst.physics**2, weights=worst.rows))),
    "worst_decile_crossfit": float(np.sqrt(np.average(
        worst.crossfit**2, weights=worst.rows))),
}
np.savez_compressed(OUT/"meta_oof.npz", crossfit=meta, global_pred=global_p,
                    y=y, columns=np.asarray(names), Z=Z)
wr.to_csv(OUT/"well_metrics.csv", index=False)
(OUT/"summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
