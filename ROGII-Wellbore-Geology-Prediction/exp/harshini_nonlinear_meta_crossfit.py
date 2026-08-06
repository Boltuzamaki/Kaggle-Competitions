"""Strict grouped nonlinear meta-model over honest Harshini/Pilkwang OOF legs.

All base predictions are OOF at the well level. The second-level models use a
separate GroupKFold and are trained only on other wells.  Training rows are
stride-8, while every validation row is scored using the competition's pooled
RMSE.
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
H = ROOT/"exp/results/harshini_cached_xgb"
P = ROOT/"exp/public_artifacts/pilkwang/oof"
OUT = ROOT/"exp/results/harshini_nonlinear_meta"
OUT.mkdir(parents=True, exist_ok=True)
mode = sys.argv[1] if len(sys.argv) > 1 else "lgb"

d = pd.read_pickle(H/"rows.pkl")
y = d.target.to_numpy(np.float32)
groups = d.well.to_numpy()
warm = 1-np.exp(-np.maximum(d.md_since.to_numpy(float), 0)/85.)
physics = warm*d.blend_d.to_numpy(float)
lgb_o = warm*np.load(H/"lgb_fast_oof.npy")
xgb_o = warm*np.load(H/"xgb_oof.npy")

# Exact Pilkwang alignment by well and official row index.
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

B = np.c_[physics, lgb_o, xgb_o, pil].astype(np.float32)
bm = B.mean(1); bs = B.std(1)
meta = pd.DataFrame({
    "physics": physics, "lgb_oof": lgb_o, "xgb_oof": xgb_o,
    "pilkwang": pil, "base_mean": bm, "base_std": bs,
    "base_range": B.max(1)-B.min(1),
    "lgb_xgb_gap": lgb_o-xgb_o, "physics_ml_gap": physics-(lgb_o+xgb_o)/2,
    "pil_consensus_gap": pil-np.median(B[:, :3], axis=1),
})
# Only inference-available quality/position columns. Avoid identifiers and any
# target-derived columns.
context = [
    "s", "frac2", "sqrt_frac", "md_since", "dZ", "Z", "dip", "z_span",
    "GR", "GR_dev", "pf_std", "pf_gap", "disagree", "imp_spread", "nn",
    "pr_min", "pr_med", "pr_max", "pr_std", "pfx_rmse", "tda_min",
    "tda_argmin", "tda_curv", "ktvt_range", "ktvt_std", "slp_all", "slp_50",
    "dzdmd", "grs21", "grs51", "grs101", "cal_a", "cal_b",
]
for c in context:
    if c in d: meta[c] = d[c].to_numpy(np.float32)
X = meta.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
sample = np.arange(len(d)) % 8 == 0
si = np.flatnonzero(sample)
pred = np.zeros(len(d), np.float32)
base_cf = np.zeros(len(d), np.float32)
coef = []
t0 = time.time()

def fit_predict(k, tr, va):
    if mode in ("res_lgb", "res_xgb"):
        # Fold-local linear baseline, then learn only its residual. This avoids
        # forcing a tree to relearn the accurate absolute trajectory.
        br = Ridge(alpha=10., positive=True, fit_intercept=False).fit(B[tr], y[tr])
        btr, bva = br.predict(B[tr]), br.predict(B[va])
        base_cf[va] = bva
        ry = y[tr]-btr
        if mode == "res_lgb":
            import lightgbm as lgb
            m = lgb.LGBMRegressor(
                objective="regression", n_estimators=500, learning_rate=.02,
                num_leaves=16, max_depth=6, min_child_samples=300,
                subsample=.8, subsample_freq=1, colsample_bytree=.7,
                reg_alpha=5., reg_lambda=50., verbosity=-1, n_jobs=16,
                random_state=7300+k)
        else:
            import xgboost as xgb
            m = xgb.XGBRegressor(
                objective="reg:squarederror", n_estimators=650,
                learning_rate=.02, max_depth=5, min_child_weight=250,
                subsample=.8, colsample_bytree=.7, reg_alpha=5., reg_lambda=50.,
                max_bin=128, tree_method="hist", device="cuda",
                random_state=7400+k)
        m.fit(X[tr], ry)
        corr = np.empty(len(va), np.float32)
        for j in range(0, len(va), 100_000):
            corr[j:j+100_000] = m.predict(X[va[j:j+100_000]])
        return bva+corr
    if mode == "ridge":
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=100., fit_intercept=True).fit(sc.transform(X[tr]), y[tr])
        coef.append(m.coef_.tolist())
        return m.predict(sc.transform(X[va]))
    if mode == "lgb":
        import lightgbm as lgb
        # Direct Huber regression is deliberately shallow: base legs already
        # encode the trajectory; meta should learn confidence/gating only.
        m = lgb.LGBMRegressor(
            objective="huber", n_estimators=700, learning_rate=.025,
            num_leaves=24, max_depth=7, min_child_samples=250,
            subsample=.8, subsample_freq=1, colsample_bytree=.75,
            reg_alpha=3., reg_lambda=30., verbosity=-1, n_jobs=16,
            random_state=7100+k)
        m.fit(X[tr], y[tr])
        return m.predict(X[va])
    if mode == "xgb":
        import xgboost as xgb
        m = xgb.XGBRegressor(
            objective="reg:pseudohubererror", n_estimators=900,
            learning_rate=.025, max_depth=6, min_child_weight=150,
            subsample=.8, colsample_bytree=.75, reg_alpha=3., reg_lambda=30.,
            max_bin=128, tree_method="hist", device="cuda", n_jobs=8,
            random_state=7200+k)
        m.fit(X[tr], y[tr])
        out = np.empty(len(va), np.float32)
        for j in range(0, len(va), 100_000):
            out[j:j+100_000] = m.predict(X[va[j:j+100_000]])
        return out
    raise ValueError(mode)

for k, (tr0, va0) in enumerate(GroupKFold(5).split(
        X[sample], groups=groups[sample])):
    tr = si[tr0]; vw = set(groups[si[va0]])
    va = np.flatnonzero(np.fromiter((w in vw for w in groups), bool, len(groups)))
    pred[va] = fit_predict(k, tr, va)
    print("fold", k, "rmse", np.sqrt(np.mean((pred[va]-y[va])**2)), flush=True)
    np.savez_compressed(OUT/f"{mode}_partial.npz", pred=pred)

def rmse(p, ix=None):
    if ix is None: ix = np.arange(len(y))
    return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))

wr = []
for w, ix in d.groupby("well", sort=False).groups.items():
    ii = np.asarray(ix, int)
    wr.append({"well": w, "rows": len(ii), "physics": rmse(physics, ii),
               "meta": rmse(pred, ii)})
wr = pd.DataFrame(wr)
worst = wr.nlargest(max(1, int(np.ceil(.1*len(wr)))), "physics")
summary = {
    "model": mode, "rmse": rmse(pred), "physics_rmse": rmse(physics),
    "well_win_rate": float((wr.meta < wr.physics).mean()),
    "p90_physics": float(wr.physics.quantile(.9)),
    "p90_meta": float(wr.meta.quantile(.9)),
    "worst_decile_physics": float(np.sqrt(np.average(
        worst.physics**2, weights=worst.rows))),
    "worst_decile_meta": float(np.sqrt(np.average(
        worst.meta**2, weights=worst.rows))),
    "features": list(meta.columns), "seconds": time.time()-t0,
}
if mode.startswith("res_"):
    corr = pred-base_cf
    grid = []
    for a in (0., .1, .2, .3, .4, .5, .65, .8, 1.):
        q = base_cf+a*corr
        grid.append({"correction_weight": a, "rmse": rmse(q)})
    best = min(grid, key=lambda z: z["rmse"])
    pred = base_cf+best["correction_weight"]*corr
    summary["linear_base_rmse"] = rmse(base_cf)
    summary["correction_grid"] = grid
    summary["best_correction"] = best
    summary["rmse"] = rmse(pred)
np.savez_compressed(OUT/f"{mode}_oof.npz", pred=pred, y=y, groups=groups)
wr.to_csv(OUT/f"{mode}_well_metrics.csv", index=False)
(OUT/f"{mode}_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
