"""Full 765-well audit of anchored main-target representations."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/harshini_target_representation"
OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_pickle(ROOT / "exp/results/harshini_cached_xgb/rows.pkl")
features = [c for c in d if c not in ("well", "flat", "target")]
X = d[features].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
y = d.target.to_numpy(np.float32)
groups = d.well.to_numpy()
md = np.maximum(d.md_since.to_numpy(np.float32), 10.)
frac = np.maximum(d.s.to_numpy(np.float32), .01)
dz = d.dZ.to_numpy(np.float32)
warm = (1-np.exp(-np.maximum(d.md_since.to_numpy(float), 0)/85.)).astype(np.float32)
physics = (warm*d.blend_d.to_numpy(float)).astype(np.float32)

# Each representation is (training label, reconstruction function). Fixed
# clipping is deliberately broad and target-independent.
representations = {
    "direct_y": (
        np.clip(y, -80, 80),
        lambda p, ix: p),
    "direct_physics_residual": (
        np.clip(y-physics, -60, 60),
        lambda p, ix: physics[ix]+p),
    "tvt_slope_y_per_md": (
        np.clip(y/md, -2, 2),
        lambda p, ix: p*md[ix]),
    "surface_U_slope": (
        np.clip((y+dz)/md, -2, 2),
        lambda p, ix: p*md[ix]-dz[ix]),
    "delta_per_suffix_fraction": (
        np.clip(y/frac, -100, 100),
        lambda p, ix: p*frac[ix]),
    "warm_normalized_physics_residual": (
        np.clip((y-physics)/np.maximum(warm, .1), -60, 60),
        lambda p, ix: physics[ix]+p*warm[ix]),
}

def model(seed):
    return lgb.LGBMRegressor(
        objective="huber", n_estimators=420, learning_rate=.035,
        num_leaves=28, max_depth=8, min_child_samples=110,
        max_bin=127, colsample_bytree=.70, subsample=.85, subsample_freq=1,
        reg_alpha=2., reg_lambda=16., verbosity=-1, n_jobs=12,
        random_state=seed)

def rmse(p, ix):
    return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))

preds = {k: np.zeros(len(d), np.float32) for k in representations}
fold_rows = []
t0 = time.time()
for fold, (tr, va) in enumerate(GroupKFold(5).split(X, groups=groups)):
    fit = tr[tr % 8 == fold % 8]
    row = {"fold": fold, "valid_wells": int(np.unique(groups[va]).size),
           "physics": rmse(physics, va)}
    for ri, (name, (label, reconstruct)) in enumerate(representations.items()):
        m = model(1000*fold+ri)
        m.fit(X[fit], label[fit])
        raw = m.predict(X[va])
        preds[name][va] = reconstruct(raw, va)
        row[name] = rmse(preds[name], va)
    fold_rows.append(row)
    print(row, flush=True)

allix = np.arange(len(d))
scores = {k: rmse(v, allix) for k, v in preds.items()}
direct = scores["direct_y"]
summary = {
    "protocol": "5-fold GroupKFold by complete well; stride-8 fitting; all held-out suffix rows scored",
    "rows": len(d), "wells": int(d.well.nunique()), "features": len(features),
    "physics_rmse": rmse(physics, allix), "scores": scores,
    "gains_vs_direct": {k: direct-v for k, v in scores.items()},
    "folds": fold_rows, "seconds": time.time()-t0,
}
(OUT/"summary.json").write_text(json.dumps(summary, indent=2))
np.savez_compressed(OUT/"oof.npz", y=y, physics=physics, **preds)
print(json.dumps(summary, indent=2), flush=True)
