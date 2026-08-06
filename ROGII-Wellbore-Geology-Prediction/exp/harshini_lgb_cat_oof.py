"""Fast complementary LightGBM/CatBoost legs and strict OOF blend."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"exp/results/harshini_rebuild"
OUT = ROOT/"exp/results/harshini_three_model"
OUT.mkdir(parents=True, exist_ok=True)
which = sys.argv[1] if len(sys.argv) > 1 else "lgb"
f = pd.read_pickle(SRC/"train_frame.pkl")
features = json.loads((SRC/"features.json").read_text())
X = f[features].to_numpy(np.float32); y = f.target.to_numpy(np.float32)
g = f.well.to_numpy(); sample = np.arange(len(f)) % 8 == 0
si = np.flatnonzero(sample)
warm = 1-np.exp(-np.maximum(f.md_since.to_numpy(float), 0)/85.)
blend = f.blend_d.to_numpy(float)

def make(seed):
    if which == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            objective="regression", n_estimators=1500, learning_rate=.025,
            num_leaves=127, max_depth=-1, min_child_samples=100,
            subsample=.8, subsample_freq=1, colsample_bytree=.65,
            reg_alpha=.5, reg_lambda=10., max_bin=255,
            n_jobs=16, verbosity=-1, random_state=seed)
    if which == "cat":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=2200, depth=8, learning_rate=.035,
            loss_function="RMSE", l2_leaf_reg=6., random_seed=seed,
            task_type="GPU", devices="0", verbose=0,
            random_strength=.5, bootstrap_type="Bernoulli", subsample=.8)
    raise ValueError(which)

oof = np.zeros(len(f), np.float32); fold_id = np.full(len(f), -1, np.int8)
t0 = time.time()
for k, (tr0, va0) in enumerate(GroupKFold(5).split(
        X[sample], y[sample], g[sample])):
    tr = si[tr0]; vw = set(g[si[va0]])
    va = np.fromiter((w in vw for w in g), bool, len(g))
    m = make(5000+k); m.fit(X[tr], y[tr])
    # Batch prediction avoids GPU temporary-memory spikes.
    ix = np.flatnonzero(va)
    for j in range(0, len(ix), 150_000):
        q = ix[j:j+150_000]; oof[q] = m.predict(X[q])
    fold_id[va] = k
    np.savez_compressed(OUT/f"{which}_oof.partial.npz",
                        oof=oof, fold=fold_id)
    print("fold", k, "rmse",
          np.sqrt(np.mean((warm[va]*oof[va]-y[va])**2)), flush=True)

def score(p): return float(np.sqrt(np.mean((warm*p-y)**2)))
np.savez_compressed(OUT/f"{which}_oof.npz", oof=oof, fold=fold_id,
                    y=y, warm=warm, blend=blend)
summary = {"model": which, "rmse": score(oof), "seconds": time.time()-t0}
(OUT/f"{which}_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

# Blend every completed OOF leg.
legs = {"blend": blend}
xp = ROOT/"exp/results/harshini_xgb/xgb_oof.npz"
if xp.exists(): legs["xgb"] = np.load(xp)["oof"]
for name in ("lgb", "cat"):
    p = OUT/f"{name}_oof.npz"
    if p.exists(): legs[name] = np.load(p)["oof"]
Z = np.column_stack(list(legs.values()))
r = Ridge(alpha=1., positive=True, fit_intercept=False).fit(Z, y)
bs = {"rmse": score(r.predict(Z)), "weights": {
    k: float(v) for k, v in zip(legs, r.coef_)},
    "legs": {k: score(v) for k, v in legs.items()}}
(OUT/"blend_summary.json").write_text(json.dumps(bs, indent=2))
print(json.dumps(bs, indent=2))
