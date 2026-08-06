"""Strict nested auxiliary-future features for Harshini XGBoost.

Outer folds match the public GroupKFold. Inner OOF auxiliary predictions are
created only for the stride-8 outer-training rows. Auxiliary models fitted on
outer-training wells then predict every row of the outer-validation wells.
"""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"exp/results/harshini_rebuild"
OUT = ROOT/"exp/results/harshini_aux_nested_xgb"
OUT.mkdir(parents=True, exist_ok=True)
f = pd.read_pickle(SRC/"train_frame.pkl")
features = json.loads((SRC/"features.json").read_text())
X = f[features].to_numpy(np.float32)
y = f.target.to_numpy(np.float32)
g = f.well.to_numpy()
warm = 1-np.exp(-np.maximum(f.md_since.to_numpy(float), 0)/85.)
blend = f.blend_d.to_numpy(float)
sample = np.arange(len(f)) % 8 == 0
si = np.flatnonzero(sample)

def future_mean(a, h):
    c = np.r_[0., np.cumsum(a, dtype=float)]
    i = np.arange(len(a)); j = np.minimum(len(a), i+h)
    return ((c[j]-c[i])/np.maximum(1, j-i)).astype(np.float32)

aux = np.zeros((len(f), 4), np.float32)
for _, q in f.groupby("well", sort=False):
    ix = q.index.to_numpy()
    a = y[ix]
    aux[ix, 0] = future_mean(a, 64)
    aux[ix, 1] = future_mean(a, 256)
    aux[ix, 2] = future_mean(a, len(a))  # remaining-zone/toe bias
    j = np.minimum(np.arange(len(a))+256, len(a)-1)
    # U change = TVT change + Z change; flat datum cancels.
    aux[ix, 3] = a[j]-a + q.dZ.to_numpy(np.float32)[j]-q.dZ.to_numpy(np.float32)

def amodel(seed):
    return lgb.LGBMRegressor(
        objective="huber", n_estimators=220, learning_rate=.04,
        num_leaves=24, max_depth=7, min_child_samples=120,
        subsample=.8, subsample_freq=1, colsample_bytree=.65,
        reg_alpha=2., reg_lambda=15., verbosity=-1, n_jobs=8,
        random_state=seed)

pred = np.zeros(len(f), np.float32)
fold_id = np.full(len(f), -1, np.int8)
t0 = time.time()
for k, (tr0, va0) in enumerate(GroupKFold(5).split(X[sample], y[sample], g[sample])):
    tr_global = si[tr0]
    valid_wells = set(g[si[va0]])
    va_mask = np.fromiter((w in valid_wells for w in g), bool, len(g))
    va_global = np.flatnonzero(va_mask)
    atr = np.zeros((len(tr_global), 4), np.float32)
    ava = np.zeros((len(va_global), 4), np.float32)
    inner = GroupKFold(3)
    for a in range(4):
        for j, (it, iv) in enumerate(inner.split(
                X[tr_global], groups=g[tr_global])):
            m = amodel(1000*k+100*a+j)
            m.fit(X[tr_global[it]], aux[tr_global[it], a])
            atr[iv, a] = m.predict(X[tr_global[iv]])
        m = amodel(9000+100*k+a)
        m.fit(X[tr_global], aux[tr_global, a])
        ava[:, a] = m.predict(X[va_global])
    main = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=1800,
        learning_rate=.025, max_depth=8, min_child_weight=60,
        subsample=.8, colsample_bytree=.70, reg_alpha=1.,
        reg_lambda=12., max_bin=256, tree_method="hist", device="cuda",
        n_jobs=8, random_state=80730+k)
    main.fit(np.c_[X[tr_global], atr], y[tr_global])
    pred[va_global] = main.predict(np.c_[X[va_global], ava])
    fold_id[va_global] = k
    print("fold", k, "rmse",
          np.sqrt(np.mean((warm[va_global]*pred[va_global]-y[va_global])**2)),
          flush=True)

def score(p): return float(np.sqrt(np.mean((warm*p-y)**2)))
Z = np.c_[blend, pred]
r = Ridge(alpha=1., positive=True, fit_intercept=False).fit(Z, y)
summary = dict(
    flat=float(np.sqrt(np.mean(y*y))), blend=score(blend),
    aux_xgb=score(pred), ridge_blend_aux_xgb=score(r.predict(Z)),
    weights={"blend": float(r.coef_[0]), "aux_xgb": float(r.coef_[1])},
    rows=len(f), wells=int(f.well.nunique()), seconds=time.time()-t0)
np.savez_compressed(OUT/"aux_xgb_oof.npz", oof=pred, fold=fold_id,
                    y=y, warm=warm, blend=blend)
(OUT/"summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
