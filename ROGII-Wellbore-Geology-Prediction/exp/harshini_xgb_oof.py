"""XGBoost third-leg OOF on the exact cached Harshini feature frame."""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "exp/results/harshini_rebuild"
OUT = ROOT / "exp/results/harshini_xgb"
OUT.mkdir(parents=True, exist_ok=True)

frame = pd.read_pickle(SRC / "train_frame.pkl")
features = json.loads((SRC / "features.json").read_text())
X = frame[features].to_numpy(np.float32)
y = frame.target.to_numpy(np.float32)
g = frame.well.to_numpy()
flat = frame.flat.to_numpy(float)
md = frame.md_since.to_numpy(float)
warm = 1-np.exp(-np.maximum(md, 0)/85.)
sample = np.arange(len(frame)) % 8 == 0
si = np.flatnonzero(sample)
oof = np.zeros(len(frame), np.float32)
fold_id = np.full(len(frame), -1, np.int8)
t0 = time.time()

for k, (tr0, va0) in enumerate(GroupKFold(5).split(
        X[sample], y[sample], g[sample])):
    tr = si[tr0]
    valid_wells = set(g[si[va0]])
    va = np.fromiter((w in valid_wells for w in g), bool, len(g))
    m = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=1800,
        learning_rate=.025, max_depth=8, min_child_weight=60,
        subsample=.80, colsample_bytree=.70, reg_alpha=1.,
        reg_lambda=12., max_bin=256, tree_method="hist", device="cuda",
        n_jobs=8, random_state=80730+k)
    m.fit(X[tr], y[tr], verbose=False)
    oof[va] = m.predict(X[va])
    fold_id[va] = k
    score = np.sqrt(np.mean((warm[va]*oof[va]-y[va])**2))
    print("fold", k, "rows", va.sum(), "rmse", score, flush=True)

blend = frame.blend_d.to_numpy(float)
def score(delta): return float(np.sqrt(np.mean((warm*delta-y)**2)))

# Same positive, no-intercept meta-form as the public notebook.
Z = np.c_[blend, oof]
ridge = Ridge(alpha=1., positive=True, fit_intercept=False).fit(Z, y)
pred = ridge.predict(Z)
summary = {
    "rows": len(frame), "wells": int(frame.well.nunique()),
    "flat": float(np.sqrt(np.mean(y*y))),
    "blend": score(blend), "xgb": score(oof), "ridge_blend_xgb": score(pred),
    "weights": {"blend": float(ridge.coef_[0]), "xgb": float(ridge.coef_[1])},
    "seconds": time.time()-t0,
}
np.savez_compressed(OUT/"xgb_oof.npz", oof=oof, fold=fold_id,
                    y=y, flat=flat, warm=warm, blend=blend)
(OUT/"summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
