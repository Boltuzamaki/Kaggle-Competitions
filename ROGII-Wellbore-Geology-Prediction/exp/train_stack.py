"""Honest LightGBM residual stack over domain-physics signals.
GroupKFold-by-well CV; row-weighted (metric) + per-well RMSE; postprocess; submit."""
import sys, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from scipy.signal import savgol_filter
_H = os.path.dirname(os.path.abspath(__file__))

train = pd.read_pickle(f"{_H}/signals/train.pkl")
test = pd.read_pickle(f"{_H}/signals/test.pkl")
DROP = {"well", "id", "target", "last_known_tvt"}
FEATS = [c for c in train.columns if c not in DROP]
print("train rows", len(train), "wells", train["well"].nunique(), "| features", len(FEATS))

X = train[FEATS].to_numpy(np.float32); y = train["target"].to_numpy(np.float32)
g = train["well"].to_numpy()
PARAMS = dict(objective="regression", n_estimators=1200, learning_rate=0.02,
              num_leaves=127, min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=1.0, verbose=-1, n_jobs=-1)

gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(y)); models = []
for k, (tr, va) in enumerate(gkf.split(X, y, g)):
    m = lgb.LGBMRegressor(**PARAMS).fit(X[tr], y[tr])
    oof[va] = m.predict(X[va]); models.append(m); print("fold", k, "done", flush=True)

def row_rmse(t, p): return float(np.sqrt(np.mean((t - p) ** 2)))
def well_rmse(df, p):
    return float(df.assign(e=(df["target"].to_numpy() - p) ** 2)
                 .groupby("well")["e"].mean().pow(0.5).mean())

print("\n=== CV (LightGBM stack) ===")
print("const row/well : %.3f / %.3f" % (row_rmse(y, 0), well_rmse(train, np.zeros(len(y)))))
print("pf   row/well  : %.3f / %.3f" % (row_rmse(y, train['pf_d'].to_numpy()),
                                         well_rmse(train, train['pf_d'].to_numpy())))
print("lgbm row/well  : %.3f / %.3f" % (row_rmse(y, oof), well_rmse(train, oof)))

# postprocess sweep on OOF (shrink toward anchor + clip)
best = (1e9, 1.0, 100)
for sh in [1.0, 0.95, 0.9, 0.85, 0.8, 0.7]:
    for cl in [40, 60, 100]:
        p = np.clip(oof * sh, -cl, cl)
        r = row_rmse(y, p)
        if r < best[0]:
            best = (r, sh, cl)
print("best pp: row %.3f at shrink=%.2f clip=%d" % best)
SH, CL = best[1], best[2]

# fit on all data, predict test, postprocess + savgol per well
full = lgb.LGBMRegressor(**PARAMS).fit(X, y)
tp = np.clip(full.predict(test[FEATS].to_numpy(np.float32)) * SH, -CL, CL)
test = test.assign(pred_d=tp)
rows = []
for w, gdf in test.groupby("well", sort=False):
    v = gdf["pred_d"].to_numpy(float)
    n = len(v); wl = min(31, n if n % 2 == 1 else n - 1)
    if wl >= 5:
        v = savgol_filter(v, wl, 3)
    tvt = gdf["last_known_tvt"].to_numpy(float) + v
    rows.append(pd.DataFrame({"id": gdf["id"].to_numpy(), "tvt": tvt}))
pred = pd.concat(rows, ignore_index=True)

sample = pd.read_csv("data/sample_submission.csv")
sub = sample[["id"]].merge(pred, on="id", how="left")
sub["tvt"] = sub["tvt"].fillna(test["last_known_tvt"].mean())
sub.to_csv("submission_stack.csv", index=False)
print("wrote submission_stack.csv rows", len(sub), "nan", sub["tvt"].isna().sum())

imp = pd.Series(full.feature_importances_, index=FEATS).sort_values(ascending=False)
print("\ntop features:\n", imp.head(15).to_string())
