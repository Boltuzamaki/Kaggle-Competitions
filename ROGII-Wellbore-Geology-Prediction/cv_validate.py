"""GroupKFold (by well) CV to confirm LightGBM beats the constant baseline."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from wellbore_lib import build_dataset, FEATURES, LGB_PARAMS, postprocess

print("Building dataset...")
df = build_dataset("data/train", has_target=True)
print("rows", len(df), "wells", df['well'].nunique())

X = df[FEATURES].to_numpy()
y = df["d_tvt"].to_numpy()
groups = df["well"].to_numpy()

gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(df))
for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(X[tr], y[tr])
    oof[va] = m.predict(X[va])
    print(f"fold {fold} done")

df["pred_raw"] = oof
np.save("oof.npy", oof)

# --- sweep shrink & clip on per-well RMSE ---
best = None
for shrink in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
    for clip in [40, 60, 100]:
        p = np.clip(oof * shrink, -clip, clip)
        tmp = df.assign(p=p)
        pw_ = tmp.groupby("well").apply(
            lambda g: np.sqrt(np.mean((g["d_tvt"] - g["p"]) ** 2)))
        score = pw_.mean()
        if best is None or score < best[0]:
            best = (score, shrink, clip)
print("best per-well RMSE %.3f at shrink=%.2f clip=%d" % best)
SHRINK, CLIP = best[1], best[2]
df["pred"] = np.clip(oof * SHRINK, -CLIP, CLIP)

def well_rmse(g, col):
    return np.sqrt(np.mean((g["d_tvt"] - g[col]) ** 2))

# per-well RMSE (the competition averages dTVT errors; report both pooled & per-well)
const_pooled = np.sqrt(np.mean(df["d_tvt"] ** 2))
model_pooled = np.sqrt(np.mean((df["d_tvt"] - df["pred"]) ** 2))
raw_pooled = np.sqrt(np.mean((df["d_tvt"] - df["pred_raw"]) ** 2))

pw = df.groupby("well").apply(lambda g: pd.Series({
    "const": well_rmse(g.assign(c=0.0), "c"),
    "model": well_rmse(g, "pred"),
}))
print("\n=== POOLED RMSE (ft) ===")
print(f"  constant (dTVT=0):     {const_pooled:.3f}")
print(f"  model raw:             {raw_pooled:.3f}")
print(f"  model post-processed:  {model_pooled:.3f}")
print("\n=== PER-WELL MEAN RMSE (ft) ===")
print(f"  constant: {pw['const'].mean():.3f}   model: {pw['model'].mean():.3f}")
print(f"  model beats constant in {100*(pw['model']<pw['const']).mean():.0f}% of wells")
