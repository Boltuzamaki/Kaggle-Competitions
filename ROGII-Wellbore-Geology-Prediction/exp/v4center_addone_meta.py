"""Strict nested add-one test for the V4-centered complete-well matcher."""
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/v4center_addone_meta"
OUT.mkdir(parents=True, exist_ok=True)
GT = pd.read_parquet(ROOT / "exp/public_artifacts/pilkwang/oof/train_gt.parquet")
yfull = GT.target_delta_from_last_known.to_numpy(float)
n = len(GT)

hr = pd.read_pickle(ROOT / "exp/results/harshini_cached_xgb/rows.pkl")
bywell = {w: q.index.to_numpy() for w, q in GT.groupby("well_id", sort=False)}
hidx = np.empty(len(hr), int)
for w, q in hr.groupby("well", sort=False):
    gi = bywell[w]
    if len(gi) != len(q):
        raise ValueError(f"row count mismatch {w}")
    hidx[q.index.to_numpy()] = gi
common = hidx
y = yfull[common]
wells = GT.well_id.to_numpy()[common]
target_err = float(np.max(np.abs(hr.target.to_numpy(float) - y)))
if target_err > .002:
    raise ValueError(f"Harshini target identity failed: {target_err}")

warm = 1 - np.exp(-np.maximum(hr.md_since.to_numpy(float), 0) / 85.)
sv = joblib.load(ROOT / "r_v4b/stack_v4_oofs.joblib")["oofs"]
pil = np.load(ROOT / "exp/public_artifacts/pilkwang/oof/blend_oof_postprocessed.npy",
              mmap_mode="r")
uz = np.load(ROOT / "exp/results/alignment_unet_v4center/oof_delta.npz")
if len(uz["prediction"]) != n or not np.array_equal(uz["ids"], GT.id.to_numpy()):
    raise ValueError("UNet OOF row identity failed")
unet_target_err = float(np.max(np.abs(uz["target"].astype(float) - yfull)))
if unet_target_err > .002:
    raise ValueError(f"UNet target identity failed: {unet_target_err}")
if np.any(uz["fold"] < 0) or not np.isfinite(uz["prediction"]).all():
    raise ValueError("UNet OOF incomplete/nonfinite")

legs = {
    "har_physics": warm * hr.blend_d.to_numpy(float),
    "har_lgb": warm * np.load(ROOT / "exp/results/harshini_cached_xgb/lgb_fast_oof.npy"),
    "har_xgb": warm * np.load(ROOT / "exp/results/harshini_cached_xgb/xgb_oof.npy"),
    "pil_blend": np.asarray(pil[common]),
    "v4_lgb7": np.asarray(sv["lgb7"])[common],
    "v4center_unet": uz["prediction"][common].astype(float),
}
names = list(legs)
Z = np.column_stack(list(legs.values()))
split = list(GroupKFold(5).split(Z, groups=wells))

def rmse(pred, mask=None):
    if mask is None:
        return float(np.sqrt(np.mean((pred-y)**2)))
    return float(np.sqrt(np.mean((pred[mask]-y[mask])**2)))

def nested(ix):
    pred = np.zeros(len(y))
    fold = np.full(len(y), -1, np.int8)
    weights = []
    for f, (tr, va) in enumerate(split):
        model = Ridge(alpha=100, positive=True, fit_intercept=False)
        model.fit(Z[tr[::8]][:, ix], y[tr[::8]])
        pred[va] = model.predict(Z[va][:, ix])
        fold[va] = f
        weights.append(dict(zip([names[i] for i in ix], model.coef_.tolist())))
    return pred, fold, weights

base, fold, base_weights = nested(list(range(5)))
add, add_fold, add_weights = nested(list(range(6)))
if not np.array_equal(fold, add_fold):
    raise AssertionError("fold mismatch")

rows = []
for w, ii in pd.Series(np.arange(len(y))).groupby(wells):
    jj = ii.to_numpy()
    rows.append({"well": w, "rows": len(jj), "base": rmse(base, jj),
                 "add": rmse(add, jj), "unet": rmse(legs["v4center_unet"], jj)})
wr = pd.DataFrame(rows)
worst = wr.nlargest(max(1, int(np.ceil(.1*len(wr)))), "base")
summary = {
    "rows": len(y), "wells": int(pd.Series(wells).nunique()),
    "identity": {"har_target_max_abs": target_err,
                 "unet_target_max_abs": unet_target_err,
                 "ids_exact": True, "unet_all_folds_complete": True},
    "individual_rmse": {k: rmse(v) for k, v in legs.items()},
    "base_nested_rmse": rmse(base),
    "add_one_nested_rmse": rmse(add),
    "delta_add_minus_base": rmse(add)-rmse(base),
    "base_folds": [rmse(base, fold == f) for f in range(5)],
    "add_one_folds": [rmse(add, fold == f) for f in range(5)],
    "base_weights": base_weights, "add_one_weights": add_weights,
    "well_win_rate": float((wr["add"] < wr["base"]).mean()),
    "p90_base": float(wr.base.quantile(.9)),
    "p90_add": float(wr["add"].quantile(.9)),
    "worst_decile_base": float(np.sqrt(np.average(worst.base**2, weights=worst.rows))),
    "worst_decile_add": float(np.sqrt(np.average(worst["add"]**2, weights=worst.rows))),
}
np.savez_compressed(OUT/"nested_oof.npz", base=base, add_one=add, target=y,
                    global_indices=common, fold=fold)
wr.to_csv(OUT/"well_metrics.csv", index=False)
(OUT/"summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
