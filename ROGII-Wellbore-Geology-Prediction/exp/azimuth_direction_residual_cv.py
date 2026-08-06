"""Direction-aware, group-cross-fitted calibration of Stack V4 OOF predictions.

This is a cheap diagnostic for the public NW/SE drilling-direction hint.  It
uses only target-free trajectory/features and existing out-of-fold predictions.
The calibrator itself is cross-fitted by the same supertype grouping as V4.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp" / "results" / "azimuth_direction"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260730


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - p))))


def trajectory_table(wells: list[str]) -> pd.DataFrame:
    rows = []
    for well in wells:
        path = ROOT / "data" / "train" / f"{well}__horizontal_well.csv"
        d = pd.read_csv(path, usecols=["MD", "X", "Y", "Z"])
        # Use endpoints only: available for train and test and independent of TVT.
        dx = float(d["X"].iloc[-1] - d["X"].iloc[0])
        dy = float(d["Y"].iloc[-1] - d["Y"].iloc[0])
        length = max(float(np.hypot(dx, dy)), 1e-6)
        rows.append(
            {
                "well": well,
                "dir_cos": dx / length,
                "dir_sin": dy / length,
                "azimuth_deg": float(np.degrees(np.arctan2(dy, dx)) % 360),
                "direction": "NW" if dy >= 0 else "SE",
            }
        )
    return pd.DataFrame(rows).set_index("well")


df = pd.read_pickle(ROOT / "r_v4b" / "train_feats.pkl")
bundle = joblib.load(ROOT / "r_v4b" / "stack_v4_oofs.joblib")
y = df["target"].to_numpy(np.float32)
base = np.asarray(bundle["oofs"]["lgb7"], dtype=np.float32)
wells = df["well"].astype(str)
traj = trajectory_table(wells.drop_duplicates().tolist())

direction = wells.map(traj["direction"]).to_numpy()
dir_cos = wells.map(traj["dir_cos"]).to_numpy(np.float32)
dir_sin = wells.map(traj["dir_sin"]).to_numpy(np.float32)
frac = df["frac"].to_numpy(np.float32)

# A deliberately compact matrix. Interactions let one Ridge model express two
# direction-specific trends without fitting arbitrary well identity.
raw_cols = [
    "d_md", "d_z", "d_xy", "dz_dmd", "frac", "slp_all", "slp_50",
    "gr", "gr_m21", "gr_s21", "pf_d", "beam_d", "ncc15_d", "ncc15_s",
    "sp_mean_d", "sp_std", "sp_dmin",
]
raw = df[raw_cols].to_numpy(np.float32)
nw = (direction == "NW").astype(np.float32)
X = np.column_stack(
    [
        base, raw, dir_cos, dir_sin, nw,
        frac * dir_cos, frac * dir_sin,
        base * dir_cos, base * dir_sin,
        df["d_z"].to_numpy(np.float32) * dir_sin,
    ]
).astype(np.float32)
groups = df["supertype"].to_numpy()
folds = list(GroupKFold(5).split(X, y, groups))
residual = y - base

preds: dict[str, np.ndarray] = {"base_lgb7": base.astype(np.float64)}

# Very low-capacity direction-only calibration (intercept + slope with depth).
X_small = np.column_stack([nw, frac, nw * frac, dir_cos, dir_sin]).astype(np.float32)
small = np.zeros(len(y), np.float32)
ridge = np.zeros(len(y), np.float32)
lgb_res = np.zeros(len(y), np.float32)
rng = np.random.default_rng(SEED)

for fold, (tr, va) in enumerate(folds):
    ss = StandardScaler()
    xs = ss.fit_transform(X_small[tr])
    small_model = Ridge(alpha=1000.0)
    small_model.fit(xs, residual[tr])
    small[va] = small_model.predict(ss.transform(X_small[va]))

    ss2 = StandardScaler()
    xr = ss2.fit_transform(X[tr])
    ridge_model = Ridge(alpha=5000.0)
    ridge_model.fit(xr, residual[tr])
    ridge[va] = ridge_model.predict(ss2.transform(X[va]))

    # Cap training rows for speed, sampling uniformly rather than favoring long
    # wells. Validation remains the complete held-out wells.
    tr_wells = np.unique(wells.to_numpy()[tr])
    chosen = []
    per_well = 700
    for w in tr_wells:
        idx = tr[wells.to_numpy()[tr] == w]
        if len(idx) > per_well:
            idx = rng.choice(idx, per_well, replace=False)
        chosen.append(idx)
    fit = np.concatenate(chosen)
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=350,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=300,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=5.0,
        reg_lambda=20.0,
        random_state=SEED + fold,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X[fit], residual[fit])
    lgb_res[va] = model.predict(X[va])
    print(f"fold={fold} fit_rows={len(fit)} valid_rows={len(va)}", flush=True)

for name, corr in {"dir_linear": small, "dir_ridge": ridge, "dir_lgb": lgb_res}.items():
    for alpha in (0.25, 0.5, 0.75, 1.0):
        for clip in (3.0, 6.0, 12.0):
            preds[f"{name}_a{alpha:g}_c{clip:g}"] = (
                base + alpha * np.clip(corr, -clip, clip)
            )

rows = []
for name, pred in preds.items():
    row = {"candidate": name, "pooled_rmse": rmse(y, pred)}
    for d in ("NW", "SE"):
        m = direction == d
        row[f"{d}_rmse"] = rmse(y[m], pred[m])
        row[f"{d}_rows"] = int(m.sum())
        row[f"{d}_wells"] = int(wells[m].nunique())
    rows.append(row)

report = pd.DataFrame(rows).sort_values("pooled_rmse")
report["gain_vs_lgb7"] = rmse(y, base) - report["pooled_rmse"]
report.to_csv(OUT / "candidates.csv", index=False)

best = report.iloc[0].to_dict()
summary = {
    "baseline_pooled_rmse": rmse(y, base),
    "reference_reported_rmse": 10.553,
    "best": best,
    "n_rows": int(len(y)),
    "n_wells": int(wells.nunique()),
    "direction_wells": traj["direction"].value_counts().to_dict(),
    "validation": "5-fold GroupKFold by supertype; complete row-level pooled RMSE",
    "caveat": (
        "Residual calibration is cross-fitted, but consumes cached base OOFs; "
        "a final promotion requires nested regeneration or an independent fold."
    ),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(report.head(15).to_string(index=False), flush=True)
print(json.dumps(summary, indent=2), flush=True)
