"""Locked TabICL gate confirmation on an independent shuffled whole-well split."""
from pathlib import Path
import contextlib, io, json, runpy

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tabicl import TabICLRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/tabicl_complete_well_gate"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = ROOT / "exp/public_artifacts/tabicl/tabicl-regressor-v2-20260212.ckpt"

c = np.load(ROOT / "exp/results/complete_well_moe/well_table.npz", allow_pickle=True)
b = np.load(ROOT / "exp/results/prefix_backtest_moe/backtest_features.npz", allow_pickle=True)
X0, L, wn, nrow = c["X"], c["L"], c["wn"], c["nrow"]
if not np.array_equal(wn.astype(str), b["wells"].astype(str)):
    raise RuntimeError("BT well identity")
X = np.c_[X0, b["features"]].astype(np.float32)

with contextlib.redirect_stdout(io.StringIO()):
    s = runpy.run_path(str(ROOT / "exp/meta_all_honest_oof.py"))
common, y, wells, legs = s["common"], s["y"], s["wells"], s["legs"]
if not np.array_equal(
    wn.astype(str), pd.Series(wells).drop_duplicates().astype(str).to_numpy()
):
    raise RuntimeError("meta well identity")
fm = np.load(ROOT / "exp/results/heel_calibrated_gr_datum/full_meta/oof.npz")
vf = pd.read_pickle(ROOT / "r_v4b/train_feats.pkl").iloc[common].reset_index(drop=True)
hq = np.load(ROOT / "exp/results/heel_calibrated_gr_datum/oof.npz")
heel = legs["v4_lgb7"] + hq["correction"][common]


def proj(p, deg):
    out = p.copy()
    for _, ii in pd.Series(np.arange(len(y))).groupby(wells, sort=False):
        ix = ii.to_numpy()
        x = vf.d_md.to_numpy(float)[ix]
        x = 2 * (x - x.min()) / max(np.ptp(x), 1e-6) - 1
        z = p[ix] + vf.d_z.to_numpy(float)[ix]
        out[ix] = np.polyval(np.polyfit(x, z, deg), x) - vf.d_z.to_numpy(float)[ix]
    return out


P = np.column_stack(
    [
        fm["add"],
        fm["base"],
        heel,
        legs["v4_lgb7"],
        legs["har_physics"],
        legs["har_lgb"],
        legs["har_xgb"],
        legs["pil_blend_oof_postprocessed"],
        proj(legs["v4_lgb7"].copy(), 2),
        proj(legs["v4_lgb7"].copy(), 3),
    ]
)
wix = [
    ii.to_numpy()
    for _, ii in pd.Series(np.arange(len(y))).groupby(wells, sort=False)
]
G = np.empty((len(wn), 10, 10))
for i, ix in enumerate(wix):
    e = P[ix] - y[ix, None]
    G[i] = e.T @ e / len(ix)


def score(W, ids):
    z = np.einsum("ni,nij,nj->n", W, G[ids], W)
    return float(np.sqrt(np.sum(nrow[ids] * z) / np.sum(nrow[ids])))


# Predeclared before inspecting this split: softmax temperature 1.0, 50% blend.
temperature, blend = 1.0, 0.5
splits = list(KFold(5, shuffle=True, random_state=20260731).split(X))
pred = np.zeros_like(L)
fid = np.full(len(wn), -1)
fold_rows = []
for fold, (tr, va) in enumerate(splits):
    for j in range(1, 10):
        m = TabICLRegressor(
            n_estimators=4,
            batch_size=4,
            kv_cache=False,
            model_path=MODEL,
            allow_auto_download=False,
            device="cuda",
            use_amp=True,
            random_state=6200 + fold * 20 + j,
            verbose=False,
        )
        m.fit(X[tr], L[tr, j] - L[tr, 0])
        pred[va, j] = m.predict(X[va])
    fid[va] = fold
    q = pred[va]
    W = np.exp(np.clip(-(q - q.min(1, keepdims=True)) / temperature, -20, 0))
    W /= W.sum(1, keepdims=True)
    baseW = np.zeros_like(W)
    baseW[:, 0] = 1
    Q = blend * W + (1 - blend) * baseW
    row = {
        "fold": fold,
        "wells": len(va),
        "base": score(baseW, va),
        "locked": score(Q, va),
    }
    row["gain"] = row["base"] - row["locked"]
    fold_rows.append(row)
    print(row, flush=True)

ids = np.arange(len(wn))
q = pred
W = np.exp(np.clip(-(q - q.min(1, keepdims=True)) / temperature, -20, 0))
W /= W.sum(1, keepdims=True)
baseW = np.zeros_like(W)
baseW[:, 0] = 1
Q = blend * W + (1 - blend) * baseW
summary = {
    "split": "KFold(5, shuffle=True, random_state=20260731) over unique wells",
    "locked_temperature": temperature,
    "locked_blend": blend,
    "accepted_base": score(baseW, ids),
    "locked_rmse": score(Q, ids),
    "gain": score(baseW, ids) - score(Q, ids),
    "fold_wins": sum(r["gain"] > 0 for r in fold_rows),
    "folds": fold_rows,
    "features": X.shape[1],
    "checkpoint": str(MODEL),
    "protocol": "configuration predeclared; no tuning/grid inspection on confirmation split",
}
(OUT / "locked_resplit_summary.json").write_text(json.dumps(summary, indent=2))
np.savez_compressed(
    OUT / "locked_resplit_oof.npz", predicted_regret=pred, fold=fid
)
print(json.dumps(summary, indent=2))
