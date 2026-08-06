"""Legal GR datum scan around honest Stack-V4 grouped OOF predictions.

For each well, calibration hGR ~= alpha * typewellGR(TVT_input) + beta uses
only the organizer-visible prefix.  Hidden GR then scores constant TVT shifts
of the V4 path.  Corrections are posterior means, never hard argmins.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/heel_calibrated_gr_datum"
OUT.mkdir(parents=True, exist_ok=True)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def robust_affine(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) > 1200:
        take = np.linspace(0, len(x) - 1, 1200).astype(int)
        x, y = x[take], y[take]
    X = np.c_[x, np.ones(len(x))]
    w = np.ones(len(x))
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(5):
        r = y - X @ coef
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-3
        w = 1 / np.maximum(1, np.abs(r) / (2.5 * scale))
        coef = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
    return float(coef[0]), float(coef[1]), float(scale)


f = pd.read_pickle(ROOT / "r_v4b/train_feats.pkl")
oofs = joblib.load(ROOT / "r_v4b/stack_v4_oofs.joblib")["oofs"]
y = f.target.to_numpy(float)
base = np.asarray(oofs["lgb7"], float)
shifts = np.arange(-60, 61, 2, dtype=float)
row_shift = {loss: np.zeros(len(f), np.float32) for loss in ("cauchy", "huber", "l1")}
row_conf = np.zeros(len(f), np.float32)
records = []

for wi, (well, ix0) in enumerate(f.groupby("well", sort=False).indices.items()):
    ix = np.asarray(ix0)
    h = pd.read_csv(ROOT / f"data/train/{well}__horizontal_well.csv")
    t = pd.read_csv(ROOT / f"data/train/{well}__typewell.csv").sort_values("TVT")
    tv = t.TVT.to_numpy(float)
    tg_s = pd.to_numeric(t.GR, errors="coerce").interpolate(limit_direction="both")
    tg = tg_s.fillna(tg_s.median() if tg_s.notna().any() else 0.0).to_numpy(float)
    hgr_s = pd.to_numeric(h.GR, errors="coerce").interpolate(limit_direction="both")
    hgr_all = hgr_s.fillna(hgr_s.median() if hgr_s.notna().any() else 0.0).to_numpy(float)
    vis = h.TVT_input.notna().to_numpy()
    vt = h.loc[vis, "TVT_input"].to_numpy(float)
    vg = hgr_all[vis]
    ref = np.interp(vt, tv, tg)
    alpha, beta, scale = robust_affine(ref, vg)

    # Feature rows are exactly the hidden suffix, identified by competition id.
    id_to_row = pd.Series(np.arange(len(h)), index=[f"{well}_{i}" for i in h.index])
    hr = id_to_row.loc[f.id.iloc[ix]].to_numpy(int)
    hg = hgr_all[hr]
    path = f.last_known_tvt.to_numpy(float)[ix] + base[ix]
    # Downsample uniformly: enough for a stable curve cost without large memory.
    take = np.linspace(0, len(ix) - 1, min(900, len(ix))).astype(int)
    q, obs = path[take], hg[take]
    pred_gr = alpha * np.vstack([np.interp(q + s, tv, tg) for s in shifts]) + beta
    z = (obs[None, :] - pred_gr) / max(scale, 5.0)
    costs = {
        "cauchy": np.mean(np.log1p((z / 2.0) ** 2), axis=1),
        "huber": np.mean(np.where(abs(z) < 1.5, .5*z*z, 1.5*abs(z)-1.125), axis=1),
        "l1": np.mean(abs(z), axis=1),
    }
    # Store costs; temperature grids are applied below without re-reading data.
    rec = {"well": well, "alpha": alpha, "beta": beta, "scale": scale}
    for loss, c in costs.items():
        rec[f"{loss}_costs"] = c.tolist()
    records.append(rec)
    if wi % 100 == 0:
        print("processed", wi, flush=True)

# Evaluate posterior temperature and explicit confidence hedges.  Cost is a
# per-row mean, so temperature is stable across well lengths.
rows = []
pred_cache = {}
for loss in ("cauchy", "huber", "l1"):
    C = np.asarray([r[f"{loss}_costs"] for r in records])
    valid_well = np.isfinite(C).all(axis=1)
    C[~valid_well] = 0.0
    for temp in (.003, .006, .012, .025, .05, .10, .20):
        P = np.exp(-(C - C.min(1, keepdims=True)) / temp)
        P /= P.sum(1, keepdims=True)
        means = P @ shifts
        means[~valid_well] = 0.0
        std = np.sqrt(P @ shifts**2 - means**2)
        for hedge in (0.25, 0.5, 0.75, 1.0):
            corr = np.zeros(len(f), np.float32)
            for j, (_, ix0) in enumerate(f.groupby("well", sort=False).indices.items()):
                corr[np.asarray(ix0)] = hedge * means[j]
            assert np.isfinite(corr).all() and np.isfinite(base).all() and np.isfinite(y).all()
            score = rmse(y, base + corr)
            key = (loss, temp, hedge)
            pred_cache[key] = corr
            rows.append({"loss": loss, "temperature": temp, "hedge": hedge,
                         "rmse": score, "mean_post_std": float(std.mean())})

grid = pd.DataFrame(rows).sort_values("rmse")
best = grid.iloc[0].to_dict()
best_key = (best["loss"], best["temperature"], best["hedge"])
corr = pred_cache[best_key]
fixed = [{"weight": float(w), "rmse": rmse(y, base + w*corr)}
         for w in np.linspace(0, 1, 21)]
summary = {
    "rows": len(f), "wells": int(f.well.nunique()),
    "v4_lgb7": rmse(y, base), "best_posterior": best,
    "best_fixed_blend": min(fixed, key=lambda x: x["rmse"]),
    "invalid_wells": int(sum(not np.isfinite(np.asarray(r["cauchy_costs"])).all()
                             for r in records)),
    "protocol": "honest grouped Stack-V4 OOF; affine calibration from visible prefix only; hidden GR is inference-available",
}
grid.to_csv(OUT / "grid.csv", index=False)
pd.DataFrame(fixed).to_csv(OUT / "fixed_blend.csv", index=False)
np.savez_compressed(OUT / "oof.npz", correction=corr, base=base, y=y,
                    groups=f.well.to_numpy())
(OUT / "costs.json").write_text(json.dumps(records))
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
