"""Honest grouped CV for a low-dimensional complete-well structural-curve model.

One row is one well.  The model sees the same information available at test
time: complete trajectory/GR/formation-marker curves and the visible TVT
prefix.  It predicts PCA coefficients of U = TVT + Z over the hidden suffix.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "data/train").glob("*__horizontal_well.csv"))
SEED = 20260730
N_IN = 48
N_OUT = 96


def interp(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    x = np.linspace(0.0, 1.0, len(v))
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.full_like(q, np.nanmedian(v) if ok.any() else 0.0)
    return np.interp(q, x[ok], v[ok])


def encode(path: Path):
    d = pd.read_csv(path)
    nvis = int(d["TVT_input"].notna().sum())
    if nvis < 20 or nvis >= len(d) - 3:
        return None
    cut = nvis - 1
    # Normalize coordinates at the last visible station.
    md = d.MD.to_numpy(float)
    z = d.Z.to_numpy(float)
    x = d.X.to_numpy(float)
    y = d.Y.to_numpy(float)
    tvt = d.TVT.to_numpy(float)
    u = tvt + z
    qfull = np.linspace(0, 1, N_IN)
    qpre = np.linspace(0, 1, 20)
    qtail = np.linspace(0, 1, N_OUT)

    def whole(v):
        return interp(np.asarray(v, float), qfull)

    # Full-well observed covariates. Scaling here is physical, while a fold
    # StandardScaler below handles remaining differences.
    chans = [
        (md - md[cut]) / 1000,
        (z - z[cut]) / 50,
        (x - x[cut]) / 3000,
        (y - y[cut]) / 3000,
        np.gradient(z) * 20,
    ]
    for c in ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]:
        a = d[c].to_numpy(float)
        chans.append((a - a[cut]) / 50)
    gr = d.GR.interpolate(limit_direction="both").fillna(d.GR.median()).to_numpy(float)
    gr = (gr - np.nanmedian(gr)) / (np.nanstd(gr) + 1e-6)
    chans += [gr, gaussian_filter1d(gr, 10), gaussian_filter1d(gr, 40)]

    f = np.concatenate([whole(a) for a in chans])
    # Prefix U shape is the most direct observed structural state.
    up = u[:nvis] - u[cut]
    f = np.r_[f, interp(up, qpre)]
    # Compact scalar descriptors and direction.
    f = np.r_[
        f,
        len(d) / 6000,
        nvis / len(d),
        (md[-1] - md[cut]) / 4000,
        np.sin(np.arctan2(y[-1] - y[cut], x[-1] - x[cut])),
        np.cos(np.arctan2(y[-1] - y[cut], x[-1] - x[cut])),
        np.polyfit(np.arange(min(300, nvis)), up[-min(300, nvis) :], 1)[0],
    ]

    # Target is U relative to its exactly known boundary value.
    tail_u = u[cut:] - u[cut]
    target = interp(tail_u, qtail)
    return dict(well=path.name.split("__")[0], f=f, target=target, d=d, cut=cut, u0=u[cut])


items = [x for x in (encode(p) for p in FILES) if x is not None]
X = np.stack([x["f"] for x in items])
Y = np.stack([x["target"] for x in items])
print("wells/features/target", X.shape, Y.shape, flush=True)

# The split unit is already a whole well. Fixed seed matches the random
# GroupKFold regime discussed by leading competitors.
kf = KFold(5, shuffle=True, random_state=SEED)
ridge_specs = [(n, a) for n in (8, 16, 24, 32) for a in (10.0, 30.0, 60.0, 120.0, 240.0)]
preds = {f"ridge_n{n}_a{a:g}": np.zeros_like(Y) for n, a in ridge_specs}
preds.update({k: np.zeros_like(Y) for k in ["ridge", "extra", "forest", "blend"]})
# Channel ablations diagnose whether the gain is a legitimate supplied input.
tail_idx = np.arange(14 * N_IN, X.shape[1])
geom_idx = np.r_[np.arange(0, 5 * N_IN), tail_idx]
form_idx = np.r_[np.arange(0, 11 * N_IN), tail_idx]  # geometry + six markers
gr_idx = np.r_[np.arange(0, 5 * N_IN), np.arange(11 * N_IN, 14 * N_IN), tail_idx]
prefix_idx = tail_idx
ablations = {"geom_only": geom_idx, "geom_form": form_idx, "geom_gr": gr_idx, "prefix_only": prefix_idx}
preds.update({k: np.zeros_like(Y) for k in ablations})
fold_rows = []
for fold, (tr, va) in enumerate(kf.split(X)):
    sx = StandardScaler().fit(X[tr])
    xt, xv = sx.transform(X[tr]), sx.transform(X[va])
    npc = min(32, len(tr) - 1)
    pca = PCA(n_components=npc, whiten=False, random_state=SEED).fit(Y[tr])
    scores = pca.transform(Y[tr])

    ridge_predictions = {}
    for nc, alpha in ridge_specs:
        model = Ridge(alpha=alpha).fit(xt, scores[:, :nc])
        coef = model.predict(xv)
        curve = coef @ pca.components_[:nc] + pca.mean_
        preds[f"ridge_n{nc}_a{alpha:g}"][va] = curve
        ridge_predictions[(nc, alpha)] = curve
    for name, cols in ablations.items():
        sa = StandardScaler().fit(X[tr][:, cols])
        ma = Ridge(alpha=10.0).fit(sa.transform(X[tr][:, cols]), scores[:, :8])
        coef = ma.predict(sa.transform(X[va][:, cols]))
        preds[name][va] = coef @ pca.components_[:8] + pca.mean_
    ridge = Ridge(alpha=120.0).fit(xt, scores[:, :16])
    extra = ExtraTreesRegressor(
        n_estimators=600, min_samples_leaf=3, max_features=0.65,
        n_jobs=-1, random_state=SEED + fold,
    ).fit(xt, scores)
    forest = RandomForestRegressor(
        n_estimators=350, min_samples_leaf=4, max_features=0.7,
        n_jobs=-1, random_state=SEED + 50 + fold,
    ).fit(xt, scores)
    pr = ridge_predictions[(16, 120.0)]
    pe = pca.inverse_transform(extra.predict(xv))
    pf = pca.inverse_transform(forest.predict(xv))
    preds["ridge"][va] = pr
    preds["extra"][va] = pe
    preds["forest"][va] = pf
    # Conservative ensemble chosen without examining this fold's labels.
    preds["blend"][va] = 0.15 * pr + 0.55 * pe + 0.30 * pf
    print("fold", fold, "explained", float(pca.explained_variance_ratio_.sum()), flush=True)


def score(P):
    sse = rows = 0
    per = []
    for i, it in enumerate(items):
        d, cut = it["d"], it["cut"]
        actual = d.TVT.to_numpy(float)[cut + 1 :]
        z = d.Z.to_numpy(float)[cut + 1 :]
        q = np.linspace(0, 1, len(actual) + 1)[1:]
        urel = np.interp(q, np.linspace(0, 1, N_OUT), P[i])
        pred = it["u0"] + urel - z
        err = actual - pred
        sse += float(err @ err)
        rows += len(err)
        per.append(float(np.sqrt(np.mean(err**2))))
    return float(np.sqrt(sse / rows)), rows, per


scores = {}
for k, p in preds.items():
    rmse, rows, per = score(p)
    scores[k] = rmse
    print(k, rmse, "median_well", float(np.median(per)), flush=True)

# Honest last-visible-value constant baseline.
const_sse = 0.0
rows = 0
for it in items:
    d, cut = it["d"], it["cut"]
    a = d.TVT.to_numpy(float)[cut + 1 :]
    anchor = float(d.TVT_input.iloc[cut])
    const_sse += float(np.sum((a - anchor) ** 2))
    rows += len(a)
scores["constant"] = float(np.sqrt(const_sse / rows))

out = {
    "n_wells": len(items),
    "hidden_rows": rows,
    "scores": scores,
    "reference_stack_v4b": 10.553,
    "protocol": "5-fold shuffled whole-well CV; organizer TVT_input prefixes",
}
print("SUMMARY", json.dumps(out, indent=2), flush=True)
(ROOT / "exp/results").mkdir(exist_ok=True)
(ROOT / "exp/results/complete_well_pca_cv.json").write_text(json.dumps(out, indent=2))
