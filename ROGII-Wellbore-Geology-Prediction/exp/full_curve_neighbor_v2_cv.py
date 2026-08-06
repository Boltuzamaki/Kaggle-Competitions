"""Honest leave-one-well-out nearby-well structural-wiggle transfer.

Candidate choice uses only geometry, drilling direction, typewell geology
sequence, and the query well's visible TVT_input prefix.  The held-out suffix
TVT is used solely for scoring.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

DATA = Path("data/train")
OUT = Path("exp/results/full_curve_neighbor_v2")
OUT.mkdir(parents=True, exist_ok=True)


def angle_diff(a, b):
    return abs(np.angle(np.exp(1j * (a - b))))


def robust_poly(x, y, degree=2):
    x0 = np.median(x)
    scale = max(np.std(x), 1.0)
    xx = (x - x0) / scale
    keep = np.isfinite(xx) & np.isfinite(y)
    coef = np.polyfit(xx[keep], y[keep], degree)
    for _ in range(3):
        resid = y - np.polyval(coef, xx)
        med = np.nanmedian(resid)
        mad = 1.4826 * np.nanmedian(np.abs(resid - med)) + 1e-6
        keep = np.isfinite(resid) & (np.abs(resid - med) < 3.5 * mad)
        coef = np.polyfit(xx[keep], y[keep], degree)
    return np.polyval(coef, xx)


def load_one(path):
    wid = os.path.basename(path).split("__")[0]
    d = pd.read_csv(path)
    ps = int(d.TVT_input.notna().sum())
    if ps < 60 or ps >= len(d) - 3:
        return None
    xy = d[["X", "Y"]].to_numpy(float)
    # Full trajectory geometry is an input at test time.
    centered = xy - xy.mean(0)
    _, _, vt = np.linalg.svd(centered[:: max(1, len(d)//1000)], full_matrices=False)
    axis = vt[0]
    # Give the PCA axis the actual drilling orientation.
    travel = xy[-1] - xy[0]
    if np.dot(axis, travel) < 0:
        axis = -axis
    az = float(np.arctan2(axis[1], axis[0]))
    s = centered @ axis
    order = np.argsort(s)
    # Deduplicate projected positions for interpolation.
    su, ui = np.unique(s[order], return_index=True)
    u = d.TVT.to_numpy(float) + d.Z.to_numpy(float)
    u_sorted = u[order][ui]
    # Remove broad structure, leaving transferable local curvature/wiggle.
    trend_sorted = robust_poly(su, u_sorted, degree=2)
    wiggle = gaussian_filter1d(u_sorted - trend_sorted, sigma=8)
    geology = tuple(
        pd.read_csv(DATA / f"{wid}__typewell.csv", usecols=["Geology"])
        .Geology.dropna().astype(str).drop_duplicates()
    )
    return dict(
        wid=wid, d=d, ps=ps, xy=xy, axis=axis, az=az, s=s, su=su,
        u_sorted=u_sorted, wiggle=wiggle, tree=cKDTree(xy[::10]),
        center=xy.mean(0), geology=geology,
    )


wells = [x for x in (load_one(p) for p in sorted(
    glob.glob(str(DATA / "*__horizontal_well.csv")))) if x is not None]
print("loaded", len(wells), flush=True)


def source_values(src, qxy, field):
    # Projection is stable for near-parallel horizontal wells and avoids
    # pointwise nearest-neighbor jumps.
    ss = (qxy - src["center"]) @ src["axis"]
    vals = src[field]
    return np.interp(ss, src["su"], vals, left=vals[0], right=vals[-1])


records, predictions = [], []
for qi, q in enumerate(wells):
    qd, ps = q["d"], q["ps"]
    qxy = q["xy"]
    qu = qd.TVT.to_numpy(float) + qd.Z.to_numpy(float)
    # Honest TVT baselines from visible prefix. Raw U is not stationary without
    # a formation surface, so only its detrended wiggle is transferred.
    qtvt = qd.TVT.to_numpy(float)
    anchor_tvt = np.full(len(qd), np.median(qtvt[max(0, ps-80):ps]))
    # Broad query trend: robust linear TVT continuation over last 400 rows.
    k0 = max(0, ps - 400)
    xfit = qd.MD.to_numpy(float)[k0:ps]
    yfit = qtvt[k0:ps]
    coef = np.polyfit(xfit - xfit[-1], yfit, 1)
    base_u = np.polyval(coef, qd.MD.to_numpy(float) - xfit[-1])

    cand = []
    for si, src in enumerate(wells):
        if si == qi or angle_diff(src["az"], q["az"]) > np.deg2rad(40):
            continue
        # Approximate physical separation at query PS (trees use every 10th row).
        dist = float(src["tree"].query(qxy[ps - 1])[0])
        geo_match = src["geology"] == q["geology"]
        cand.append((not geo_match, dist, si))
    cand.sort()
    # Prefer same geology/master-family, but retain nearby alternatives.
    shortlist = cand[:24]
    scored = []
    # Prefix train/audit split prevents selecting on the same points used to align.
    a0 = max(0, ps - 500)
    cut = a0 + max(30, int(.70 * (ps - a0)))
    for geo_bad, dist, si in shortlist:
        src = wells[si]
        wig = source_values(src, qxy[:ps], "wiggle")
        wbias = np.median(qtvt[a0:cut] - base_u[a0:cut] - wig[a0:cut])
        waudit = np.sqrt(np.mean(
            (qtvt[cut:ps] - (base_u[cut:ps] + wig[cut:ps] + wbias)) ** 2))
        # Also assess wiggle on the safer constant-TVT baseline.
        cbias = np.median(qtvt[a0:cut] - anchor_tvt[a0:cut] - wig[a0:cut])
        caudit = np.sqrt(np.mean(
            (qtvt[cut:ps] - (anchor_tvt[cut:ps] + wig[cut:ps] + cbias)) ** 2))
        # Extrapolated prefix slopes are unstable over multi-thousand-row
        # suffixes; rank the transfer on the conservative constant baseline.
        scored.append((caudit, waudit, caudit, dist, si))
    scored.sort()
    if scored:
        _, waudit, caudit, distance, si = scored[0]
        src = wells[si]
        wig_all = source_values(src, qxy, "wiggle")
        chosen_base = anchor_tvt
        wig_all += np.median(
            qtvt[max(0, ps-300):ps] - chosen_base[max(0, ps-300):ps]
            - wig_all[max(0, ps-300):ps])
        chosen_u = chosen_base + wig_all
        neighbor = src["wid"]
    else:
        waudit = caudit = distance = np.inf
        chosen_u, neighbor = base_u, ""

    tail = np.arange(ps, len(qd))
    actual = qd.TVT.to_numpy(float)[tail]
    variants = {
        "constant_tvt": anchor_tvt[tail],
        "linear_tvt": base_u[tail],
        "neighbor_wiggle": chosen_u[tail],
    }
    for alpha in (.25, .5, .75, 1.0):
        variants[f"linear_neighbor_{alpha}"] = (
            (1-alpha)*anchor_tvt[tail] + alpha*chosen_u[tail])
    for name, pred in variants.items():
        predictions.append(pd.DataFrame(dict(
            well=q["wid"], row=tail, variant=name, actual=actual, pred=pred,
            distance=distance)))
    records.append(dict(well=q["wid"], neighbor=neighbor, distance=distance,
                        prefix_audit=min(waudit, caudit), n=len(tail)))
    if (qi + 1) % 100 == 0:
        print("done", qi + 1, flush=True)

pred = pd.concat(predictions, ignore_index=True)
meta = pd.DataFrame(records)
summary = pred.groupby("variant").apply(
    lambda x: np.sqrt(np.mean((x.actual-x.pred)**2)),
    include_groups=False).rename("rmse").sort_values().reset_index()
bins = [-np.inf, 75, 150, 300, 600, 1200, np.inf]
pred["distance_bin"] = pd.cut(pred.distance, bins)
bydist = pred.groupby(["variant", "distance_bin"], observed=True).apply(
    lambda x: pd.Series(dict(n=len(x),
        wells=x.well.nunique(), rmse=np.sqrt(np.mean((x.actual-x.pred)**2)))),
    include_groups=False).reset_index()
pred["distance_bin"] = pred["distance_bin"].astype(str)
bydist["distance_bin"] = bydist["distance_bin"].astype(str)
pred.to_parquet(OUT / "predictions.parquet", index=False)
meta.to_csv(OUT / "well_neighbors.csv", index=False)
summary.to_csv(OUT / "summary.csv", index=False)
bydist.to_csv(OUT / "distance_bins.csv", index=False)
print(summary.to_string(index=False), flush=True)
print(bydist.to_string(index=False), flush=True)
