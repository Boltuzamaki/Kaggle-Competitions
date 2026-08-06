"""Prefix-calibrated, sequence-aware typewell GR matcher.

Only inputs present for a query well are used: trajectory, horizontal GR,
vertical typewell GR, and the visible TVT_input prefix.  The experiment measures
incremental value relative to the existing honest GroupKFold LightGBM OOF.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from numba import njit
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/contextual_reference_matcher_v1"
OUT.mkdir(parents=True, exist_ok=True)
OFF = np.arange(-30.0, 30.01, 0.5)


def robust_affine(x, y):
    """Robustly map typewell GR amplitude into horizontal-log amplitude."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 30:
        return 1.0, np.nanmedian(y) - np.nanmedian(x)
    keep = np.ones(len(x), bool)
    for _ in range(4):
        A = np.c_[x[keep], np.ones(keep.sum())]
        a, b = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        r = y - (a*x+b)
        s = 1.4826*np.nanmedian(np.abs(r-np.nanmedian(r))) + 1e-6
        keep = np.abs(r-np.nanmedian(r)) < 2.8*s
    return float(np.clip(a, .25, 2.5)), float(b)


@njit(cache=True)
def viterbi(cost, jump_pen, center_pen):
    n, k = cost.shape
    prev = cost[0].copy()
    bp = np.empty((n, k), np.int16)
    mid = (k-1)/2
    for i in range(1, n):
        cur = np.empty(k, np.float32)
        for j in range(k):
            best = 1e30
            arg = j
            lo, hi = max(0, j-4), min(k, j+5)
            for q in range(lo, hi):
                z = prev[q] + jump_pen*(j-q)*(j-q)
                if z < best:
                    best, arg = z, q
            cur[j] = best + cost[i, j] + center_pen*((j-mid)/mid)**2
            bp[i, j] = arg
        prev = cur
    out = np.empty(n, np.int16)
    q = np.argmin(prev)
    for i in range(n-1, -1, -1):
        out[i] = q
        if i:
            q = bp[i, q]
    return out


def one_well(w, g, base):
    hp = ROOT/"data/train"/f"{w}__horizontal_well.csv"
    tp = ROOT/"data/train"/f"{w}__typewell.csv"
    h = pd.read_csv(hp)
    tw = pd.read_csv(tp).dropna(subset=["TVT", "GR"]).sort_values("TVT")
    tw = tw.drop_duplicates("TVT")
    p = h.TVT_input.notna().sum()
    # Feature cache consists precisely of hidden suffix rows.
    if len(h)-p != len(g):
        return None
    ht = h.TVT_input.iloc[:p].to_numpy(float)
    hg = h.GR.interpolate(limit_direction="both").to_numpy(float)
    tt, tg0 = tw.TVT.to_numpy(float), tw.GR.to_numpy(float)
    a, b = robust_affine(np.interp(ht, tt, tg0), hg[:p])
    tg = a*tg0+b
    # Prefix residual measures calibration reliability.
    cal = np.interp(ht, tt, tg)
    cal_rmse = float(np.sqrt(np.mean((hg[:p]-cal)**2)))
    abs_base = float(g.last_known_tvt.iloc[0]) + base
    cand = abs_base[:, None] + OFF[None, :]

    # Multi-scale reference representations. Broad scales stabilize aliases;
    # fine scale is deliberately downweighted.
    cost = np.zeros(cand.shape, np.float32)
    scale = max(8., 1.4826*np.median(np.abs(hg[:p]-np.median(hg[:p]))))
    ev = hg[p:]
    for sig, weight in ((1.2, .10), (4., .25), (10., .35), (22., .30)):
        eh = gaussian_filter1d(ev, sig, mode="nearest")
        rt = gaussian_filter1d(tg, sig, mode="nearest")
        rr = np.interp(cand, tt, rt)
        cost += weight*np.minimum(np.abs(eh[:, None]-rr)/scale, 4.0)
    # Texture/facies representation: local variance is less sensitive to the
    # absolute GR calibration and helps distinguish repeated broad motifs.
    eh2 = gaussian_filter1d(ev*ev, 10, mode="nearest")
    ehm = gaussian_filter1d(ev, 10, mode="nearest")
    estd = np.sqrt(np.maximum(0, eh2-ehm*ehm))
    rt2 = gaussian_filter1d(tg*tg, 10, mode="nearest")
    rtm = gaussian_filter1d(tg, 10, mode="nearest")
    rstd = np.sqrt(np.maximum(0, rt2-rtm*rtm))
    cost += .12*np.minimum(np.abs(estd[:, None]-np.interp(cand, tt, rstd))/10., 3.)
    # Smooth evidence along the well before global path decoding.
    cost = gaussian_filter1d(cost, 4, axis=0, mode="nearest")
    idx = viterbi(cost, jump_pen=.10, center_pen=.010)
    path = base + OFF[idx]

    # Independent soft posterior provides a conservative continuous estimate.
    logits = -(cost-cost.min(1, keepdims=True))/.20
    prob = np.exp(np.clip(logits, -30, 0))
    prob /= prob.sum(1, keepdims=True)
    soft = base + prob@OFF
    return {
        "well": w, "y": g.target.to_numpy(float), "base": base,
        "path": path, "soft": soft, "cal_rmse": cal_rmse,
        "entropy": float(np.mean(-np.sum(prob*np.log(prob+1e-9), axis=1))),
    }


def score(rows, fn):
    y = np.concatenate([r["y"] for r in rows])
    p = np.concatenate([fn(r) for r in rows])
    return float(np.sqrt(np.mean((y-p)**2)))


def main(limit=120):
    d = pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
    oo = joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb123"]
    rng = np.random.RandomState(731)
    wells = np.array(sorted(d.well.unique()))
    if limit and limit < len(wells):
        wells = np.sort(rng.choice(wells, limit, False))
    rows = []
    for j, w in enumerate(wells):
        g = d[d.well == w]
        r = one_well(w, g, np.asarray(oo)[g.index])
        if r is not None:
            rows.append(r)
        if (j+1) % 20 == 0:
            print("processed", j+1, flush=True)
    grid = []
    base_score = score(rows, lambda r:r["base"])
    for kind in ("path", "soft"):
        for a in (.05, .10, .15, .20, .30, .40, .50, .75, 1.):
            s = score(rows, lambda r,k=kind,a=a:(1-a)*r["base"]+a*r[k])
            grid.append({"kind":kind, "blend":a, "rmse":s, "gain":base_score-s})
    grid = pd.DataFrame(grid).sort_values("rmse")
    summary = {
        "wells":len(rows), "base_rmse":base_score,
        "raw_path":score(rows, lambda r:r["path"]),
        "raw_soft":score(rows, lambda r:r["soft"]),
        "best":grid.iloc[0].to_dict(),
        "median_calibration_rmse":float(np.median([r["cal_rmse"] for r in rows])),
    }
    print(json.dumps(summary, indent=2), flush=True)
    suffix = f"{len(rows)}w"
    grid.to_csv(OUT/f"grid_{suffix}.csv", index=False)
    (OUT/f"summary_{suffix}.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([{"well":r["well"], "base_rmse":np.sqrt(np.mean((r["y"]-r["base"])**2)),
                   "path_rmse":np.sqrt(np.mean((r["y"]-r["path"])**2)),
                   "soft_rmse":np.sqrt(np.mean((r["y"]-r["soft"])**2)),
                   "cal_rmse":r["cal_rmse"], "entropy":r["entropy"]}
                  for r in rows]).to_csv(OUT/f"wells_{suffix}.csv", index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 120)
