"""
Honest formation-plane spatial prior (target-free, no leak):
  TVT ≈ -Z + S_f(X,Y) + b_w
where S_f(X,Y) is formation-top surface for formation f, fit as a LOCAL PLANE
from the k nearest OTHER wells (leave-one-well-out -> fold-safe), and b_w is a
per-well offset estimated from the known prefix only.

This is the honest version of `tvt_from_contacts` (which used the well's own
formation column = leak). Here the well's own formation tops are never used for
its own prediction.
"""
import sys, os, glob, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_H))
from wellbore_lib import load_well, list_wells, ps_index

FORMS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def build_plane_index(split_dir, wells):
    """One row per well: median X,Y and median formation tops."""
    rows = []
    for w in wells:
        try:
            df = pd.read_csv(f"{split_dir}/{w}__horizontal_well.csv",
                             usecols=["X", "Y"] + FORMS).dropna()
        except Exception:
            continue
        if len(df) == 0:
            continue
        r = {"wid": w, "x": float(df["X"].median()), "y": float(df["Y"].median())}
        for c in FORMS:
            r[f"{c}_m"] = float(df[c].median())
        rows.append(r)
    idx = pd.DataFrame(rows)
    xy = idx[["x", "y"]].to_numpy()
    scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
    tree = cKDTree(xy / scale)
    return idx, tree, scale


def impute_plane(idx, tree, scale, xy_q, self_wid, k=10):
    """Local weighted-plane fit of each formation top at query (X,Y) points."""
    wmap = {w: i for i, w in enumerate(idx["wid"])}
    xa = idx["x"].to_numpy(); ya = idx["y"].to_numpy()
    fa = idx[[f"{c}_m" for c in FORMS]].to_numpy(np.float64)
    q = xy_q / scale
    nf = min(k + 6, len(idx))
    dist, ii = tree.query(q, k=nf, workers=-1)
    if self_wid in wmap:                      # leave-one-well-out
        dist = np.where(ii == wmap[self_wid], np.inf, dist)
    order = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
    dk = np.take_along_axis(dist, order, 1); ik = np.take_along_axis(ii, order, 1)
    vk = np.isfinite(dk); wt = np.where(vk, 1.0 / (dk + 1e-3), 0.0)
    xn = xa[ik]; yn = ya[ik]; fn = fa[ik]; wx = wt * xn; wy = wt * yn
    n = len(q); A = np.zeros((n, 3, 3))
    A[:, 0, 0] = (wx * xn).sum(1); A[:, 0, 1] = (wx * yn).sum(1); A[:, 0, 2] = wx.sum(1)
    A[:, 1, 0] = A[:, 0, 1]; A[:, 1, 1] = (wy * yn).sum(1); A[:, 1, 2] = wy.sum(1)
    A[:, 2, 0] = A[:, 0, 2]; A[:, 2, 1] = A[:, 1, 2]; A[:, 2, 2] = wt.sum(1)
    for d in range(3):
        A[:, d, d] += 1e-9
    rhs = np.stack([(wx[:, :, None] * fn).sum(1), (wy[:, :, None] * fn).sum(1),
                    (wt[:, :, None] * fn).sum(1)], 1)
    try:
        coef = np.linalg.solve(A, rhs)
    except Exception:
        coef = np.stack([np.linalg.pinv(A[r]) @ rhs[r] for r in range(n)])
    Xq = xy_q[:, 0]; Yq = xy_q[:, 1]
    pred = Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]
    return pred.astype(np.float64)          # (n, 6) formation tops


def eval_well(w, idx, tree, scale):
    h, _ = load_well("data/train", w); ps = ps_index(h)
    if ps >= len(h) - 5 or ps < 10:
        return None
    kn = h.iloc[:ps]; ev = h.iloc[ps:]
    y = ev["TVT"].to_numpy()
    tps = float(h["TVT"].iloc[ps - 1])
    xy_ev = ev[["X", "Y"]].to_numpy(float); xy_kn = kn[["X", "Y"]].to_numpy(float)
    z_ev = ev["Z"].to_numpy(float); z_kn = kn["Z"].to_numpy(float)
    ktvt = kn["TVT_input"].to_numpy(float)
    form_ev = impute_plane(idx, tree, scale, xy_ev, w)   # (nev,6)
    form_kn = impute_plane(idx, tree, scale, xy_kn, w)   # (nkn,6)
    out = {"y": y, "const": np.full(len(y), tps)}
    preds = []
    for fi in range(len(FORMS)):
        b_w = float(np.median(ktvt + z_kn - form_kn[:, fi]))     # prefix offset
        pred = -z_ev + form_ev[:, fi] + b_w
        out[f"form_{FORMS[fi]}"] = pred
        preds.append(pred)
    out["form_mean"] = np.mean(preds, 0)
    return out


def pooled(res, k): return float(np.sqrt(np.mean(np.concatenate([(r["y"] - r[k]) ** 2 for r in res]))))
def perwell(res, k): return float(np.mean([np.sqrt(np.mean((r["y"] - r[k]) ** 2)) for r in res]))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    allw = list_wells("data/train")
    idx, tree, scale = build_plane_index("data/train", allw)
    print("plane index wells:", len(idx))
    rng = np.random.RandomState(7)
    samp = [allw[i] for i in rng.permutation(len(allw))[:n]]
    t0 = time.time()
    res = [r for r in Parallel(n_jobs=10)(
        delayed(eval_well)(w, idx, tree, scale) for w in samp) if r]
    print("wells=%d time=%.0fs" % (len(res), time.time() - t0))
    keys = ["const"] + [f"form_{f}" for f in FORMS] + ["form_mean"]
    print("%-14s %8s %8s" % ("prior", "pooled", "perWell"))
    for k in keys:
        print("%-14s %8.3f %8.3f" % (k, pooled(res, k), perwell(res, k)))
