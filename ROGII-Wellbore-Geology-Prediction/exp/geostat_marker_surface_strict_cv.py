"""Strict outer-well geostatistical EGFDU surface reconstruction."""
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"exp/results/geostat_marker_surface"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT/"query_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def load_wells(limit=200):
    files = sorted((ROOT/"data/train").glob("*__horizontal_well.csv"))
    rng = np.random.RandomState(805)
    if limit and limit < len(files):
        files = sorted(rng.choice(files, limit, False))
    out = []
    for p in files:
        d = pd.read_csv(p)
        known = np.flatnonzero(d.TVT_input.notna().to_numpy())
        if not len(known) or known[-1] >= len(d)-5:
            continue
        out.append({"well": p.name.split("__")[0], "d": d,
                    "ps": int(known[-1]),
                    "cx": float(d.X.mean()), "cy": float(d.Y.mean())})
    return out


def cloud(items, n=24):
    xy, z = [], []
    for s in items:
        d = s["d"]
        ix = np.linspace(0, len(d)-1, min(n, len(d))).astype(int)
        xy.append(d[["X", "Y"]].to_numpy(float)[ix])
        z.append(d.EGFDU.to_numpy(float)[ix])
    return np.vstack(xy), np.concatenate(z)


def design(x, degree):
    a, b = x[:, 0], x[:, 1]
    if degree == 1:
        return np.c_[np.ones(len(x)), a, b]
    return np.c_[np.ones(len(x)), a, b, a*a, a*b, b*b]


def predict_methods(train_items, query, ratio):
    xy, val = cloud(train_items)
    d = query["d"]; qxy = d[["X", "Y"]].to_numpy(float)
    center = np.array([query["cx"], query["cy"]])
    # Rotate into the query-well along/across coordinates.
    direction = qxy[-1]-qxy[0]
    direction /= np.linalg.norm(direction)+1e-9
    cross = np.array([-direction[1], direction[0]])
    R = np.stack([direction, cross], axis=1)
    xs = (xy-center)@R
    qs = (qxy-center)@R
    xs[:, 1] *= ratio; qs[:, 1] *= ratio
    dist = np.sqrt(np.sum(xs*xs, axis=1))
    take = np.argsort(dist)[:min(600, len(dist))]
    xn, yn, dn = xs[take], val[take], dist[take]
    scale = max(np.quantile(dn, .45), 100.)
    out = {}
    # Row-local anisotropic IDW and Gaussian kernel.
    D = np.sqrt(((qs[:, None]-xn[None])**2).sum(2))
    for k in (40, 120):
        ii = np.argpartition(D, min(k, D.shape[1])-1, axis=1)[:, :k]
        dk = np.take_along_axis(D, ii, 1)
        vk = yn[ii]
        wt = 1/(dk+15.)**2
        out[f"idw{k}_a{ratio}"] = (wt*vk).sum(1)/wt.sum(1)
    wt = np.exp(-.5*(D/scale)**2)
    out[f"gauss_a{ratio}"] = (wt*yn[None]).sum(1)/(wt.sum(1)+1e-9)
    # Local weighted polynomial/LOESS around the query well.
    w = np.exp(-.5*(dn/scale)**2)
    for degree in (1, 2):
        A, Q = design(xn/1000, degree), design(qs/1000, degree)
        reg = 1e-3 if degree == 1 else 2e-2
        lhs = A.T@(w[:, None]*A)+reg*np.eye(A.shape[1])
        coef = np.linalg.solve(lhs, A.T@(w*yn))
        out[f"loess{degree}_a{ratio}"] = Q@coef
    # Curved thin-plate surface; smoothing prevents point/well memorization.
    try:
        nr = min(180, len(xn))
        rbf = RBFInterpolator(xn[:nr], yn[:nr], kernel="thin_plate_spline",
                              smoothing=100., degree=1)
        out[f"thinplate_a{ratio}"] = rbf(qs)
    except Exception:
        pass
    return out, float(np.min(np.sqrt(
        (np.array([[x["cx"], x["cy"]] for x in train_items])-center)**2
    ).sum(1)))


def calibrate_score(s, surf):
    d, ps = s["d"], s["ps"]
    z = d.Z.to_numpy(float)
    b = np.median(d.TVT_input.to_numpy(float)[:ps+1]+z[:ps+1]-surf[:ps+1])
    pred = surf-z+b
    m = np.arange(len(d)) > ps
    return d.TVT.to_numpy(float)[m], pred[m]


def main(limit=200):
    cached = pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
    oof = joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
    eligible = set(cached.well.astype(str).unique())
    all_items = [s for s in load_wells(0) if s["well"] in eligible]
    rng = np.random.RandomState(805)
    items = all_items if not limit or limit >= len(all_items) else [
        all_items[i] for i in sorted(rng.choice(len(all_items), limit, False))]
    pred = {}
    distances = {}
    # Leave-query-well-out spatial CV. Every one of the other 772 wells is a
    # legal training control; the query's marker never enters its surface.
    for i, s in enumerate(items):
        w = s["well"]
        cp = CACHE/f"{w}.npz"
        if cp.exists():
            zz = np.load(cp)
            pred[w] = {k: zz[k] for k in zz.files if k != "_distance"}
            distances[w] = float(zz["_distance"])
        else:
            train = [q for q in all_items if q["well"] != w]
            pred[w] = {}
            for ratio in (1., 2., 4.):
                q, dist = predict_methods(train, s, ratio)
                pred[w].update(q); distances[w] = dist
            np.savez_compressed(cp, **pred[w],
                                _distance=np.array(distances[w]))
        if (i+1) % 25 == 0:
            print("queries", i+1, flush=True)
    methods = sorted(next(iter(pred.values())))
    rows = []
    # Strong deployable OOF anchor on exactly the same wells/rows.
    truth, base = [], []
    cgroups = {w: g for w, g in cached.groupby("well", sort=False)}
    for s in items:
        g = cgroups[s["well"]]
        ix = g.index.to_numpy()
        b = (.55*np.asarray(oof["lgb123"])[ix]+.20*np.asarray(oof["lgb7"])[ix]+
             .15*np.asarray(oof["xgb"])[ix]+.10*np.asarray(oof["cat"])[ix])
        # Cached target/base are deltas from the last visible TVT.
        last = float(g.last_known_tvt.iloc[0])
        truth.append(g.target.to_numpy(float)); base.append(b)
    yall, ball = np.concatenate(truth), np.concatenate(base)
    lasts = np.concatenate([
        np.repeat(float(cgroups[s["well"]].last_known_tvt.iloc[0]),
                  len(s["d"])-s["ps"]-1) for s in items])
    grid = []
    method_preds = {}
    for method in methods:
        yy, pp = [], []
        for s in items:
            y, p = calibrate_score(s, pred[s["well"]][method])
            yy.append(y); pp.append(p)
        y_abs, p_abs = np.concatenate(yy), np.concatenate(pp)
        y, p = y_abs-lasts, p_abs-lasts
        method_preds[method] = p
        if not np.allclose(y, yall, atol=.05):
            raise RuntimeError("raw/cached target alignment mismatch")
        for blend in (0, .05, .1, .15, .2, .3, .4, .55, .7, 1):
            grid.append({"method": method, "blend": blend,
                         "rmse": rmse(y, (1-blend)*ball+blend*p)})
        rows.append({"method": method, "direct_rmse": rmse(y, p)})
    grid = pd.DataFrame(grid).sort_values("rmse")
    direct = pd.DataFrame(rows).sort_values("direct_rmse")
    best_method = str(grid.iloc[0].method)
    # Neighbor-distance audit for best direct spatial method.
    well_rows = []
    cursor = 0
    for s in items:
        n = len(s["d"])-s["ps"]-1
        yy = yall[cursor:cursor+n]
        pp = method_preds[best_method][cursor:cursor+n]
        well_rows.append({"well": s["well"], "rows": n,
                          "neighbor_distance": distances[s["well"]],
                          "rmse": rmse(yy, pp)})
        cursor += n
    wr = pd.DataFrame(well_rows)
    wr["distance_bin"] = pd.cut(wr.neighbor_distance,
                                [-np.inf, 150, 300, 600, np.inf])
    bins = []
    for name, q in wr.groupby("distance_bin", observed=True):
        bins.append({"bin": str(name), "wells": len(q),
                     "weighted_rmse": float(np.sqrt(np.average(
                         q.rmse**2, weights=q.rows)))})
    result = {"wells": len(items), "rows": len(yall),
              "anchor": rmse(yall, ball),
              "best_direct": direct.iloc[0].to_dict(),
              "best_blend": grid.iloc[0].to_dict(),
              "distance_bins": bins}
    print(json.dumps(result, indent=2), flush=True)
    direct.to_csv(OUT/"direct_methods.csv", index=False)
    grid.to_csv(OUT/"blend_grid.csv", index=False)
    wr.to_csv(OUT/"well_metrics.csv", index=False)
    (OUT/"summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 200)
