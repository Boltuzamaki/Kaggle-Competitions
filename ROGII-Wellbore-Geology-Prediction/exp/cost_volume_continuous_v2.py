"""Full-context GR/typewell cost-volume -> continuous OOF correction.

For every evaluation station this exposes matching costs over a TVT-offset
grid plus centered, forward and backward path-integrated versions.  All inputs
exist at inference; validation is grouped by complete well.
"""
from pathlib import Path
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"exp/results/cost_volume_continuous_v2"
OUT.mkdir(parents=True, exist_ok=True)
OFF = np.array([-40, -20, -10, -5, 0, 5, 10, 20, 40], np.float32)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def directional_means(a):
    """Mean cost from heel->row and row->toe, per offset."""
    n = len(a)
    den_f = np.arange(1, n+1, dtype=np.float32)[:, None]
    den_b = np.arange(n, 0, -1, dtype=np.float32)[:, None]
    return np.cumsum(a, axis=0)/den_f, np.cumsum(a[::-1], axis=0)[::-1]/den_b


def posterior(cost):
    scale = np.median(np.std(cost, axis=1))+0.03
    q = -(cost-cost.min(1, keepdims=True))/scale
    w = np.exp(np.clip(q, -35, 0))
    w /= w.sum(1, keepdims=True)+1e-9
    mean = w@OFF
    sd = np.sqrt(np.sum(w*(OFF[None]-mean[:, None])**2, axis=1))
    ent = -np.sum(w*np.log(w+1e-9), axis=1)
    s = np.sort(cost, axis=1)
    return mean, sd, ent, s[:, 1]-s[:, 0]


def well_features(g, base):
    z = pd.DataFrame(index=g.index)
    volumes = {}
    for prefix in ("tda", "tdpf"):
        r = g[[f"{prefix}{int(o)}" for o in OFF]].to_numpy(np.float32)
        scale = np.median(np.abs(r), axis=1, keepdims=True)+5
        cost = np.log1p((r/scale)**2).astype(np.float32)
        volumes[f"{prefix}_raw"] = cost
        for h in (15, 60, 200):
            size = min(len(g), 2*h+1)
            volumes[f"{prefix}_c{h}"] = uniform_filter1d(
                cost, size=size, axis=0, mode="nearest")
        fw, bw = directional_means(cost)
        volumes[f"{prefix}_fw"] = fw
        volumes[f"{prefix}_bw"] = bw
        volumes[f"{prefix}_whole"] = np.broadcast_to(cost.mean(0), cost.shape)

    # Every cost bin is retained: the trees learn nonlinear alias decisions.
    for name, cost in volumes.items():
        for j, o in enumerate(OFF):
            z[f"{name}_{int(o)}"] = cost[:, j]
        m, sd, ent, margin = posterior(cost)
        z[f"{name}_mean"] = m
        z[f"{name}_sd"] = sd
        z[f"{name}_entropy"] = ent
        z[f"{name}_margin"] = margin
        z[f"{name}_argmin"] = OFF[np.argmin(cost, axis=1)]

    z["base"] = base
    pf = g.pf_d.to_numpy(np.float32)
    for k in ("pf_d", "pf3_d", "beam_d", "pf_vs_beam", "sp_mean_d",
              "sp_std", "sp_dmin", "ncc8_d", "ncc8_s", "ncc15_d",
              "ncc15_s", "ncc25_d", "ncc25_s", "d_md", "d_z", "d_xy",
              "dz_dmd", "gr", "gr_m21", "gr_s21", "frac", "slp_all",
              "slp_50", "ktvt_rng", "ktvt_std", "pfx_gr_rmse"):
        z[k] = g[k].to_numpy(np.float32)
    # Translate the PF-centered volume posterior to the common datum.
    for suffix in ("raw", "c15", "c60", "c200", "fw", "bw", "whole"):
        z[f"tdpf_{suffix}_absolute"] = pf+z[f"tdpf_{suffix}_mean"]
        z[f"cons_{suffix}"] = (
            .5*z[f"tda_{suffix}_mean"]+
            .5*z[f"tdpf_{suffix}_absolute"])
        z[f"disagree_{suffix}"] = (
            z[f"tda_{suffix}_mean"]-z[f"tdpf_{suffix}_absolute"])
    # Full observed sequence summaries are legal for test wells.
    for k in ("pfx_gr_rmse", "pf_vs_beam", "sp_dmin",
              "tda_whole_sd", "tdpf_whole_sd", "disagree_whole"):
        v = z[k].to_numpy(float)
        z[f"ctx_{k}_mean"] = np.float32(np.mean(v))
        z[f"ctx_{k}_std"] = np.float32(np.std(v))
        z[f"ctx_{k}_p90"] = np.float32(np.quantile(v, .9))
    z["well"] = g.well.iloc[0]
    z["target"] = g.target.to_numpy(np.float32)
    return z


def main(limit=120, folds=5):
    d = pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
    oo = joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
    rng = np.random.RandomState(731)
    wells = np.array(sorted(d.well.unique()))
    # After a failed pilot gate for the learned model, validate fixed posterior
    # controls streaming over all wells without retaining the huge cost volume.
    if limit < 0:
        names = [f"cons_{x}" for x in
                 ("raw", "c15", "c60", "c200", "fw", "bw", "whole")]
        blends = (.15, .3, .45, .6, .75, .9, 1)
        sse = {(n, b): 0. for n in names for b in blends}
        base_sse = 0.; count = 0
        for i, (_, g) in enumerate(d.groupby("well", sort=True)):
            ix = g.index.to_numpy()
            base = (.55*np.asarray(oo["lgb123"])[ix]+
                    .20*np.asarray(oo["lgb7"])[ix]+
                    .15*np.asarray(oo["xgb"])[ix]+
                    .10*np.asarray(oo["cat"])[ix]).astype(np.float32)
            f = well_features(g, base)
            y = f.target.to_numpy(float)
            base_sse += np.sum((y-base)**2); count += len(y)
            for n in names:
                v = f[n].to_numpy(float)
                for b in blends:
                    sse[n, b] += np.sum((y-((1-b)*base+b*v))**2)
            if (i+1) % 50 == 0:
                print("stream", i+1, flush=True)
        q = pd.DataFrame([{"feature": n, "blend": b,
                           "rmse": np.sqrt(v/count)}
                          for (n, b), v in sse.items()]).sort_values("rmse")
        result = {"wells": len(wells), "rows": count,
                  "base": np.sqrt(base_sse/count),
                  "best_direct": q.iloc[0].to_dict()}
        q.to_csv(OUT/"full_direct.csv", index=False)
        (OUT/"full_direct_summary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        return
    if limit and limit < len(wells):
        wells = rng.choice(wells, limit, replace=False)
        d = d[d.well.isin(wells)]
    pieces = []
    for i, (_, g) in enumerate(d.groupby("well", sort=True)):
        ix = g.index.to_numpy()
        base = (.55*np.asarray(oo["lgb123"])[ix]+
                .20*np.asarray(oo["lgb7"])[ix]+
                .15*np.asarray(oo["xgb"])[ix]+
                .10*np.asarray(oo["cat"])[ix]).astype(np.float32)
        pieces.append(well_features(g, base))
        if (i+1) % 25 == 0:
            print("features", i+1, flush=True)
    f = pd.concat(pieces).sort_index()
    cols = [c for c in f if c not in ("well", "target", "base")]
    X = f[cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = f.target.to_numpy(float)
    base = f.base.to_numpy(float)
    groups = f.well.to_numpy()
    pred = np.zeros(len(f), np.float32)
    importance = np.zeros(len(cols))
    fold_rows = []
    for fold, (tr, va) in enumerate(GroupKFold(folds).split(X, groups=groups)):
        q = pd.DataFrame({"i": tr, "w": groups[tr]})
        tr2 = np.concatenate([
            a.i.to_numpy() if len(a) <= 1200 else
            rng.choice(a.i.to_numpy(), 1200, False)
            for _, a in q.groupby("w", sort=False)])
        m = LGBMRegressor(
            objective="huber", n_estimators=700, learning_rate=.02,
            num_leaves=20, max_depth=7, min_child_samples=160,
            max_bin=127, reg_alpha=3, reg_lambda=20,
            colsample_bytree=.72, subsample=.85, subsample_freq=1,
            verbosity=-1, random_state=731+fold,
            n_jobs=max(1, (os.cpu_count() or 4)//2))
        m.fit(X.iloc[tr2], (y-base)[tr2])
        pred[va] = m.predict(X.iloc[va])
        importance += m.feature_importances_
        fold_rows.append({"fold": fold, "base": rmse(y[va], base[va]),
                          "raw": rmse(y[va], base[va]+pred[va])})
        print(fold_rows[-1], flush=True)
    grid = []
    for blend in (.05, .1, .15, .2, .3, .4, .55, .7, .85, 1):
        for clip in (1, 2, 4, 6, 10, 16):
            p = base+blend*np.clip(pred, -clip, clip)
            grid.append({"blend": blend, "clip": clip, "rmse": rmse(y, p)})
    grid = pd.DataFrame(grid).sort_values("rmse")
    # Direct continuous posterior controls.
    direct = []
    for c in [x for x in f if x.startswith("cons_")]:
        for blend in (.15, .3, .45, .6, .75, .9, 1):
            direct.append({"feature": c, "blend": blend,
                           "rmse": rmse(y, (1-blend)*base+blend*f[c])})
    direct = pd.DataFrame(direct).sort_values("rmse")
    result = {"wells": int(f.well.nunique()), "rows": len(f),
              "base": rmse(y, base), "raw": rmse(y, base+pred),
              "best": grid.iloc[0].to_dict(),
              "best_direct": direct.iloc[0].to_dict(),
              "folds": fold_rows}
    tag = "pilot" if limit else "full"
    print(json.dumps(result, indent=2), flush=True)
    grid.to_csv(OUT/f"{tag}_grid.csv", index=False)
    direct.to_csv(OUT/f"{tag}_direct.csv", index=False)
    pd.DataFrame({"feature": cols, "importance": importance}).sort_values(
        "importance", ascending=False).to_csv(OUT/f"{tag}_importance.csv",
                                               index=False)
    (OUT/f"{tag}_summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
