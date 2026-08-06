"""Alias-aware posterior over complete smooth TVT curves, then grouped GBDT.

Only inference-time columns are used.  Candidate paths are deterministic
functions of OOF physical/tabular paths.  GR likelihoods come from horizontal
GR versus typewell GR residual grids (tda*, tdpf*).  No target is used when
constructing or weighting candidates.
"""
from pathlib import Path
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/posterior_curve_regression"
OUT.mkdir(parents=True, exist_ok=True)
OFF = np.array([-40, -20, -10, -5, 0, 5, 10, 20, 40], np.float32)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def robust_poly(x, y, deg=3):
    z = (x-x.mean())/(x.std()+1e-6)
    keep = np.ones(len(x), bool)
    coef = np.polyfit(z, y, deg)
    for _ in range(3):
        fit = np.polyval(coef, z)
        r = y-fit
        s = 1.4826*np.median(np.abs(r[keep]))+1e-4
        keep = np.abs(r) < 2.5*s
        if keep.sum() < deg+5:
            break
        coef = np.polyfit(z[keep], y[keep], deg)
    return np.polyval(coef, z).astype(np.float32)


def interp_rows(grid, query):
    """Row-wise linear interpolation of residual grid at query offsets."""
    q = np.clip(query, OFF[0], OFF[-1])
    j = np.searchsorted(OFF, q, side="right")-1
    j = np.clip(j, 0, len(OFF)-2)
    lo, hi = OFF[j], OFF[j+1]
    a = (q-lo)/(hi-lo)
    ii = np.arange(len(q))
    return grid[ii, j]*(1-a)+grid[ii, j+1]*a


def candidate_lattice(g, base):
    """Diverse smooth complete-well curves; no hard path selection."""
    n = len(g)
    u = np.linspace(-1, 1, n, dtype=np.float32)
    pf = g.pf_d.to_numpy(np.float32)
    beam = g.beam_d.to_numpy(np.float32)
    smooth = gaussian_filter1d(base, min(16, max(2, n/30)), mode="nearest")
    poly = robust_poly(g.d_md.to_numpy(float), base, 3)
    centers = [
        base, .75*base+.25*smooth, .50*base+.50*poly,
        .8*pf+.2*beam, .65*base+.35*pf,
    ]
    paths, kinds = [], []
    # Translation/slope families explicitly preserve alias branches.
    for ci, center in enumerate(centers):
        for shift in (-12, -6, -3, 0, 3, 6, 12):
            for slope in (-6, 0, 6):
                paths.append(center + shift + slope*u)
                kinds.append(ci)
    # Curved alternatives around the strongest center.
    for curve in (-8, -4, 4, 8):
        paths.append(base + curve*(u*u-.33))
        kinds.append(5)
    return np.stack(paths, axis=1).astype(np.float32), np.asarray(kinds)


def softstats(paths, costs, temperature):
    """Posterior summaries. paths/costs are [rows,candidates]."""
    logits = -(costs-costs.min(1, keepdims=True))/temperature
    logits = np.clip(logits, -35, 0)
    w = np.exp(logits)
    w /= w.sum(1, keepdims=True)+1e-9
    mean = np.sum(w*paths, 1)
    var = np.sum(w*(paths-mean[:, None])**2, 1)
    entropy = -np.sum(w*np.log(w+1e-9), 1)
    top = np.partition(w, -2, axis=1)[:, -2:]
    # Weighted branch masses around the zero/datum split expose bimodality.
    neg = np.sum(w*(paths < 0), 1)
    pos = np.sum(w*(paths >= 0), 1)
    return mean, np.sqrt(var), entropy, top[:, 1]-top[:, 0], neg, pos


def make_well(g, oo):
    idx = g.index.to_numpy()
    raw = {k: np.asarray(v)[idx].astype(np.float32) for k, v in oo.items()}
    base = (.55*raw["lgb123"]+.20*raw["lgb7"]+
            .15*raw["xgb"]+.10*raw["cat"]).astype(np.float32)
    paths, kinds = candidate_lattice(g, base)
    tda = g[[f"tda{int(o)}" for o in OFF]].to_numpy(np.float32)
    tdpf = g[[f"tdpf{int(o)}" for o in OFF]].to_numpy(np.float32)
    pf = g.pf_d.to_numpy(np.float32)

    # Robust amplitude normalization reduces domination by GR spikes.
    scale_a = np.nanmedian(np.abs(tda), 1)+5
    scale_p = np.nanmedian(np.abs(tdpf), 1)+5
    local = np.empty_like(paths)
    for j in range(paths.shape[1]):
        ra = interp_rows(tda, paths[:, j])/scale_a
        rp = interp_rows(tdpf, paths[:, j]-pf)/scale_p
        # Cauchy-like loss preserves multiple plausible aliases.
        local[:, j] = .45*np.log1p(ra*ra)+.55*np.log1p(rp*rp)
    # Costs at several contextual horizons plus a whole-suffix likelihood.
    costs = {
        "l15": uniform_filter1d(local, size=min(31, len(g)), axis=0,
                                mode="nearest"),
        "l60": uniform_filter1d(local, size=min(121, len(g)), axis=0,
                                mode="nearest"),
        "global": np.broadcast_to(local.mean(0), local.shape),
    }
    z = pd.DataFrame(index=g.index)
    for name, c in costs.items():
        # Temperature relative to across-candidate dispersion is stable across
        # wells with very different GR quality.
        temp = np.median(np.std(c, axis=1))+0.025
        stats = softstats(paths, c, temp)
        for suffix, val in zip(
                ("mean", "sd", "entropy", "margin", "negmass", "posmass"),
                stats):
            z[f"post_{name}_{suffix}"] = val.astype(np.float32)
    # Alias-aware mixture: local evidence for shape, global for branch/datum.
    z["post_mix"] = (.65*z.post_l60_mean+.35*z.post_global_mean)
    z["post_disagree"] = z.post_l15_mean-z.post_global_mean
    z["base"] = base
    z["cand_min"] = paths.min(1)
    z["cand_max"] = paths.max(1)
    z["cand_sd"] = paths.std(1)
    for k in ("pf_d", "pf3_d", "beam_d", "pf_vs_beam", "sp_mean_d",
              "sp_std", "sp_dmin", "ncc8_d", "ncc8_s", "ncc15_d",
              "ncc15_s", "ncc25_d", "ncc25_s", "d_md", "d_z", "d_xy",
              "dz_dmd", "gr", "gr_m21", "gr_s21", "frac", "slp_all",
              "slp_50", "ktvt_rng", "ktvt_std", "pfx_gr_rmse"):
        z[k] = g[k].to_numpy(np.float32)
    # Whole-well legal context.
    for k in ("post_global_sd", "post_global_entropy", "post_disagree",
              "pfx_gr_rmse", "pf_vs_beam", "sp_dmin"):
        v = z[k].to_numpy(float)
        z[f"ctx_{k}_mean"] = np.float32(np.mean(v))
        z[f"ctx_{k}_p90"] = np.float32(np.quantile(v, .9))
    z["well"] = g.well.iloc[0]
    z["target"] = g.target.to_numpy(np.float32)
    return z


def main(limit=120, folds=5):
    d = pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
    oo = joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
    rng = np.random.RandomState(730)
    wells = np.array(sorted(d.well.unique()))
    if limit and limit < len(wells):
        wells = rng.choice(wells, limit, replace=False)
        d = d[d.well.isin(wells)]
    pieces = []
    for i, (_, g) in enumerate(d.groupby("well", sort=True)):
        pieces.append(make_well(g, oo))
        if (i+1) % 25 == 0:
            print("features", i+1, flush=True)
    f = pd.concat(pieces).sort_index()
    cols = [c for c in f if c not in ("well", "target", "base")]
    X = f[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y, base, groups = f.target.to_numpy(), f.base.to_numpy(), f.well.to_numpy()
    pred_r = np.zeros(len(f), np.float32)
    fold_rows = []
    for fold, (tr, va) in enumerate(GroupKFold(folds).split(X, groups=groups)):
        # Bound large-well influence only in fitting; score all validation rows.
        q = pd.DataFrame({"i": tr, "w": groups[tr]})
        tr2 = np.concatenate([
            a.i.to_numpy() if len(a) <= 1200 else
            rng.choice(a.i.to_numpy(), 1200, replace=False)
            for _, a in q.groupby("w", sort=False)])
        m = LGBMRegressor(
            objective="huber", n_estimators=600, learning_rate=.025,
            num_leaves=24, max_depth=7, min_child_samples=140,
            reg_alpha=2, reg_lambda=18, colsample_bytree=.78,
            subsample=.85, subsample_freq=1, verbosity=-1,
            random_state=730+fold, n_jobs=max(1, (os.cpu_count() or 4)//2))
        m.fit(X.iloc[tr2], (y-base)[tr2])
        pred_r[va] = m.predict(X.iloc[va])
        fold_rows.append({"fold": fold, "base": rmse(y[va], base[va]),
                          "raw": rmse(y[va], base[va]+pred_r[va])})
        print(fold_rows[-1], flush=True)
    grid = []
    for blend in (.1, .2, .3, .4, .55, .7, .85, 1):
        for clip in (2, 4, 6, 10, 16):
            p = base+blend*np.clip(pred_r, -clip, clip)
            grid.append({"blend": blend, "clip": clip, "rmse": rmse(y, p)})
    grid = pd.DataFrame(grid).sort_values("rmse")
    direct = []
    for name in ("post_l15_mean", "post_l60_mean", "post_global_mean",
                 "post_mix"):
        for blend in (0, .15, .3, .45, .6, .75, .9, 1):
            p = (1-blend)*base+blend*f[name].to_numpy()
            direct.append({"posterior": name, "blend": blend,
                           "rmse": rmse(y, p)})
    direct = pd.DataFrame(direct).sort_values("rmse")
    result = {"wells": int(f.well.nunique()), "rows": len(f),
              "base_rmse": rmse(y, base), "raw_rmse": rmse(y, base+pred_r),
              "posterior_only": rmse(y, f.post_mix),
              "best": grid.iloc[0].to_dict(),
              "best_direct": direct.iloc[0].to_dict(), "folds": fold_rows}
    print(json.dumps(result, indent=2), flush=True)
    tag = "pilot" if limit else "full"
    grid.to_csv(OUT/f"{tag}_grid.csv", index=False)
    direct.to_csv(OUT/f"{tag}_direct_grid.csv", index=False)
    (OUT/f"{tag}_summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
