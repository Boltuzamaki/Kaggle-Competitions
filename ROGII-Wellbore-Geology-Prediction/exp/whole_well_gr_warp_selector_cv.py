"""Legal full-suffix horizontal/typewell GR warp scoring.

Candidate TVT curves are smooth datum/slope/curvature warps around honest OOF
paths. Horizontal-vs-typewell GR amplitude is calibrated only on visible TVT
prefix rows. Selection uses the entire observed hidden-suffix GR sequence.
"""
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"exp/results/whole_well_gr_warp_selector"
OUT.mkdir(parents=True, exist_ok=True)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def robust_calibrate(hw, tw):
    """Map typewell GR amplitude to horizontal GR using visible prefix only."""
    known = hw.TVT_input.notna().to_numpy()
    tvt = hw.TVT_input.to_numpy(float)[known]
    h = hw.GR.to_numpy(float)[known]
    tg = np.interp(tvt, tw.TVT.to_numpy(float), tw.GR.to_numpy(float))
    ok = np.isfinite(h+tg)
    h, tg = h[ok], tg[ok]
    if len(h) < 20:
        return 1., 0., 99.
    keep = np.ones(len(h), bool)
    a, b = 1., 0.
    for _ in range(4):
        A = np.c_[tg[keep], np.ones(keep.sum())]
        a, b = np.linalg.lstsq(A, h[keep], rcond=None)[0]
        a = float(np.clip(a, .2, 4.))
        r = h-(a*tg+b)
        s = 1.4826*np.median(np.abs(r[keep]))+3
        keep = np.abs(r) < 2.5*s
    return a, float(b), float(np.sqrt(np.mean(r[keep]**2)))


def candidate_paths(g, oof):
    ix = g.index.to_numpy()
    raw = {k: np.asarray(v)[ix].astype(float) for k, v in oof.items()}
    base = .55*raw["lgb123"]+.20*raw["lgb7"]+.15*raw["xgb"]+.10*raw["cat"]
    pf = g.pf_d.to_numpy(float)
    centers = [base, .7*base+.3*pf]
    u = np.linspace(-1, 1, len(g))
    paths, labels = [], []
    for ci, c in enumerate(centers):
        for shift in (-20, -12, -7, -3, 0, 3, 7, 12, 20):
            for slope in (-12, -6, 0, 6, 12):
                for curve in (-6, 0, 6):
                    paths.append(c+shift+slope*u+curve*(u*u-1/3))
                    labels.append((ci, shift, slope, curve))
    return base, np.asarray(paths), labels


def costs_for(hw, tw, rows, paths):
    idx = rows.id.astype(str).str.rsplit("_", n=1).str[1].astype(int).to_numpy()
    hgr = hw.GR.to_numpy(float)[idx]
    fill = np.nanmedian(hgr)
    hgr = np.nan_to_num(hgr, nan=fill)
    last = float(rows.last_known_tvt.iloc[0])
    a, b, prefix_rmse = robust_calibrate(hw, tw)
    tt, tg = tw.TVT.to_numpy(float), tw.GR.to_numpy(float)
    # candidates x rows typewell traces
    cand_gr = np.interp((last+paths).ravel(), tt, tg).reshape(paths.shape)
    cand_gr = a*cand_gr+b
    hscale = 1.4826*np.median(np.abs(hgr-np.median(hgr)))+8
    per = []
    for sig, wt in ((0, .30), (5, .25), (20, .25), (60, .20)):
        if sig:
            hh = gaussian_filter1d(hgr, sig)
            cc = gaussian_filter1d(cand_gr, sig, axis=1)
        else:
            hh, cc = hgr, cand_gr
        r = (cc-hh[None])/hscale
        # Cauchy loss is robust to facies/amplitude mismatch.
        per.append(wt*np.mean(np.log1p(r*r), axis=1))
    cost = np.sum(per, axis=0)
    # Shape-only derivative cost can disambiguate amplitude aliases.
    dh = gaussian_filter1d(np.gradient(hgr), 5)
    dc = gaussian_filter1d(np.gradient(cand_gr, axis=1), 5, axis=1)
    cost += .12*np.mean(np.log1p(((dc-dh[None])/(hscale/5+1))**2), axis=1)
    return cost, prefix_rmse


def main(limit=200):
    d = pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
    oof = joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
    # Stratify the pilot over baseline difficulty quartiles, so a matcher is
    # not accepted merely because random wells happened to be easy.
    stats = []
    for w, g in d.groupby("well"):
        base, _, _ = candidate_paths(g, oof)
        stats.append((w, rmse(g.target, base)))
    st = pd.DataFrame(stats, columns=["well", "base_rmse"])
    st["bin"] = pd.qcut(st.base_rmse, 4, labels=False, duplicates="drop")
    rng = np.random.RandomState(802)
    selected = []
    for _, q in st.groupby("bin"):
        selected.extend(rng.choice(q.well, min(len(q), limit//4),
                                   replace=False))
    d = d[d.well.isin(selected)]
    records = []
    truth, baseline, chosen = [], [], []
    oracle_pred = []
    soft_pred = []
    for i, (w, g) in enumerate(d.groupby("well", sort=True)):
        hf = ROOT/f"data/train/{w}__horizontal_well.csv"
        tf = ROOT/f"data/train/{w}__typewell.csv"
        hw, tw = pd.read_csv(hf), pd.read_csv(tf).sort_values("TVT")
        base, paths, labels = candidate_paths(g, oof)
        cost, pfx = costs_for(hw, tw, g, paths)
        y = g.target.to_numpy(float)
        losses = np.sqrt(np.mean((paths-y[None])**2, axis=1))
        ci, oi = int(np.argmin(cost)), int(np.argmin(losses))
        # Alias-aware posterior average; temperature is relative within well.
        temp = np.std(cost)+1e-4
        ww = np.exp(np.clip(-(cost-cost.min())/(.35*temp), -30, 0))
        ww /= ww.sum()
        sp = ww@paths
        truth.append(y); baseline.append(base); chosen.append(paths[ci])
        oracle_pred.append(paths[oi]); soft_pred.append(sp)
        records.append({"well": w, "rows": len(g), "prefix_gr_rmse": pfx,
                        "base_rmse": rmse(y, base),
                        "selected_rmse": losses[ci],
                        "soft_rmse": rmse(y, sp),
                        "oracle_rmse": losses[oi],
                        "selected": str(labels[ci]),
                        "oracle": str(labels[oi]),
                        "cost_margin": float(np.partition(cost, 1)[1]-cost.min())})
        if (i+1) % 25 == 0:
            print("wells", i+1, flush=True)
    y, base, pick, oracle, soft = map(np.concatenate, (
        truth, baseline, chosen, oracle_pred, soft_pred))
    # Conservative blends reveal whether likelihood has incremental signal.
    grid = []
    for name, p in (("hard", pick), ("soft", soft)):
        for blend in (0, .1, .2, .3, .4, .55, .7, .85, 1):
            grid.append({"method": name, "blend": blend,
                         "rmse": rmse(y, (1-blend)*base+blend*p)})
    grid = pd.DataFrame(grid).sort_values("rmse")
    wr = pd.DataFrame(records)
    result = {"wells": len(wr), "rows": len(y),
              "baseline": rmse(y, base), "hard": rmse(y, pick),
              "soft": rmse(y, soft), "oracle": rmse(y, oracle),
              "best_blend": grid.iloc[0].to_dict(),
              "selector_equals_oracle_rate": float(
                  (wr.selected == wr.oracle).mean()),
              "well_win_rate_hard": float(
                  (wr.selected_rmse < wr.base_rmse).mean()),
              "well_win_rate_soft": float(
                  (wr.soft_rmse < wr.base_rmse).mean())}
    print(json.dumps(result, indent=2), flush=True)
    wr.to_csv(OUT/"pilot_wells.csv", index=False)
    grid.to_csv(OUT/"pilot_grid.csv", index=False)
    (OUT/"pilot_summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 200)
