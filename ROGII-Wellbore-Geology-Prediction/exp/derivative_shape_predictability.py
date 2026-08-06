"""Quantify integrated structural-derivative baselines and oracle ceilings."""
from pathlib import Path
import glob
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"exp/results/derivative_shape"
OUT.mkdir(parents=True, exist_ok=True)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def main(limit=200):
    files = sorted(glob.glob(str(ROOT/"data/train/*__horizontal_well.csv")))
    rng = np.random.RandomState(806)
    if limit and limit < len(files):
        files = sorted(rng.choice(files, limit, False))
    store = {}
    derivative_rows = []
    for f in files:
        d = pd.read_csv(f)
        known = np.flatnonzero(d.TVT_input.notna().to_numpy())
        if len(known) < 50 or known[-1] >= len(d)-5:
            continue
        ps = int(known[-1]); md = d.MD.to_numpy(float)
        S = d.TVT.to_numpy(float)+d.Z.to_numpy(float)
        m = np.arange(len(d)) > ps
        x = md-md[ps]
        preds = {}
        # Zero structural derivative is exact trajectory-only physics.
        preds["zero"] = np.full(m.sum(), S[ps])-d.Z.to_numpy(float)[m]
        # Prefix structural dip, robustly estimated at several horizons.
        for win in (100, 300, 800, 1600):
            ix = np.arange(max(0, ps-win+1), ps+1)
            sl = np.polyfit(md[ix]-md[ps], S[ix]-S[ps], 1)[0]
            sl = np.clip(sl, -.03, .03)
            for sh in (.1, .25, .5, 1.):
                s_pred = S[ps]+sh*sl*x[m]
                preds[f"prefix{win}_sh{sh:g}"] = s_pred-d.Z.to_numpy(float)[m]
        # Oracle constant/linear derivative and quadratic surface shape.
        xe = x[m]; se = S[m]-S[ps]
        for deg in (1, 2, 3):
            coef = np.polyfit(xe, se, deg)
            coef[-1] = 0.  # integrate from the observed boundary exactly
            preds[f"oracle_poly{deg}"] = S[ps]+np.polyval(coef, xe)-d.Z.to_numpy(float)[m]
        # Smooth derivative oracle: low-pass true dS/dMD, integrated.
        from scipy.ndimage import gaussian_filter1d
        ds = np.gradient(S, md)
        for sig in (20, 60, 150):
            q = gaussian_filter1d(ds, sig)
            integ = np.zeros(len(d))
            integ[ps+1:] = np.cumsum(
                .5*(q[ps:-1]+q[ps+1:])*np.diff(md[ps:]))
            preds[f"oracle_smooth{sig}"] = S[ps]+integ[m]-d.Z.to_numpy(float)[m]
        store[Path(f).name.split("__")[0]] = {
            "y": d.TVT.to_numpy(float)[m], "preds": preds}
        derivative_rows.append({
            "well": Path(f).name.split("__")[0],
            "suffix_dS_mean": float(np.mean(ds[m])),
            "suffix_dS_std": float(np.std(ds[m])),
            "prefix_dS_mean": float(np.mean(ds[max(0, ps-500):ps+1])),
            "corr_gr_dS": float(np.corrcoef(
                np.nan_to_num(d.GR.to_numpy(float)[m],
                              nan=np.nanmedian(d.GR)), ds[m])[0, 1]),
        })
    methods = list(next(iter(store.values()))["preds"])
    scores = {}
    for k in methods:
        y = np.concatenate([q["y"] for q in store.values()])
        p = np.concatenate([q["preds"][k] for q in store.values()])
        scores[k] = rmse(y, p)
    dr = pd.DataFrame(derivative_rows)
    result = {"wells": len(store), "rows": int(sum(len(q["y"]) for q in store.values())),
              "scores": dict(sorted(scores.items(), key=lambda q: q[1])),
              "prefix_suffix_mean_derivative_corr": float(
                  dr[["prefix_dS_mean", "suffix_dS_mean"]].corr().iloc[0, 1]),
              "median_abs_gr_derivative_corr": float(
                  dr.corr_gr_dS.abs().median())}
    print(json.dumps(result, indent=2), flush=True)
    dr.to_csv(OUT/"predictability_wells.csv", index=False)
    (OUT/"predictability_summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 200)
