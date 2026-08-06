"""Visible-prefix reconstruction of the typewell for complete-path scoring.

Only TVT_input/GR pairs before Prediction Start modify the reference curve.
The hidden suffix GR is then projected into 0.5-ft TVT bins and correlated
against that corrected reference.  Truth is used only for final scoring.
"""
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/prefix_corrected_typewell_path"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "exp"))
from whole_well_gr_warp_selector_cv import candidate_paths, robust_calibrate, rmse
from binned_stratigraphic_correlation_cv import percentile_normalize, corr

BIN = 0.5
BANDWIDTHS = (1.0, 4.0, 12.0)
WEIGHTS = (0.25, 0.6, 1.0)
DECAYS = (3.0, 10.0, 40.0)


def corrected_references(hw, tw):
    tt = tw.TVT.to_numpy(float)
    tg0 = tw.GR.interpolate(limit_direction="both").fillna(tw.GR.median()).to_numpy(float)
    a, b, prefix_rmse = robust_calibrate(hw, tw)
    base = a*tg0+b
    known = hw.TVT_input.notna().to_numpy() & hw.GR.notna().to_numpy()
    vt = hw.TVT_input.to_numpy(float)[known]
    hg = hw.GR.to_numpy(float)[known]
    if len(vt) < 20:
        return tt, {"base": percentile_normalize(base)}, prefix_rmse
    residual = hg-np.interp(vt, tt, base)
    lo = np.floor(vt.min()/BIN)*BIN; hi = np.ceil(vt.max()/BIN)*BIN
    grid = np.arange(lo, hi+BIN/2, BIN)
    bi = np.clip(np.floor((vt-lo)/BIN).astype(int), 0, len(grid)-1)
    count = np.bincount(bi, minlength=len(grid))
    total = np.bincount(bi, weights=residual, minlength=len(grid))
    occupied = np.flatnonzero(count)
    rgrid = np.interp(np.arange(len(grid)), occupied,
                      total[occupied]/count[occupied])
    # Distance to the nearest actually observed TVT bin controls extrapolation.
    nearest = np.minimum.reduce([np.abs(grid[:, None]-vt[None, :])], axis=0).min(1)
    refs = {"base": percentile_normalize(base)}
    for bandwidth in BANDWIDTHS:
        smooth = gaussian_filter1d(rgrid, bandwidth/BIN, mode="nearest")
        for weight in WEIGHTS:
            for decay in DECAYS:
                support = np.exp(-nearest/decay)
                correction = np.interp(tt, grid, smooth*support,
                                       left=0.0, right=0.0)
                key = f"bw{bandwidth:g}_w{weight:g}_d{decay:g}"
                refs[key] = percentile_normalize(base+weight*correction)
    return tt, refs, prefix_rmse


def candidate_evidence(hw, rows, paths, tt, refs):
    station = rows.id.astype(str).str.rsplit("_", n=1).str[1].astype(int).to_numpy()
    raw = hw.GR.to_numpy(float)
    observed = percentile_normalize(raw)[station]
    valid = np.isfinite(observed)
    last = float(rows.last_known_tvt.iloc[0])
    evidence = {key: np.full(len(paths), -8.0) for key in refs}
    for j, path in enumerate(paths):
        tvt = last+path
        use = valid & np.isfinite(tvt)
        if use.sum() < 20:
            continue
        origin = np.floor(tvt[use].min()/BIN)*BIN
        bins = np.floor((tvt[use]-origin)/BIN).astype(int)
        count = np.bincount(bins); total = np.bincount(bins, weights=observed[use])
        occupied = np.flatnonzero(count >= 2)
        if len(occupied) < 8:
            occupied = np.flatnonzero(count)
        lateral = total[occupied]/count[occupied]
        centers = origin+(occupied+.5)*BIN
        for key, ref in refs.items():
            r = np.clip(corr(lateral, np.interp(centers, tt, ref)), -.999, .999)
            evidence[key][j] = np.arctanh(r)
    return evidence


def main(limit=80):
    frame = pd.read_pickle(ROOT / "r_v4b/train_feats.pkl")
    oof = joblib.load(ROOT / "r_v4b/stack_v4_oofs.joblib")["oofs"]
    stats=[]
    for well, group in frame.groupby("well"):
        base, _, _ = candidate_paths(group, oof)
        stats.append((well, rmse(group.target, base)))
    stats=pd.DataFrame(stats, columns=["well", "base_rmse"])
    stats["bin"]=pd.qcut(stats.base_rmse, 4, labels=False)
    rng=np.random.RandomState(802); selected=[]
    for _, group in stats.groupby("bin"):
        selected.extend(rng.choice(group.well, min(len(group), limit//4), False))
    frame=frame[frame.well.isin(selected)]

    store={}; diagnostics=[]
    for i,(well,group) in enumerate(frame.groupby("well",sort=True)):
        hw=pd.read_csv(ROOT/f"data/train/{well}__horizontal_well.csv")
        tw=pd.read_csv(ROOT/f"data/train/{well}__typewell.csv").sort_values("TVT")
        base,paths,labels=candidate_paths(group,oof)
        tt,refs,pfx=corrected_references(hw,tw)
        evidence=candidate_evidence(hw,group,paths,tt,refs)
        store[well]=(group.target.to_numpy(float),base,paths,labels,evidence)
        diagnostics.append({"well":well,"prefix_rmse":pfx,"references":len(refs)})
        if (i+1)%10==0: print(f"wells={i+1}",flush=True)

    configs=[]
    keys=next(iter(store.values()))[-1].keys()
    for key in keys:
        for prior in (0.1,0.3,0.6):
            for temperature in (0.15,0.25,0.5):
                yy=[];bb=[];pp=[]
                for y,base,paths,labels,evidence in store.values():
                    complexity=np.asarray([(s/20)**2+(q/12)**2+(c/6)**2
                                           for _,s,q,c in labels])
                    cost=-evidence[key]+prior*complexity
                    scale=np.std(cost)*temperature+1e-6
                    weight=np.exp(np.clip(-(cost-cost.min())/scale,-30,0));weight/=weight.sum()
                    yy.append(y);bb.append(base);pp.append(weight@paths)
                ya,ba,pa=map(np.concatenate,(yy,bb,pp))
                for blend in (0.1,0.2,0.3,0.4,0.5):
                    configs.append({"reference":key,"prior":prior,
                                    "temperature":temperature,"blend":blend,
                                    "rmse":rmse(ya,(1-blend)*ba+blend*pa)})
    grid=pd.DataFrame(configs).sort_values("rmse")
    ya=np.concatenate([v[0] for v in store.values()]);ba=np.concatenate([v[1] for v in store.values()])
    base_rows=grid[grid.reference.eq("base")]
    summary={"wells":len(store),"rows":len(ya),"baseline":rmse(ya,ba),
             "best_uncorrected":base_rows.iloc[0].to_dict(),
             "best_corrected":grid[~grid.reference.eq("base")].iloc[0].to_dict(),
             "correction_gain_over_uncorrected":float(base_rows.rmse.min()-grid[~grid.reference.eq("base")].rmse.min()),
             "truth_used_only_for_final_scoring":True}
    grid.to_csv(OUT/"grid.csv",index=False);pd.DataFrame(diagnostics).to_csv(OUT/"wells.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main(int(sys.argv[1]) if len(sys.argv)>1 else 80)
