"""Score complete TVT paths against the well's own visible-prefix GR.

This implements the discussion hint that lateral GR has higher resolution than
the typewell.  A candidate receives evidence only where its hidden TVT revisits
the TVT interval covered by visible TVT_input.  Unsupported wells retain the
baseline prediction.  Truth is used only for final scoring and diagnostics.
"""
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/self_lateral_gr_path"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "exp"))
from whole_well_gr_warp_selector_cv import candidate_paths, rmse
from binned_stratigraphic_correlation_cv import percentile_normalize, corr

BIN = 0.5


def visible_reference(hw, smooth_ft):
    mask = hw.TVT_input.notna().to_numpy() & hw.GR.notna().to_numpy()
    tvt = hw.TVT_input.to_numpy(float)[mask]
    gr = percentile_normalize(hw.GR.to_numpy(float))[mask]
    if len(tvt) < 30:
        return None
    lo = np.floor(tvt.min()/BIN)*BIN
    hi = np.ceil(tvt.max()/BIN)*BIN
    grid = np.arange(lo, hi+BIN/2, BIN)
    index = np.clip(np.floor((tvt-lo)/BIN).astype(int), 0, len(grid)-1)
    count = np.bincount(index, minlength=len(grid))
    total = np.bincount(index, weights=gr, minlength=len(grid))
    occupied = np.flatnonzero(count)
    if len(occupied) < 20:
        return None
    curve = np.interp(np.arange(len(grid)), occupied,
                      total[occupied]/count[occupied])
    if smooth_ft:
        curve = gaussian_filter1d(curve, smooth_ft/BIN, mode="nearest")
    return grid, curve, float(tvt.min()), float(tvt.max())


def self_scores(hw, rows, paths, smooth_ft):
    ref = visible_reference(hw, smooth_ft)
    if ref is None:
        return np.zeros(len(paths)), np.zeros(len(paths))
    grid, curve, lo, hi = ref
    station = rows.id.astype(str).str.rsplit("_", n=1).str[1].astype(int).to_numpy()
    gr = percentile_normalize(hw.GR.to_numpy(float))[station]
    last = float(rows.last_known_tvt.iloc[0])
    score = np.zeros(len(paths)); support = np.zeros(len(paths))
    for j, path in enumerate(paths):
        candidate = last+path
        use = np.isfinite(gr) & (candidate >= lo) & (candidate <= hi)
        support[j] = use.mean()
        if use.sum() < 20:
            continue
        # Correlation is evaluated along the hidden station sequence. Repeated
        # TVT visits remain useful rather than being averaged away.
        r = np.clip(corr(gr[use], np.interp(candidate[use], grid, curve)), -.999, .999)
        score[j] = np.arctanh(r) * np.sqrt(use.mean())
    return score, support


def weighted_path_fast(weight, paths, labels):
    """Exact weight@paths using the candidate family's five coefficients."""
    lab = np.asarray(labels, float)
    u = np.linspace(-1, 1, paths.shape[1])
    v = u*u-1/3
    zero0 = labels.index((0, 0, 0, 0))
    zero1 = labels.index((1, 0, 0, 0))
    center_mass = np.bincount(lab[:, 0].astype(int), weights=weight,
                              minlength=2)
    return (center_mass[0]*paths[zero0] + center_mass[1]*paths[zero1]
            + np.dot(weight, lab[:, 1])
            + np.dot(weight, lab[:, 2])*u
            + np.dot(weight, lab[:, 3])*v)


def main(limit=200):
    frame = pd.read_pickle(ROOT / "r_v4b/train_feats.pkl")
    oof = joblib.load(ROOT / "r_v4b/stack_v4_oofs.joblib")["oofs"]
    stats=[]
    for well, group in frame.groupby("well"):
        base, _, _ = candidate_paths(group, oof)
        stats.append((well, rmse(group.target, base)))
    stats=pd.DataFrame(stats, columns=["well","base_rmse"])
    stats["bin"]=pd.qcut(stats.base_rmse, 4, labels=False)
    rng=np.random.RandomState(913); selected=[]
    for _, group in stats.groupby("bin"):
        selected.extend(rng.choice(group.well, min(len(group), limit//4), False))
    frame=frame[frame.well.isin(selected)]

    store={}; records=[]
    for i,(well,group) in enumerate(frame.groupby("well",sort=True)):
        hw=pd.read_csv(ROOT/f"data/train/{well}__horizontal_well.csv")
        base,paths,labels=candidate_paths(group,oof)
        variants={}
        for smooth in (0, .5, 1, 2, 4):
            variants[smooth]=self_scores(hw,group,paths,smooth)
        store[well]=(group.target.to_numpy(float),base,paths,labels,variants)
        records.append({"well":well,"baseline_rmse":rmse(group.target,base),
                        **{f"max_support_s{s:g}":float(v[1].max()) for s,v in variants.items()}})
        if (i+1)%20==0: print(f"wells={i+1}",flush=True)

    configs=[]
    for smooth in (0,.5,1,2,4):
      for minimum_support in (.03,.05,.1,.2,.3):
       for prior in (.03,.1,.3,.6):
        for temperature in (.1,.2,.4,.8):
            yy=[];bb=[];posterior=[]; used=0
            for y,base,paths,labels,variants in store.values():
                evidence,support=variants[smooth]
                complexity=np.asarray([(s/20)**2+(q/12)**2+(c/6)**2 for _,s,q,c in labels])
                eligible=support >= minimum_support
                if not eligible.any(): pred=base
                else:
                    cost=-evidence+prior*complexity
                    cost[~eligible]=np.inf
                    finite=np.isfinite(cost)
                    scale=np.std(cost[finite])*temperature+1e-6
                    weight=np.zeros(len(cost)); weight[finite]=np.exp(np.clip(-(cost[finite]-cost[finite].min())/scale,-30,0)); weight/=weight.sum()
                    pred=weighted_path_fast(weight,paths,labels); used+=1
                yy.append(y);bb.append(base);posterior.append(pred)
            ya,ba,pa=map(np.concatenate,(yy,bb,posterior))
            for blend in (.1,.2,.3,.4,.5,.7,1):
                configs.append({"smooth":smooth,"minimum_support":minimum_support,
                                "prior":prior,"temperature":temperature,"blend":blend,
                                "used_wells":used,"rmse":rmse(ya,(1-blend)*ba+blend*pa)})
    grid=pd.DataFrame(configs).sort_values("rmse")
    yall=np.concatenate([v[0] for v in store.values()]);ball=np.concatenate([v[1] for v in store.values()])
    summary={"wells":len(store),"rows":len(yall),"baseline":rmse(yall,ball),
             "best":grid.iloc[0].to_dict(),"truth_used_only_for_final_scoring":True}
    grid.to_csv(OUT/"grid.csv",index=False);pd.DataFrame(records).to_csv(OUT/"wells.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__": main(int(sys.argv[1]) if len(sys.argv)>1 else 200)
