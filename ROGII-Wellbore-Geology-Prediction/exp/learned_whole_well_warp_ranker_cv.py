"""Strict outer-group learned ranker for complete-well warp candidates."""
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"exp/results/learned_whole_well_warp_ranker"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT/"exp"))
from whole_well_gr_warp_selector_cv import (  # noqa
    candidate_paths, robust_calibrate, rmse)


def diagnostics(hw, tw, g, paths, labels, base):
    idx = g.id.astype(str).str.rsplit("_", n=1).str[1].astype(int).to_numpy()
    hgr = hw.GR.to_numpy(float)[idx]
    hgr = np.nan_to_num(hgr, nan=np.nanmedian(hgr))
    last = float(g.last_known_tvt.iloc[0])
    a, b, pfx = robust_calibrate(hw, tw)
    tt, tg = tw.TVT.to_numpy(float), tw.GR.to_numpy(float)
    cg = np.interp((last+paths).ravel(), tt, tg).reshape(paths.shape)*a+b
    hs = 1.4826*np.median(np.abs(hgr-np.median(hgr)))+8
    pf = g.pf_d.to_numpy(float)
    # Candidate-specific entire-sequence evidence.
    evidence = {}
    for sig in (0, 5, 20, 60):
        hh = gaussian_filter1d(hgr, sig) if sig else hgr
        cc = gaussian_filter1d(cg, sig, axis=1) if sig else cg
        r = (cc-hh[None])/hs
        evidence[f"cauchy{sig}"] = np.mean(np.log1p(r*r), 1)
        evidence[f"mae{sig}"] = np.mean(np.abs(r), 1)
        evidence[f"bias{sig}"] = np.mean(r, 1)
        # Correlation computed vectorially.
        h0 = hh-hh.mean()
        c0 = cc-cc.mean(1, keepdims=True)
        evidence[f"corr{sig}"] = np.sum(c0*h0, 1)/(
            np.sqrt(np.sum(c0*c0, 1)*np.sum(h0*h0))+1e-7)
    dh = gaussian_filter1d(np.gradient(hgr), 5)
    dc = gaussian_filter1d(np.gradient(cg, axis=1), 5, axis=1)
    evidence["grad_cost"] = np.mean(np.log1p(
        ((dc-dh[None])/(hs/5+1))**2), 1)
    aggregate = (.30*evidence["cauchy0"]+.25*evidence["cauchy5"]+
                 .25*evidence["cauchy20"]+.20*evidence["cauchy60"]+
                 .12*evidence["grad_cost"])
    order = aggregate.argsort().argsort()/max(1, len(aggregate)-1)

    dx = hw.X.iloc[-1]-hw.X.iloc[0]; dy = hw.Y.iloc[-1]-hw.Y.iloc[0]
    az = np.arctan2(dy, dx)
    common = {
        "n": len(g), "zspan": float(np.ptp(g.d_z)),
        "xyspan": float(np.ptp(g.d_xy)), "gr_std": float(np.std(hgr)),
        "gr_rough": float(np.std(np.gradient(hgr))),
        "prefix_fit_rmse": pfx, "cal_a": a,
        "typewell_tvt_span": float(np.ptp(tt)),
        "typewell_gr_std": float(np.std(tg)),
        "az_sin": float(np.sin(az)), "az_cos": float(np.cos(az)),
        "pf_base_gap": rmse(pf, base),
        "pf_beam_gap": float(np.mean(np.abs(g.pf_vs_beam))),
        "sp_std": float(np.mean(g.sp_std)),
        "sp_dmin": float(np.mean(g.sp_dmin)),
        "ncc8": float(np.mean(g.ncc8_s)),
        "ncc15": float(np.mean(g.ncc15_s)),
        "ncc25": float(np.mean(g.ncc25_s)),
        "supertype": float(g.supertype.iloc[0]),
    }
    rows = []
    y = g.target.to_numpy(float)
    base_loss = rmse(y, base)
    for j, (ci, shift, slope, curve) in enumerate(labels):
        p = paths[j]
        q = dict(common)
        q.update({"center": ci, "shift": shift, "slope": slope,
                  "curve": curve, "abs_shift": abs(shift),
                  "abs_slope": abs(slope), "abs_curve": abs(curve),
                  "cost": aggregate[j], "cost_rank": order[j],
                  "cost_gap": aggregate[j]-aggregate.min(),
                  "base_path_gap": rmse(p, base), "pf_path_gap": rmse(p, pf),
                  "path_slope": float(np.polyfit(
                      np.linspace(-1, 1, len(p)), p, 1)[0]),
                  "path_end": float(p[-1]), "path_span": float(np.ptp(p)),
                  "loss": rmse(y, p), "regret": rmse(y, p)-base_loss})
        for k, v in evidence.items():
            q[k] = v[j]
        rows.append(q)
    return rows, aggregate


def main(limit=200):
    d = pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
    oof = joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
    # Same difficulty-stratified well selection as the oracle experiment.
    stats = []
    for w, g in d.groupby("well"):
        ix = g.index.to_numpy()
        b = (.55*np.asarray(oof["lgb123"])[ix]+
             .20*np.asarray(oof["lgb7"])[ix]+
             .15*np.asarray(oof["xgb"])[ix]+.10*np.asarray(oof["cat"])[ix])
        stats.append((w, rmse(g.target, b)))
    st = pd.DataFrame(stats, columns=["well", "base_rmse"])
    st["bin"] = pd.qcut(st.base_rmse, 4, labels=False)
    rng = np.random.RandomState(802)
    selected = []
    for _, q in st.groupby("bin"):
        selected.extend(rng.choice(q.well, min(len(q), limit//4), False))
    d = d[d.well.isin(selected)]

    table, paths_by, truth_by, base_by = [], {}, {}, {}
    for i, (w, g) in enumerate(d.groupby("well", sort=True)):
        hw = pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv")
        tw = pd.read_csv(ROOT/f"data/train/{w}__typewell.csv").sort_values("TVT")
        base, paths, labels = candidate_paths(g, oof)
        rows, _ = diagnostics(hw, tw, g, paths, labels, base)
        for j, q in enumerate(rows):
            q.update(well=w, candidate=j)
            table.append(q)
        paths_by[w], truth_by[w], base_by[w] = (
            paths, g.target.to_numpy(float), base)
        if (i+1) % 25 == 0:
            print("features", i+1, flush=True)
    tab = pd.DataFrame(table)
    drop = {"well", "candidate", "loss", "regret"}
    cols = [c for c in tab if c not in drop]
    X = tab[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    groups = tab.well.to_numpy()
    pred_cat = np.zeros(len(tab)); pred_et = np.zeros(len(tab))
    fold_rows = []
    for fold, (tr, va) in enumerate(GroupKFold(5).split(X, groups=groups)):
        cat = CatBoostRegressor(
            iterations=900, depth=7, learning_rate=.035,
            loss_function="RMSE", l2_leaf_reg=10, random_seed=820+fold,
            verbose=False, allow_writing_files=False, thread_count=8)
        et = ExtraTreesRegressor(
            n_estimators=500, max_features=.75, min_samples_leaf=4,
            max_depth=16, n_jobs=8, random_state=920+fold)
        cat.fit(X.iloc[tr], tab.regret.iloc[tr])
        et.fit(X.iloc[tr], tab.regret.iloc[tr])
        pred_cat[va] = cat.predict(X.iloc[va])
        pred_et[va] = et.predict(X.iloc[va])
        fold_rows.append({"fold": fold,
                          "candidate_rmse_cat": rmse(tab.regret.iloc[va],
                                                     pred_cat[va]),
                          "candidate_rmse_et": rmse(tab.regret.iloc[va],
                                                    pred_et[va])})
        print(fold_rows[-1], flush=True)

    def evaluate(score, temperature=0):
        yy, bb, pp = [], [], []
        choices = {}
        for w, q in tab.assign(score=score).groupby("well", sort=False):
            y, base, paths = truth_by[w], base_by[w], paths_by[w]
            if temperature:
                s = q.score.to_numpy()
                temp = np.std(s)*temperature+1e-5
                wt = np.exp(np.clip(-(s-s.min())/temp, -30, 0)); wt /= wt.sum()
                p = wt@paths
                choices[w] = int(q.iloc[np.argmin(s)].candidate)
            else:
                j = int(q.iloc[np.argmin(q.score)].candidate)
                p = paths[j]; choices[w] = j
            yy.append(y); bb.append(base); pp.append(p)
        return np.concatenate(yy), np.concatenate(bb), np.concatenate(pp), choices

    result_rows = []
    for name, score in (("cat", pred_cat), ("et", pred_et),
                        ("mean", .5*pred_cat+.5*pred_et)):
        for temp in (0, .25, .5, 1.):
            y, base, p, _ = evaluate(score, temp)
            for blend in (0, .1, .2, .3, .4, .55, .7, .85, 1):
                result_rows.append({"model": name, "temperature": temp,
                                    "blend": blend,
                                    "rmse": rmse(y, (1-blend)*base+blend*p)})
    grid = pd.DataFrame(result_rows).sort_values("rmse")
    y, base, _, _ = evaluate(pred_cat)
    oracle = np.concatenate([
        paths_by[w][np.argmin(np.mean(
            (paths_by[w]-truth_by[w][None])**2, axis=1))]
        for w in sorted(paths_by)])
    # evaluate() concatenates in tab group order, also sorted because table is.
    summary = {"wells": len(paths_by), "rows": len(y),
               "baseline": rmse(y, base), "oracle": rmse(y, oracle),
               "best": grid.iloc[0].to_dict(), "folds": fold_rows}
    print(json.dumps(summary, indent=2), flush=True)
    grid.to_csv(OUT/"pilot_grid.csv", index=False)
    (OUT/"pilot_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 200)
