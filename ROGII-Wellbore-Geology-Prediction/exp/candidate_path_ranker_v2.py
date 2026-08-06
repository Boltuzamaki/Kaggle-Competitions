"""Whole-well candidate-path ranker on honest Stack-V4 OOF predictions.

The ranker never sees rows from a held-out well.  Each example is an entire
candidate TVT continuation for one well; labels are candidate whole-well RMSE.
Candidate features use only predictions/trajectory/log features (never TVT).
"""
from pathlib import Path
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/candidate_ranker_v2"
OUT.mkdir(parents=True, exist_ok=True)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def smooth(x, sigma):
    if len(x) < 5:
        return x.copy()
    return gaussian_filter1d(x.astype(float), sigma=min(sigma, len(x) / 8),
                             mode="nearest")


def poly_path(x, p, degree):
    """Robust-ish complete-path structural projection, prediction-only."""
    if len(x) < 20:
        return p.copy()
    z = (x - x.mean()) / (x.std() + 1e-6)
    keep = np.ones(len(x), bool)
    coef = None
    for _ in range(4):
        coef = np.polyfit(z[keep], p[keep], degree)
        fit = np.polyval(coef, z)
        r = p - fit
        scale = np.median(np.abs(r[keep])) * 1.4826 + 1e-5
        keep = np.abs(r) < 2.5 * scale
        if keep.sum() < degree + 5:
            break
    return np.polyval(coef, z)


def make_candidates(g, oo):
    idx = g.index.to_numpy()
    bases = {k: np.asarray(v)[idx].astype(float) for k, v in oo.items()}
    ens = .55 * bases["lgb123"] + .20 * bases["lgb7"] + \
          .15 * bases["xgb"] + .10 * bases["cat"]
    mean4 = np.mean(np.stack(list(bases.values())), axis=0)
    x = g.d_md.to_numpy(float)
    hold = np.zeros(len(g))
    pf = g.pf_d.to_numpy(float)
    beam = g.beam_d.to_numpy(float)
    c = {
        "lgb123": bases["lgb123"], "lgb7": bases["lgb7"],
        "xgb": bases["xgb"], "cat": bases["cat"],
        "weighted": ens, "mean4": mean4,
        "weighted_s4": .75 * ens + .25 * smooth(ens, 4),
        "weighted_s12": .50 * ens + .50 * smooth(ens, 12),
        "weighted_poly2_25": .75 * ens + .25 * poly_path(x, ens, 2),
        "weighted_poly3_25": .75 * ens + .25 * poly_path(x, ens, 3),
        "weighted_poly3_50": .50 * ens + .50 * poly_path(x, ens, 3),
        "pf": pf, "pfbeam": .80 * pf + .20 * beam,
        "hold10": .90 * ens + .10 * hold,
    }
    # Translation families let the selector correct systematic whole-path datum.
    for off in (-3., -1.5, 1.5, 3.):
        c[f"weighted_off{off:+g}"] = ens + off
    return c


def cand_features(g, pred, allpred, cand_id):
    x = g.d_md.to_numpy(float)
    gr = g.gr.to_numpy(float)
    dz = g.d_z.to_numpy(float)
    disp = np.std(np.stack(list(allpred.values())), axis=0)
    grad = np.gradient(pred)
    curv = np.gradient(grad)
    def corr(a, b):
        if len(a) < 3 or np.std(a) < 1e-7 or np.std(b) < 1e-7:
            return 0.
        return float(np.corrcoef(a, b)[0, 1])
    q = lambda a, z: float(np.quantile(a, z))
    feat = {
        "n": len(g), "xspan": np.ptp(x),
        "zspan": np.ptp(dz), "gr_std": np.std(gr),
        "gr_rough": np.std(np.gradient(gr)),
        "pred_mean": np.mean(pred), "pred_std": np.std(pred),
        "pred_span": np.ptp(pred), "pred_end": pred[-1],
        "pred_slope": np.polyfit(x, pred, 1)[0] if np.ptp(x) > 1 else 0.,
        "rough1": np.sqrt(np.mean(grad ** 2)),
        "rough2": np.sqrt(np.mean(curv ** 2)),
        "disp_mean": np.mean(disp), "disp_p90": q(disp, .9),
        "corr_gr": corr(pred, gr), "corr_dz": corr(pred, dz),
        "pf_gap": np.sqrt(np.mean((pred-g.pf_d.to_numpy())**2)),
        "beam_gap": np.sqrt(np.mean((pred-g.beam_d.to_numpy())**2)),
        "ncc8_mean": np.mean(g.ncc8_s), "ncc15_mean": np.mean(g.ncc15_s),
        "pfx_gr_rmse": np.mean(g.pfx_gr_rmse),
        "sp_std": np.mean(g.sp_std), "sp_dmin": np.mean(g.sp_dmin),
        "frac_end": np.max(g.frac),
    }
    # Model identity is a legitimate fixed candidate descriptor.
    feat.update({f"is_{k}": float(cand_id == k) for k in allpred})
    return feat


def main(limit=0, folds=5):
    d = pd.read_pickle(ROOT / "r_v4b/train_feats.pkl").reset_index(drop=True)
    pack = joblib.load(ROOT / "r_v4b/stack_v4_oofs.joblib")
    oo = pack["oofs"]
    wells = sorted(d.well.unique())
    rng = np.random.RandomState(20260730)
    if limit:
        wells = list(rng.choice(wells, min(limit, len(wells)), replace=False))
        d = d[d.well.isin(wells)].copy()
        # Preserve original row indices for OOF arrays.
    rows, paths, truths = [], {}, {}
    for w, g in d.groupby("well", sort=True):
        cands = make_candidates(g, oo)
        y = g.target.to_numpy(float)
        truths[w] = y
        paths[w] = cands
        for cid, pred in cands.items():
            f = cand_features(g, pred, cands, cid)
            f.update(well=w, candidate=cid, label=rmse(y, pred))
            rows.append(f)
    tab = pd.DataFrame(rows)
    # Predict within-well regret, not absolute RMSE.  Absolute RMSE is dominated
    # by intrinsic well difficulty, which is common to every candidate and
    # obscures the much smaller path-choice signal.
    base_loss = (tab[tab.candidate == "weighted"]
                 .set_index("well").label.to_dict())
    tab["rank_label"] = tab.label - tab.well.map(base_loss)
    names = [c for c in tab if c not in ("well", "candidate", "label")]
    X = tab[names].replace([np.inf, -np.inf], np.nan).fillna(0)
    groups = tab.well
    chosen, oracle, baseline = {}, {}, {}
    names.remove("rank_label")
    X = tab[names].replace([np.inf, -np.inf], np.nan).fillna(0)
    gkf = GroupKFold(n_splits=folds)
    for fold, (tr, va) in enumerate(gkf.split(X, tab.rank_label, groups)):
        model = LGBMRegressor(n_estimators=500, learning_rate=.025,
                              num_leaves=15, min_child_samples=30,
                              subsample=.8, colsample_bytree=.8,
                              reg_lambda=5., verbosity=-1,
                              random_state=400+fold, n_jobs=max(1, os.cpu_count()//2))
        model.fit(X.iloc[tr], tab.rank_label.iloc[tr])
        z = tab.iloc[va][["well", "candidate", "label"]].copy()
        z["score"] = model.predict(X.iloc[va])
        for w, q in z.groupby("well"):
            chosen[w] = q.loc[q.score.idxmin(), "candidate"]
            oracle[w] = q.loc[q.label.idxmin(), "candidate"]
            baseline[w] = "weighted"
    def pooled(selection):
        yy, pp = [], []
        for w, cid in selection.items():
            yy.append(truths[w]); pp.append(paths[w][cid])
        return rmse(np.concatenate(yy), np.concatenate(pp))
    result = {
        "wells": len(truths), "rows": int(sum(map(len, truths.values()))),
        "candidates": len(next(iter(paths.values()))),
        "baseline_weighted": pooled(baseline),
        "ranker": pooled(chosen), "oracle": pooled(oracle),
    }
    result["gain"] = result["baseline_weighted"] - result["ranker"]
    print(json.dumps(result, indent=2))
    pd.DataFrame({"well": list(chosen), "chosen": list(chosen.values()),
                  "oracle": [oracle[w] for w in chosen]}).to_csv(
                      OUT / ("pilot_choices.csv" if limit else "full_choices.csv"),
                      index=False)
    (OUT / ("pilot_summary.json" if limit else "summary.json")).write_text(
        json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(lim)
