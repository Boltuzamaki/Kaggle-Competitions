"""Continuous row-level correction over candidate paths, honest grouped OOF."""
from pathlib import Path
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold
from pandas.errors import PerformanceWarning

warnings.simplefilter("ignore", PerformanceWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/continuous_candidate_combiner_v1"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "exp"))
from candidate_path_ranker_v2 import make_candidates, rmse  # noqa


def make_frame(d, oo):
    pieces = []
    for w, g in d.groupby("well", sort=True):
        c = make_candidates(g, oo)
        z = pd.DataFrame(index=g.index)
        # All paths and their consensus/confidence are inference-available.
        for k, v in c.items():
            z[f"cand_{k}"] = v
        a = np.stack(list(c.values()))
        z["cand_median"] = np.median(a, axis=0)
        z["cand_std"] = np.std(a, axis=0)
        z["cand_range"] = np.ptp(a, axis=0)
        z["cand_iqr"] = np.quantile(a, .75, axis=0)-np.quantile(a, .25, axis=0)
        base = c["weighted"]
        z["base"] = base
        z["base_grad"] = np.gradient(base)
        z["base_curv"] = np.gradient(np.gradient(base))
        for sig in (2, 5, 12, 25):
            z[f"base_s{sig}"] = gaussian_filter1d(
                base, min(sig, max(1, len(base)/8)), mode="nearest")
            z[f"gr_s{sig}"] = gaussian_filter1d(
                g.gr.to_numpy(float), min(sig, max(1, len(g)/8)),
                mode="nearest")
        # Existing features are created by the same train/test pipeline.
        keep = [
            "d_md", "d_z", "d_xy", "dz_dmd", "gr", "gr_m21", "gr_s21",
            "frac", "slp_all", "slp_50", "ktvt_rng", "ktvt_std",
            "pfx_gr_rmse", "pf_vs_beam", "ncc8_s", "ncc15_s", "ncc25_s",
            "sp_mean_d", "sp_std", "sp_dmin", "pf_d", "pf3_d", "beam_d",
            "ncc8_d", "ncc15_d", "ncc25_d",
        ] + [x for x in g if x.startswith("tda") or x.startswith("tdpf")]
        for k in keep:
            z[f"src_{k}"] = g[k].to_numpy()
        # Candidate-to-evidence gaps are row-local likelihood surrogates.
        for k in ("pf_d", "beam_d", "ncc8_d", "ncc15_d", "ncc25_d",
                  "sp_mean_d"):
            z[f"base_gap_{k}"] = base - g[k].to_numpy(float)
        # Complete-well context: systematic datum/shape errors are invisible to
        # a purely local row model. All statistics use the full observed
        # trajectory/log, which is available for test wells.
        context_cols = [
            "gr", "dz_dmd", "pfx_gr_rmse", "ncc8_s", "ncc15_s",
            "ncc25_s", "sp_std", "sp_dmin", "pf_vs_beam",
        ]
        for k in context_cols:
            v = g[k].to_numpy(float)
            for name, val in (
                ("mean", np.nanmean(v)), ("std", np.nanstd(v)),
                ("p10", np.nanquantile(v, .1)), ("p90", np.nanquantile(v, .9)),
            ):
                z[f"ctx_{k}_{name}"] = val
        for k, v in c.items():
            z[f"ctx_{k}_mean"] = np.mean(v)
            z[f"ctx_{k}_std"] = np.std(v)
            z[f"ctx_{k}_end"] = v[-1]
            z[f"ctx_{k}_slope"] = (
                np.polyfit(g.d_md.to_numpy(float), v, 1)[0]
                if np.ptp(g.d_md.to_numpy(float)) > 1 else 0.)
        z["well"] = w
        z["target"] = g.target.to_numpy(float)
        num = z.select_dtypes(include=[np.number]).columns
        z[num] = z[num].astype(np.float32)
        pieces.append(z)
    return pd.concat(pieces).sort_index()


def well_report(frame, pred, base):
    q = frame[["well", "target"]].copy()
    q["pred"] = pred
    q["base"] = base
    rows = []
    for w, g in q.groupby("well"):
        rows.append((w, len(g), rmse(g.target, g.base), rmse(g.target, g.pred)))
    return pd.DataFrame(rows, columns=["well", "rows", "base_rmse", "new_rmse"])


def pooled(y, p):
    return rmse(np.asarray(y), np.asarray(p))


def main(limit=120, folds=5):
    d = pd.read_pickle(ROOT / "r_v4b/train_feats.pkl")
    oo = joblib.load(ROOT / "r_v4b/stack_v4_oofs.joblib")["oofs"]
    rng = np.random.RandomState(20260730)
    wells = sorted(d.well.unique())
    if limit:
        wells = sorted(rng.choice(wells, min(limit, len(wells)),
                                  replace=False).tolist())
        d = d[d.well.isin(wells)]
    f = make_frame(d, oo)
    cols = [c for c in f if c not in ("well", "target")]
    X = f[cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y = f.target.to_numpy(float)
    base = f.base.to_numpy(float)
    # Learn a bounded correction; predicting residual is easier and protects
    # the strong fixed ensemble.
    residual = y - base
    pred_res = np.zeros(len(f))
    groups = f.well.to_numpy()
    for fold, (tr, va) in enumerate(GroupKFold(folds).split(X, groups=groups)):
        # Cap per-well training rows without changing validation metric.
        trdf = pd.DataFrame({"i": tr, "w": groups[tr]})
        sampled = []
        for _, q in trdf.groupby("w"):
            if len(q) > 1800:
                sampled.extend(rng.choice(q.i, 1800, replace=False))
            else:
                sampled.extend(q.i)
        tr2 = np.asarray(sampled, dtype=int)
        m = LGBMRegressor(
            objective="huber", n_estimators=550, learning_rate=.025,
            num_leaves=24, max_depth=7, min_child_samples=120,
            max_bin=127, reg_alpha=2, reg_lambda=15,
            colsample_bytree=.72, subsample=.85, subsample_freq=1,
            verbosity=-1, random_state=1200+fold,
            n_jobs=max(1, os.cpu_count()//2))
        m.fit(X.iloc[tr2], residual[tr2])
        pred_res[va] = m.predict(X.iloc[va])
        print("fold", fold, "base", pooled(y[va], base[va]),
              "raw", pooled(y[va], base[va]+pred_res[va]), flush=True)
    result = {"wells": len(wells), "rows": len(f),
              "baseline": pooled(y, base)}
    report_rows = []
    for blend in (0.1, .2, .3, .4, .5, .65, .8, 1.0):
        # Conservative correction clipping is test-stable.
        for clip in (2., 4., 6., 10., 20.):
            p = base + blend*np.clip(pred_res, -clip, clip)
            wr = well_report(f, p, base)
            worst = wr.sort_values("base_rmse", ascending=False).head(
                max(1, int(np.ceil(.1*len(wr)))))
            report_rows.append({
                "blend": blend, "clip": clip, "rmse": pooled(y, p),
                "gain": pooled(y, base)-pooled(y, p),
                "well_win_rate": float((wr.new_rmse < wr.base_rmse).mean()),
                "p90_new": float(wr.new_rmse.quantile(.9)),
                "p90_base": float(wr.base_rmse.quantile(.9)),
                "worst_new": float(np.sqrt(np.mean(worst.new_rmse**2))),
                "worst_base": float(np.sqrt(np.mean(worst.base_rmse**2))),
            })
    grid = pd.DataFrame(report_rows).sort_values("rmse")
    result["best"] = grid.iloc[0].to_dict()
    result["raw_unclipped"] = pooled(y, base+pred_res)
    print(json.dumps(result, indent=2))
    grid.to_csv(OUT / ("pilot_grid.csv" if limit else "grid.csv"), index=False)
    (OUT / ("pilot_summary.json" if limit else "summary.json")).write_text(
        json.dumps(result, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
