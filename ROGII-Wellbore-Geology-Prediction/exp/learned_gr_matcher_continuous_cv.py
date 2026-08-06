"""Continuous GR/typewell matcher features with honest GroupKFold CV.

This deliberately does not classify/select a trajectory.  Around two legal
reference curves (the prefix-anchored structural path and PF path), it turns
multi-offset GR residuals into a soft posterior over continuous TVT correction.
Those posterior summaries and the existing physical paths feed a row-level
LightGBM residual model.  All validation wells are held out whole.
"""
from pathlib import Path
import json, os, sys

import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/learned_gr_matcher_continuous"
OUT.mkdir(parents=True, exist_ok=True)
OFF = np.array([-40, -20, -10, -5, 0, 5, 10, 20, 40], float)


def posterior_features(d, prefix):
    """Convert GR residuals sampled across TVT offsets to continuous features."""
    R = d[[f"{prefix}{int(o)}" for o in OFF]].to_numpy(np.float32)
    # Robust scale per row makes the posterior about relative matching evidence,
    # not raw GR amplitude. Smooth cost across adjacent offsets to resist aliases.
    scale = np.nanmedian(np.abs(R), axis=1, keepdims=True) + 6.0
    C = (R / scale) ** 2
    C[:, 1:-1] = .25*C[:, :-2] + .5*C[:, 1:-1] + .25*C[:, 2:]
    logits = -C / .55
    logits -= np.nanmax(logits, axis=1, keepdims=True)
    P = np.exp(np.clip(logits, -40, 0))
    P /= np.nansum(P, axis=1, keepdims=True) + 1e-9
    mean = P @ OFF
    var = np.sum(P * (OFF[None, :] - mean[:, None])**2, axis=1)
    order = np.sort(C, axis=1)
    ent = -np.sum(P*np.log(P+1e-9), axis=1)
    return {
        f"{prefix}_post_mean": mean,
        f"{prefix}_post_sd": np.sqrt(var),
        f"{prefix}_entropy": ent,
        f"{prefix}_margin": order[:, 1]-order[:, 0],
        f"{prefix}_mincost": order[:, 0],
        f"{prefix}_p0": P[:, 4],
        f"{prefix}_posmass": P[:, 5:].sum(1),
        f"{prefix}_negmass": P[:, :4].sum(1),
    }


def make_frame(d, strong_oof):
    z = pd.DataFrame(index=d.index)
    for p in ("tda", "tdpf"):
        for k, v in posterior_features(d, p).items():
            z[k] = v
    keep = [
        "pf_d", "pf3_d", "beam_d", "pf_vs_beam", "sp_mean_d", "sp_std",
        "sp_dmin", "ncc8_d", "ncc8_s", "ncc15_d", "ncc15_s",
        "ncc25_d", "ncc25_s", "d_md", "d_z", "d_xy", "dz_dmd",
        "gr", "gr_m21", "gr_s21", "frac", "slp_all", "slp_50",
        "ktvt_rng", "ktvt_std", "pfx_gr_rmse",
    ]
    for c in keep:
        z[c] = d[c].to_numpy()
    # Existing honest OOF LightGBM is the strong anchor. The new matcher is
    # evaluated only by its incremental correction to that OOF prediction.
    z["base"] = np.asarray(strong_oof)[d.index]
    z["matcher_consensus"] = (
        .55*z["tdpf_post_mean"] + .45*z["tda_post_mean"])
    z["matcher_disagree"] = z["tdpf_post_mean"] - z["tda_post_mean"]
    z["well"] = d.well.to_numpy()
    z["target"] = d.target.to_numpy()
    return z


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def main(limit=180, folds=5):
    d = pd.read_pickle(ROOT / "r_v4b/train_feats.pkl")
    oo = joblib.load(ROOT / "r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb123"]
    rng = np.random.RandomState(256)
    wells = np.array(sorted(d.well.unique()))
    if limit and limit < len(wells):
        wells = rng.choice(wells, limit, replace=False)
        d = d[d.well.isin(wells)].copy()
    f = make_frame(d, oo)
    cols = [c for c in f if c not in ("well", "target", "base")]
    X = f[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = f.target.to_numpy(float)
    base = f.base.to_numpy(float)
    groups = f.well.to_numpy()
    pred_r = np.zeros(len(f), float)
    fold_rows = []
    for fold, (tr, va) in enumerate(GroupKFold(folds).split(X, groups=groups)):
        # Equal-ish well influence while preserving every validation point.
        keep = []
        td = pd.DataFrame({"i": tr, "w": groups[tr]})
        for _, q in td.groupby("w", sort=False):
            keep.extend(q.i if len(q) <= 1000 else rng.choice(q.i, 1000, False))
        tr2 = np.asarray(keep, int)
        model = LGBMRegressor(
            objective="huber", n_estimators=500, learning_rate=.025,
            num_leaves=24, max_depth=7, min_child_samples=140,
            reg_alpha=2, reg_lambda=18, colsample_bytree=.82,
            verbosity=-1, random_state=900+fold,
            n_jobs=max(1, (os.cpu_count() or 4)//2))
        model.fit(X.iloc[tr2], (y-base)[tr2])
        pred_r[va] = model.predict(X.iloc[va])
        row = {"fold": fold, "wells": len(np.unique(groups[va])),
               "base": rmse(y[va], base[va]),
               "raw": rmse(y[va], base[va]+pred_r[va])}
        fold_rows.append(row); print(row, flush=True)
    rows = []
    for blend in (.15, .25, .4, .55, .7, .85, 1.):
        for clip in (2., 4., 6., 10., 16.):
            p = base + blend*np.clip(pred_r, -clip, clip)
            rows.append({"blend": blend, "clip": clip, "rmse": rmse(y, p)})
    grid = pd.DataFrame(rows).sort_values("rmse")
    summary = {"wells": int(f.well.nunique()), "rows": len(f),
               "base_rmse": rmse(y, base), "raw_rmse": rmse(y, base+pred_r),
               "best": grid.iloc[0].to_dict(), "folds": fold_rows}
    print(json.dumps(summary, indent=2), flush=True)
    grid.to_csv(OUT / f"grid_{len(wells)}w.csv", index=False)
    (OUT / f"summary_{len(wells)}w.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 180)
