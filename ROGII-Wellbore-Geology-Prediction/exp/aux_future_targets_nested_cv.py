"""Leakage-safe auxiliary-future-target experiment.

The auxiliary models learn legal, future-looking summaries (future residual
means, toe bias, and future structural-coordinate changes).  For every outer
fold, auxiliary features for the outer-training wells are produced by an inner
GroupKFold, while outer-validation features are predictions from models fitted
only on outer-training wells.  Thus the final residual model never sees an
auxiliary prediction fitted on the target well.
"""
from pathlib import Path
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/aux_future_targets_nested"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "exp"))
from continuous_candidate_combiner_v1 import make_frame, well_report, pooled

warnings.filterwarnings("ignore")


def future_mean(x, horizon):
    """Mean of the next horizon rows, including current, in O(n)."""
    x = np.asarray(x, float)
    cs = np.r_[0., np.cumsum(x)]
    i = np.arange(len(x))
    j = np.minimum(len(x), i + horizon)
    return (cs[j] - cs[i]) / np.maximum(1, j-i)


def add_aux_targets(frame):
    pieces = []
    for _, g in frame.groupby("well", sort=False):
        g = g.copy()
        resid = g.target.to_numpy(float) - g.base.to_numpy(float)
        # Exact point-horizon residual states target future curve shape rather
        # than the previously rejected future averages.
        for h in (64, 256, 1024):
            j = np.minimum(np.arange(len(g)) + h, len(g)-1)
            g[f"aux_y_respoint_{h}"] = resid[j]
        # Structural U = TVT + Z. src_d_z is the row-to-row Z change, so
        # cumulative Z is sufficient; the arbitrary datum cancels in changes.
        zrel = np.cumsum(g.src_d_z.to_numpy(float))
        u = g.target.to_numpy(float) + zrel
        for h in (64, 256, 1024):
            j = np.minimum(np.arange(len(g)) + h, len(g)-1)
            g[f"aux_y_udelta_{h}"] = u[j] - u
        j1 = np.minimum(np.arange(len(g))+256, len(g)-1)
        j2 = np.minimum(np.arange(len(g))+1024, len(g)-1)
        g["aux_y_future_curvature"] = (u[j2]-u)/(j2-np.arange(len(g))+1) - (u[j1]-u)/(j1-np.arange(len(g))+1)
        pieces.append(g)
    return pd.concat(pieces).sort_index()


def sample_rows(indices, groups, rng, cap=900):
    q = pd.DataFrame({"i": indices, "w": groups[indices]})
    out = []
    for _, z in q.groupby("w"):
        if len(z) > cap:
            out.extend(rng.choice(z.i.to_numpy(), cap, replace=False))
        else:
            out.extend(z.i.to_numpy())
    return np.asarray(out, int)


def model(seed, aux=False):
    return LGBMRegressor(
        objective="huber", n_estimators=260 if aux else 480,
        learning_rate=.035 if aux else .025,
        num_leaves=20 if aux else 28, max_depth=7,
        min_child_samples=140, max_bin=127, reg_alpha=2.,
        reg_lambda=18., colsample_bytree=.70, subsample=.85,
        subsample_freq=1, verbosity=-1, random_state=seed,
        n_jobs=max(1, os.cpu_count()//3))


def main(limit=120, outer_folds=5, inner_folds=3):
    raw = pd.read_pickle(ROOT / "r_v4b/train_feats.pkl")
    oo = joblib.load(ROOT / "r_v4b/stack_v4_oofs.joblib")["oofs"]
    rng = np.random.RandomState(260730)
    wells = np.array(sorted(raw.well.unique()))
    if limit:
        wells = np.sort(rng.choice(wells, min(limit, len(wells)), False))
        raw = raw[raw.well.isin(wells)]
    print("building frame", len(wells), flush=True)
    f = add_aux_targets(make_frame(raw, oo))
    aux_y = [c for c in f if c.startswith("aux_y_")]
    base_cols = [c for c in f if c not in ("well", "target") and
                 not c.startswith("aux_y_")]
    X0 = f[base_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    y = f.target.to_numpy(float)
    base = f.base.to_numpy(float)
    residual = y-base
    groups = f.well.to_numpy()
    pred_plain = np.zeros(len(f))
    pred_aux = np.zeros(len(f))
    aux_oof_all = np.full((len(f), len(aux_y)), np.nan, np.float32)
    outer = GroupKFold(min(outer_folds, len(wells)))

    for fold, (tr, va) in enumerate(outer.split(X0, groups=groups)):
        print("outer", fold, "train/valid", len(tr), len(va), flush=True)
        tr_aux = np.zeros((len(tr), len(aux_y)), np.float32)
        va_aux = np.zeros((len(va), len(aux_y)), np.float32)
        # inner OOF auxiliary features for outer-training rows
        inner = GroupKFold(min(inner_folds, len(np.unique(groups[tr]))))
        for ai, target in enumerate(aux_y):
            yy = f[target].to_numpy(float)
            for ik, (itr0, iva0) in enumerate(inner.split(
                    X0.iloc[tr], groups=groups[tr])):
                itr, iva = tr[itr0], tr[iva0]
                fitidx = sample_rows(itr, groups, rng)
                m = model(10000+fold*100+ai*10+ik, aux=True)
                m.fit(X0.iloc[fitidx], yy[fitidx])
                tr_aux[iva0, ai] = m.predict(X0.iloc[iva])
            # Refit on all outer train to predict outer validation.
            fitidx = sample_rows(tr, groups, rng)
            m = model(20000+fold*100+ai, aux=True)
            m.fit(X0.iloc[fitidx], yy[fitidx])
            va_aux[:, ai] = m.predict(X0.iloc[va])
        aux_oof_all[tr, :] = np.where(
            np.isnan(aux_oof_all[tr, :]), tr_aux, aux_oof_all[tr, :])

        fitidx = sample_rows(tr, groups, rng, cap=1400)
        plain = model(30000+fold)
        plain.fit(X0.iloc[fitidx], residual[fitidx])
        pred_plain[va] = plain.predict(X0.iloc[va])

        Xtr = np.c_[X0.iloc[tr].to_numpy(), tr_aux]
        Xva = np.c_[X0.iloc[va].to_numpy(), va_aux]
        # sample_rows returns global indices; map them into tr positions.
        pos = {v: i for i, v in enumerate(tr)}
        local_fit = np.array([pos[i] for i in fitidx], int)
        final = model(40000+fold)
        final.fit(Xtr[local_fit], residual[fitidx])
        pred_aux[va] = final.predict(Xva)
        print(" scores", pooled(y[va], base[va]),
              pooled(y[va], base[va]+pred_plain[va]),
              pooled(y[va], base[va]+pred_aux[va]), flush=True)

    rows = []
    variants = [("plain", pred_plain), ("aux", pred_aux)]
    for a in (.25, .5, .75):
        variants.append((f"mix_aux_{a}", (1-a)*pred_plain+a*pred_aux))
    for name, pr in variants:
        for blend in (.1, .2, .3, .4, .5, .65, .8, 1.):
            for clip in (2., 4., 6., 10.):
                p = base + blend*np.clip(pr, -clip, clip)
                wr = well_report(f, p, base)
                worst = wr.sort_values("base_rmse", ascending=False).head(
                    max(1, int(np.ceil(.1*len(wr)))))
                rows.append(dict(
                    model=name, blend=blend, clip=clip, rmse=pooled(y, p),
                    gain=pooled(y, base)-pooled(y, p),
                    win_rate=float((wr.new_rmse < wr.base_rmse).mean()),
                    p90=float(wr.new_rmse.quantile(.9)),
                    worst=float(np.sqrt(np.mean(worst.new_rmse**2))),
                    worst_base=float(np.sqrt(np.mean(worst.base_rmse**2)))))
    grid = pd.DataFrame(rows).sort_values("rmse")
    plain_best = grid[grid.model == "plain"].iloc[0].to_dict()
    aux_best = grid[grid.model == "aux"].iloc[0].to_dict()
    result = dict(wells=len(wells), rows=len(f), baseline=pooled(y, base),
                  plain_best=plain_best, aux_best=aux_best,
                  aux_vs_plain=float(plain_best["rmse"]-aux_best["rmse"]),
                  targets=aux_y)
    print(json.dumps(result, indent=2), flush=True)
    tag = "point_pilot" if limit else "point_full"
    grid.to_csv(OUT/f"{tag}_grid.csv", index=False)
    (OUT/f"{tag}_summary.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(OUT/f"{tag}_oof.npz", y=y, base=base,
                        pred_plain=pred_plain, pred_aux=pred_aux,
                        groups=groups)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
