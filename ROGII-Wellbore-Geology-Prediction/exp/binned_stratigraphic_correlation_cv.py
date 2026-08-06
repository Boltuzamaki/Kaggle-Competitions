"""Wu et al.-style complete-path scoring in stratigraphic coordinates.

Each candidate curve projects the observed lateral GR into TVT bins.  We then
aggregate repeated visits to a stratigraphic interval and compare that curve to
the typewell with Pearson/Spearman correlation and a Fisher transform.  TVT
truth is touched only after every candidate score has been computed.
"""
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/binned_stratigraphic_correlation"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "exp"))
from whole_well_gr_warp_selector_cv import candidate_paths, rmse

BIN_WIDTHS = (0.5, 1.0, 2.0, 4.0)


def percentile_normalize(x):
    x = np.asarray(x, float)
    lo, hi = np.nanpercentile(x, (1, 99))
    return np.clip((x-lo)/max(hi-lo, 1e-6), 0, 1)


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a -= a.mean(); b -= b.mean()
    return float(np.dot(a, b) /
                 max(np.sqrt(np.dot(a, a)*np.dot(b, b)), 1e-9))


def path_scores(hw, tw, rows, paths):
    station = rows.id.astype(str).str.rsplit("_", n=1).str[1].astype(int).to_numpy()
    observed = percentile_normalize(hw.GR.to_numpy(float))[station]
    valid_gr = np.isfinite(observed)
    last = float(rows.last_known_tvt.iloc[0])
    tt = tw.TVT.to_numpy(float)
    tg_raw = tw.GR.interpolate(limit_direction="both").fillna(tw.GR.median()).to_numpy(float)
    tg = percentile_normalize(tg_raw)
    scores = {f"pearson_{width:g}": np.zeros(len(paths)) for width in BIN_WIDTHS}
    scores.update({f"spearman_{width:g}": np.zeros(len(paths)) for width in BIN_WIDTHS})
    supports = {width: np.zeros(len(paths)) for width in BIN_WIDTHS}
    for j, path in enumerate(paths):
        tvt = last + path
        for width in BIN_WIDTHS:
            use = valid_gr & np.isfinite(tvt)
            if use.sum() < 20:
                scores[f"pearson_{width:g}"][j] = -1
                scores[f"spearman_{width:g}"][j] = -1
                continue
            origin = np.floor(np.min(tvt[use])/width)*width
            bins = np.floor((tvt[use]-origin)/width).astype(int)
            count = np.bincount(bins)
            total = np.bincount(bins, weights=observed[use])
            occupied = np.flatnonzero(count >= 2)
            if len(occupied) < 8:
                occupied = np.flatnonzero(count >= 1)
            lateral = total[occupied]/count[occupied]
            centers = origin+(occupied+.5)*width
            reference = np.interp(centers, tt, tg)
            p = np.clip(corr(lateral, reference), -.999, .999)
            s = np.clip(corr(rankdata(lateral), rankdata(reference)), -.999, .999)
            # Fisher z makes high correlations more separable for optimization.
            scores[f"pearson_{width:g}"][j] = np.arctanh(p)
            scores[f"spearman_{width:g}"][j] = np.arctanh(s)
            supports[width][j] = len(occupied)
    return scores, supports


def main(limit=200):
    frame = pd.read_pickle(ROOT / "r_v4b/train_feats.pkl")
    oof = joblib.load(ROOT / "r_v4b/stack_v4_oofs.joblib")["oofs"]
    stats = []
    for well, group in frame.groupby("well"):
        base, _, _ = candidate_paths(group, oof)
        stats.append((well, rmse(group.target, base)))
    stats = pd.DataFrame(stats, columns=["well", "base_rmse"])
    stats["bin"] = pd.qcut(stats.base_rmse, 4, labels=False)
    rng = np.random.RandomState(802)
    selected = []
    for _, group in stats.groupby("bin"):
        selected.extend(rng.choice(group.well, min(len(group), limit//4), False))
    frame = frame[frame.well.isin(selected)]

    records = []
    store = {}
    for i, (well, group) in enumerate(frame.groupby("well", sort=True)):
        hw = pd.read_csv(ROOT / f"data/train/{well}__horizontal_well.csv")
        tw = pd.read_csv(ROOT / f"data/train/{well}__typewell.csv").sort_values("TVT")
        base, paths, labels = candidate_paths(group, oof)
        scores, supports = path_scores(hw, tw, group, paths)
        y = group.target.to_numpy(float)
        losses = np.sqrt(np.mean((paths-y[None])**2, axis=1))
        store[well] = (y, base, paths, labels, scores)
        for j, label in enumerate(labels):
            row = {"well": well, "candidate": j, "loss": losses[j]}
            row.update({key: value[j] for key, value in scores.items()})
            row.update({f"support_{width:g}": supports[width][j]
                        for width in BIN_WIDTHS})
            records.append(row)
        if (i+1) % 20 == 0:
            print(f"features={i+1}", flush=True)

    configs = []
    metric_sets = []
    for width in BIN_WIDTHS:
        metric_sets += [
            (f"pearson_{width:g}",), (f"spearman_{width:g}",),
            (f"pearson_{width:g}", f"spearman_{width:g}")]
    metric_sets += [tuple(f"pearson_{w:g}" for w in BIN_WIDTHS),
                    tuple(f"spearman_{w:g}" for w in BIN_WIDTHS),
                    tuple(k for w in BIN_WIDTHS
                          for k in (f"pearson_{w:g}", f"spearman_{w:g}"))]
    for metrics in metric_sets:
        for prior in (0, .01, .03, .1, .3):
            for temperature in (0, .25, .5, 1.0):
                yy, bb, pp = [], [], []
                for well, (y, base, paths, labels, scores) in store.items():
                    evidence = np.mean([scores[k] for k in metrics], axis=0)
                    complexity = np.asarray([
                        (shift/20)**2+(slope/12)**2+(curve/6)**2
                        for _, shift, slope, curve in labels])
                    cost = -evidence + prior*complexity
                    if temperature == 0:
                        pred = paths[np.argmin(cost)]
                    else:
                        scale = np.std(cost)*temperature+1e-6
                        weight = np.exp(np.clip(-(cost-cost.min())/scale, -30, 0))
                        weight /= weight.sum(); pred = weight@paths
                    yy.append(y); bb.append(base); pp.append(pred)
                yall, baseall, predall = map(np.concatenate, (yy, bb, pp))
                for blend in (0, .1, .2, .3, .5, .7, 1):
                    configs.append({"metrics": "+".join(metrics),
                                    "prior": prior, "temperature": temperature,
                                    "blend": blend,
                                    "rmse": rmse(yall, (1-blend)*baseall+blend*predall)})
    grid = pd.DataFrame(configs).sort_values("rmse")
    table = pd.DataFrame(records)
    oracle = []
    for well, (y, _, paths, _, _) in store.items():
        oracle.append(paths[np.argmin(np.mean((paths-y[None])**2, axis=1))])
    yall = np.concatenate([v[0] for v in store.values()])
    baseall = np.concatenate([v[1] for v in store.values()])
    summary = {"wells": len(store), "rows": len(yall),
               "baseline": rmse(yall, baseall),
               "oracle": rmse(yall, np.concatenate(oracle)),
               "best": grid.iloc[0].to_dict(),
               "bin_widths": BIN_WIDTHS,
               "truth_used_only_for_final_scoring": True}
    table.to_parquet(OUT / "candidate_scores.parquet", index=False)
    grid.to_csv(OUT / "grid.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
