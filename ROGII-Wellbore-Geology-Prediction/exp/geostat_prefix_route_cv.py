"""Visible-prefix self-routing and residual-dip calibration of geostat surfaces."""
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOUT = ROOT/"exp/results/geostat_marker_surface"
OUT = GOUT/"prefix_routing"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT/"exp"))
from geostat_marker_surface_strict_cv import load_wells, rmse  # noqa


def predict_tvt(d, surf, fit_end, shrink, rows=None):
    """Fit bias+shrunk residual trend using prefix through fit_end."""
    md = d.MD.to_numpy(float)
    structural = d.TVT_input.to_numpy(float)+d.Z.to_numpy(float)
    use = np.arange(fit_end+1)
    # Recent and full robust-ish slopes; clipping avoids catastrophic dip.
    recent = use[max(0, len(use)-600):]
    slopes = []
    for ix in (use, recent):
        if len(ix) > 10 and np.ptp(md[ix]) > 1:
            slopes.append(np.polyfit(md[ix]-md[fit_end],
                                     (structural-surf)[ix], 1)[0])
    slope = float(np.clip(np.median(slopes) if slopes else 0., -.02, .02))
    adj = surf+shrink*slope*(md-md[fit_end])
    b = float(np.median(structural[use]-adj[use]))
    p = adj-d.Z.to_numpy(float)+b
    return p if rows is None else p[rows]


def main(limit=200):
    cached = pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
    cgroups = {w: g for w, g in cached.groupby("well", sort=False)}
    eligible = set(cached.well.astype(str).unique())
    all_items = [s for s in load_wells(0) if s["well"] in eligible]
    rng = np.random.RandomState(805)
    items = all_items if not limit or limit >= len(all_items) else [
        all_items[i] for i in sorted(rng.choice(len(all_items), limit, False))]
    oof = joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
    candidates = {}
    pseudo = {}
    truth, base = [], []
    for s in items:
        w, d, ps = s["well"], s["d"], s["ps"]
        z = np.load(GOUT/f"query_cache/{w}.npz")
        candidates[w], pseudo[w] = {}, {}
        for method in [k for k in z.files if k != "_distance"]:
            surf = z[method]
            for sh in (0., .1, .25, .5):
                key = f"{method}_s{sh:g}"
                errs = []
                for frac in (.5, .7, .85):
                    cut = max(20, int(ps*frac))
                    rows = np.arange(cut+1, ps+1)
                    p = predict_tvt(d, surf, cut, sh, rows)
                    errs.append(np.mean(
                        (d.TVT_input.to_numpy(float)[rows]-p)**2))
                pseudo[w][key] = float(np.sqrt(np.mean(errs)))
                rows = np.arange(ps+1, len(d))
                candidates[w][key] = predict_tvt(d, surf, ps, sh, rows)
        g = cgroups[w]
        ix = g.index.to_numpy()
        b = (.55*np.asarray(oof["lgb123"])[ix]+.20*np.asarray(oof["lgb7"])[ix]+
             .15*np.asarray(oof["xgb"])[ix]+.10*np.asarray(oof["cat"])[ix])
        truth.append(g.target.to_numpy(float)); base.append(b)
    yall, ball = np.concatenate(truth), np.concatenate(base)
    keys = sorted(next(iter(candidates.values())))
    global_rows = []
    global_blends = []
    for key in keys:
        p = np.concatenate([
            candidates[s["well"]][key]-float(
                cgroups[s["well"]].last_known_tvt.iloc[0])
            for s in items])
        global_rows.append({"candidate": key, "rmse": rmse(yall, p)})
        for blend in (0, .025, .05, .075, .1, .125, .15, .2, .3):
            global_blends.append({"candidate": key, "blend": blend,
                                  "rmse": rmse(
                                      yall, (1-blend)*ball+blend*p)})
    global_df = pd.DataFrame(global_rows).sort_values("rmse")
    global_blends = pd.DataFrame(global_blends).sort_values("rmse")
    # Per-well hard and soft routing by pseudo-hidden visible-prefix error.
    hard, soft = [], []
    route_rows = []
    for s in items:
        w = s["well"]; scores = np.array([pseudo[w][k] for k in keys])
        j = int(np.argmin(scores))
        top = np.argsort(scores)[:8]
        temp = np.std(scores[top])+1e-4
        wt = np.exp(np.clip(-(scores[top]-scores[top].min())/temp, -20, 0))
        wt /= wt.sum()
        last = float(cgroups[w].last_known_tvt.iloc[0])
        hard.append(candidates[w][keys[j]]-last)
        soft.append(sum(wt[q]*candidates[w][keys[k]] for q, k in enumerate(top))-last)
        route_rows.append({"well": w, "selected": keys[j],
                           "pseudo_rmse": scores[j]})
    hard, soft = np.concatenate(hard), np.concatenate(soft)
    grid = []
    for name, p in (("hard", hard), ("soft", soft)):
        for blend in (0, .05, .1, .15, .2, .3, .4, .55, .7, 1):
            grid.append({"route": name, "blend": blend,
                         "rmse": rmse(yall, (1-blend)*ball+blend*p)})
    grid = pd.DataFrame(grid).sort_values("rmse")
    result = {"wells": len(items), "rows": len(yall),
              "anchor": rmse(yall, ball),
              "best_global_direct": global_df.iloc[0].to_dict(),
              "best_global_blend": global_blends.iloc[0].to_dict(),
              "hard_route": rmse(yall, hard), "soft_route": rmse(yall, soft),
              "best_route_blend": grid.iloc[0].to_dict()}
    print(json.dumps(result, indent=2), flush=True)
    global_df.to_csv(OUT/"global_candidates.csv", index=False)
    global_blends.to_csv(OUT/"global_blends.csv", index=False)
    grid.to_csv(OUT/"route_blend_grid.csv", index=False)
    pd.DataFrame(route_rows).to_csv(OUT/"routes.csv", index=False)
    (OUT/"summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 200)
