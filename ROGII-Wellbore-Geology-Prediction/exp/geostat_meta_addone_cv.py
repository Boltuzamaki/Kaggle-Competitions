"""Align cached geostat prediction to restricted meta OOF and add-one audit."""
from pathlib import Path
import contextlib
import io
import json
import runpy
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
GOUT = ROOT/"exp/results/geostat_marker_surface"
OUT = GOUT/"meta_addone"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT/"exp"))
from geostat_marker_surface_strict_cv import load_wells, rmse  # noqa
from geostat_prefix_route_cv import predict_tvt  # noqa

with contextlib.redirect_stdout(io.StringIO()):
    state = runpy.run_path(str(ROOT/"exp/meta_all_honest_oof.py"))
legs, y, wells = state["legs"], state["y"], state["wells"]
base_names = ["har_physics", "har_lgb", "har_xgb",
              "pil_blend_oof_postprocessed", "v4_lgb7"]

# Recreate exact cached-query set and candidate, indexed by official ID.
eligible = set(pd.Series(wells).astype(str).unique())
all_items = [s for s in load_wells(0) if s["well"] in eligible]
cached_queries = {p.stem for p in (GOUT/"query_cache").glob("*.npz")}
items = [s for s in all_items if s["well"] in cached_queries]
records = []
for s in items:
    w, d, ps = s["well"], s["d"], s["ps"]
    surf = np.load(GOUT/f"query_cache/{w}.npz")["idw40_a1.0"]
    rows = np.arange(ps+1, len(d))
    p = predict_tvt(d, surf, ps, .25, rows)
    last = float(d.TVT_input.iloc[ps])
    for j, tvt in zip(rows, p):
        records.append({"id": f"{w}_{j}", "well": w, "row": int(j),
                        "target": float(d.TVT.iloc[j]-last),
                        "pred": float(tvt-last)})
geo = pd.DataFrame(records)
geo.to_parquet(OUT/"geostat_oof_predictions.parquet", index=False)

# Meta common rows are in exact official IDs.
gt = pd.read_parquet(ROOT/"exp/public_artifacts/pilkwang/oof/train_gt.parquet",
                     columns=["id"])
common_ids = gt.id.to_numpy()[state["common"]]
lookup = pd.Series(np.arange(len(common_ids)), index=common_ids)
mi = lookup.reindex(geo.id).to_numpy()
if np.isnan(mi).any():
    raise RuntimeError("geostat ID missing from restricted meta rows")
mi = mi.astype(int)
if np.max(np.abs(y[mi]-geo.target.to_numpy())) > .002:
    raise RuntimeError("geostat/meta target identity failed")
yg, wg, gp = y[mi], wells[mi], geo.pred.to_numpy(float)
Z = np.column_stack([legs[k][mi] for k in base_names])
splits = list(GroupKFold(5).split(Z, groups=wg))


def crossfit(x):
    p = np.zeros(len(yg)); coef = []
    for tr, va in splits:
        m = Ridge(alpha=100, positive=True, fit_intercept=False)
        m.fit(x[tr[::8]], yg[tr[::8]])
        p[va] = m.predict(x[va]); coef.append(m.coef_.tolist())
    return p, coef


bp, bc = crossfit(Z)
ap, ac = crossfit(np.c_[Z, gp])
grid = []
for blend in (0, .025, .05, .075, .1, .125, .15, .2):
    grid.append({"blend": blend,
                 "rmse": rmse(yg, (1-blend)*bp+blend*gp)})
grid = pd.DataFrame(grid).sort_values("rmse")
result = {"rows": len(yg), "wells": int(pd.Series(wg).nunique()),
          "base_crossfit": rmse(yg, bp),
          "add_geostat_crossfit": rmse(yg, ap),
          "gain": rmse(yg, bp)-rmse(yg, ap),
          "base_weights": bc, "add_weights": ac,
          "best_fixed_blend": grid.iloc[0].to_dict()}
print(json.dumps(result, indent=2), flush=True)
grid.to_csv(OUT/"fixed_blend_grid.csv", index=False)
(OUT/"summary.json").write_text(json.dumps(result, indent=2))
