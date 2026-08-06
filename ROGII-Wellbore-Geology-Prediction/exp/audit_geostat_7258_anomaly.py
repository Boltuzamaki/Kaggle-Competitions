"""Audit why geostat meta-addone reports a 7.2585 baseline on 196 wells."""
from pathlib import Path
import contextlib, io, json, runpy

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/geostat_7258_anomaly_audit_v1"
OUT.mkdir(parents=True, exist_ok=True)

with contextlib.redirect_stdout(io.StringIO()):
    state = runpy.run_path(str(ROOT / "exp/meta_all_honest_oof.py"))
legs, y, wells = state["legs"], state["y"], state["wells"].astype(str)
names = ["har_physics", "har_lgb", "har_xgb",
         "pil_blend_oof_postprocessed", "v4_lgb7"]
Z = np.column_stack([legs[k] for k in names])

geo = pd.read_parquet(ROOT / "exp/results/geostat_marker_surface/meta_addone/geostat_oof_predictions.parquet")
cached_wells = set(geo.well.astype(str))
mask = np.fromiter((w in cached_wells for w in wells), bool, len(wells))
gt = pd.read_parquet(ROOT / "exp/public_artifacts/pilkwang/oof/train_gt.parquet", columns=["id"])
common_ids = gt.id.to_numpy()[state["common"]]
id_lookup = pd.Series(np.arange(len(common_ids)), index=common_ids)
geo_ix = id_lookup.reindex(geo.id).to_numpy().astype(int)

def rmse(a, b, m=None):
    if m is not None: a, b = a[m], b[m]
    return float(np.sqrt(np.mean((a-b)**2)))

def crossfit(X, yy, ww):
    p = np.zeros(len(yy)); fold = np.full(len(yy), -1, np.int8); coefs = []
    for f, (tr, va) in enumerate(GroupKFold(5).split(X, groups=ww)):
        m = Ridge(alpha=100, positive=True, fit_intercept=False)
        m.fit(X[tr[::8]], yy[tr[::8]])
        p[va] = m.predict(X[va]); fold[va] = f; coefs.append(m.coef_.tolist())
    return p, fold, coefs

p_full, f_full, c_full = crossfit(Z, y, wells)
p_sub, f_sub, c_sub = crossfit(Z[mask], y[mask], wells[mask])
p_geo_order, f_geo_order, c_geo_order = crossfit(Z[geo_ix], y[geo_ix], wells[geo_ix])

# Fixed full-cohort model (descriptive only) reveals whether the subset remains easy
# without giving it cohort-specific folds/weights.
m_global = Ridge(alpha=100, positive=True, fit_intercept=False)
m_global.fit(Z[::8], y[::8])
p_global = m_global.predict(Z)

# Well-level error distributions and cohort attributes.
rows = []
for w in np.unique(wells):
    ii = wells == w
    rows.append({"well": w, "cached": bool(w in cached_wells), "rows": int(ii.sum()),
                 "full_crossfit_rmse": rmse(y, p_full, ii),
                 "global_fit_rmse": rmse(y, p_global, ii)})
wr = pd.DataFrame(rows)

# Compare against modern Student/v6 on precisely the same row identities.
s = np.load(ROOT / "exp/results/pf_student10_full_oof/meta_oof.npz", allow_pickle=True)
assert len(s["y"]) == len(y) and np.max(np.abs(s["y"]-y)) < 1e-9
student = .1*s["accepted"] + .9*s["replacement"]
v6z = np.load(ROOT / "exp/results/generative_curve_prior_expert_errors_v6/oof.npz", allow_pickle=True)
v6map = {str(w): v6z["pred"][i] for i, w in enumerate(v6z["wells"])}
v6 = np.empty(len(y))
for w in np.unique(wells):
    ii = np.flatnonzero(wells == w)
    r = np.interp(np.linspace(0, 1, len(ii)), np.linspace(0, 1, 128), v6map[w])
    v6[ii] = student[ii] + r

res = {
    "rows_full": int(len(y)), "wells_full": int(len(np.unique(wells))),
    "rows_cached": int(mask.sum()), "wells_cached": int(len(np.unique(wells[mask]))),
    "cached_fraction_rows": float(mask.mean()),
    "five_legs": names,
    "reported_exact_reproduction_geo_order": rmse(y[geo_ix], p_geo_order),
    "cached_refit_native_order": rmse(y[mask], p_sub),
    "same_stack_full_765_crossfit": rmse(y, p_full),
    "full_crossfit_scored_cached_rows": rmse(y, p_full, mask),
    "full_crossfit_scored_non_cached_rows": rmse(y, p_full, ~mask),
    "global_descriptive_cached": rmse(y, p_global, mask),
    "global_descriptive_non_cached": rmse(y, p_global, ~mask),
    "student_full": rmse(y, student), "student_cached": rmse(y, student, mask),
    "student_non_cached": rmse(y, student, ~mask),
    "v6_full": rmse(y, v6), "v6_cached": rmse(y, v6, mask),
    "v6_non_cached": rmse(y, v6, ~mask),
    "cached_well_rmse_median_full_crossfit": float(wr.loc[wr.cached, "full_crossfit_rmse"].median()),
    "non_cached_well_rmse_median_full_crossfit": float(wr.loc[~wr.cached, "full_crossfit_rmse"].median()),
    "full_coefs": c_full, "cached_refit_coefs": c_sub,
    "reported_geo_order_coefs": c_geo_order,
    "provenance_warning": (
        "The level-2 GroupKFold is not a fully nested outer refit of level-1 models. "
        "Saved level-1 OOF predictions exclude each prediction's own base fold, but "
        "their training folds are not synchronized to this new outer split; level-2 "
        "training features can therefore have been produced by base models trained on "
        "labels from level-2 validation wells."
    )
}
wr.to_csv(OUT / "well_metrics.csv", index=False)
np.savez_compressed(OUT / "oof.npz", y=y, wells=wells, cached=mask,
                    full_crossfit=p_full, cached_crossfit=p_sub,
                    full_fold=f_full, cached_fold=f_sub, geo_indices=geo_ix,
                    reported_geo_order_crossfit=p_geo_order,
                    reported_geo_order_fold=f_geo_order)
(OUT / "summary.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
