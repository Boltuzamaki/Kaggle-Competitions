"""Strict nested-CV audit of conservative routing for predeclared hard wells."""
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/conservative_group_routing"
OUT.mkdir(parents=True, exist_ok=True)
GT = pd.read_parquet(ROOT / "exp/public_artifacts/pilkwang/oof/train_gt.parquet")
yfull = GT.target_delta_from_last_known.to_numpy(float)
hr = pd.read_pickle(ROOT / "exp/results/harshini_cached_xgb/rows.pkl")
bywell = {w: q.index.to_numpy() for w, q in GT.groupby("well_id", sort=False)}
hidx = np.empty(len(hr), int)
for w, q in hr.groupby("well", sort=False):
    hidx[q.index.to_numpy()] = bywell[w]
y = yfull[hidx]
wells = GT.well_id.to_numpy()[hidx]
warm = 1 - np.exp(-np.maximum(hr.md_since.to_numpy(float), 0) / 85.)
legs = {
    "har_physics": warm * hr.blend_d.to_numpy(float),
    "har_lgb": warm * np.load(ROOT / "exp/results/harshini_cached_xgb/lgb_fast_oof.npy"),
    "har_xgb": warm * np.load(ROOT / "exp/results/harshini_cached_xgb/xgb_oof.npy"),
}
sv = joblib.load(ROOT / "r_v4b/stack_v4_oofs.joblib")["oofs"]
for k in ("lgb123", "lgb7", "xgb", "cat"):
    legs["v4_" + k] = np.asarray(sv[k])[hidx]
pp = ROOT / "exp/public_artifacts/pilkwang/oof"
for k in ("blend_oof_postprocessed", "catboost_oof", "sequence_tcn_oof", "lgb_oof"):
    legs["pil_" + k] = np.asarray(np.load(pp / f"{k}.npy", mmap_mode="r")[hidx])
names = list(legs)
Z = np.column_stack(list(legs.values()))
candidate_names = names + ["family_har_mean", "family_v4_mean", "family_pil_mean"]
C = np.column_stack([
    Z,
    np.mean(Z[:, [names.index(x) for x in names if x.startswith("har_")]], axis=1),
    np.mean(Z[:, [names.index(x) for x in names if x.startswith("v4_")]], axis=1),
    np.mean(Z[:, [names.index(x) for x in names if x.startswith("pil_")]], axis=1),
])
M = pd.read_csv(ROOT / "exp/results/two_group_split_meta/well_groups.csv").set_index("well")
meta_cols = ["dx", "dy", "prefix_u_slope", "prefix_tvt_slope", "ps_frac",
             "gr_missing", "gr_std", "geology_coverage"]
Xwell = M[meta_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
Xrow = Xwell.loc[wells].to_numpy(float)
rules_w = {
    "missing_typewell_formation": ~M.all6.astype(bool),
    "azimuth_major_EW": M.major_ew.astype(bool),
    "negative_prefix_TVT_slope": M.prefix_tvt_slope < 0,
    "azimuth_dx_positive": M.dx > 0,
}
rules = {k: pd.Series(wells).map(v.to_dict()).to_numpy(bool) for k, v in rules_w.items()}
outer = list(GroupKFold(5).split(Z, groups=wells))

def rmse(pred, idx):
    return float(np.sqrt(np.mean((pred[idx] - y[idx]) ** 2)))

def meta_fit_predict(tr, va):
    model = Ridge(alpha=100, positive=True, fit_intercept=False)
    model.fit(Z[tr[::8]], y[tr[::8]])
    return model.predict(Z[va])

base = np.zeros(len(y))
for tr, va in outer:
    base[va] = meta_fit_predict(tr, va)

alphas = np.array([0, .1, .2, .35, .5, .7, 1.0])
results = {}
for rule_name, hard in rules.items():
    routed = base.copy()
    gated = base.copy()
    choices = []
    for fold, (tr, va) in enumerate(outer):
        # Cross-fit the global meta inside outer-train. Thus fallback/gate choice never
        # evaluates an in-sample meta prediction.
        inner_pred = np.zeros(len(tr))
        inner_groups = wells[tr]
        for itr0, iva0 in GroupKFold(4).split(tr, groups=inner_groups):
            itr, iva = tr[itr0], tr[iva0]
            inner_pred[iva0] = meta_fit_predict(itr, iva)
        htr = hard[tr]
        hva = hard[va]
        # Select one existing OOF leg and a conservative shrink coefficient solely
        # on hard-group outer-training rows.
        best = (np.inf, None, None)
        for j, name in enumerate(candidate_names):
            d = C[tr, j] - inner_pred
            for a in alphas:
                p = inner_pred + a * d
                score = rmse(p, np.flatnonzero(htr)) if False else float(
                    np.sqrt(np.mean((p[htr] - y[tr][htr]) ** 2)))
                if score < best[0]:
                    best = (score, j, float(a))
        _, j, a = best
        ridx = va[hva]
        routed[ridx] = base[ridx] + a * (C[ridx, j] - base[ridx])

        # Continuous low-dimensional gate for the selected fallback. Features are
        # standardized well metadata; Ridge predicts the local mixing coefficient
        # through residual interactions d*[1, metadata], clipped to [0,1].
        scaler = StandardScaler().fit(Xrow[tr])
        Xm_tr = scaler.transform(Xrow[tr])
        Xm_va = scaler.transform(Xrow[va])
        dtr = C[tr, j] - inner_pred
        Ftr = np.column_stack([dtr, dtr[:, None] * Xm_tr])
        # Fit only hard wells; high penalty and row thinning limit flexibility.
        ih = np.flatnonzero(htr)[::8]
        gate_model = Ridge(alpha=10000, fit_intercept=False).fit(
            Ftr[ih], (y[tr] - inner_pred)[ih])
        # Convert the fitted residual correction to an explicit gate coefficient.
        coef_gate = Xm_va @ gate_model.coef_[1:] + gate_model.coef_[0]
        coef_gate = np.clip(coef_gate, 0, 1)
        gated[ridx] = base[ridx] + coef_gate[hva] * (C[ridx, j] - base[ridx])
        choices.append({"fold": fold, "leg": candidate_names[j], "alpha": a,
                        "train_hard_wells": int(np.unique(wells[tr][htr]).size),
                        "valid_hard_wells": int(np.unique(wells[va][hva]).size)})

    leg_group_scores = {name: rmse(C[:, j], np.flatnonzero(hard))
                        for j, name in enumerate(candidate_names)}
    fold_route = [rmse(base, va) - rmse(routed, va) for _, va in outer]
    fold_gate = [rmse(base, va) - rmse(gated, va) for _, va in outer]
    results[rule_name] = {
        "hard_wells": int(rules_w[rule_name].sum()),
        "hard_rows": int(hard.sum()),
        "global_meta_hard_rmse": rmse(base, np.flatnonzero(hard)),
        "individual_leg_hard_rmse": leg_group_scores,
        "routed_rmse": rmse(routed, np.arange(len(y))),
        "routed_gain": rmse(base, np.arange(len(y))) - rmse(routed, np.arange(len(y))),
        "routed_fold_gains": fold_route,
        "routed_fold_wins": int(sum(x > 0 for x in fold_route)),
        "gate_rmse": rmse(gated, np.arange(len(y))),
        "gate_gain": rmse(base, np.arange(len(y))) - rmse(gated, np.arange(len(y))),
        "gate_fold_gains": fold_gate,
        "gate_fold_wins": int(sum(x > 0 for x in fold_gate)),
        "choices": choices,
    }

summary = {
    "rows": len(y), "wells": int(np.unique(wells).size),
    "baseline_rmse": rmse(base, np.arange(len(y))),
    "results": results,
    "acceptance": "Positive pooled gain and positive gain in all 5 outer folds.",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
