"""Strict 3x3 nested row-level future-target pilot on legal Harshini features."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/harshini_future_aux_3x3"
OUT.mkdir(parents=True, exist_ok=True)
N_WELLS, STRIDE = 180, 12

print("loading cached legal frame", flush=True)
d = pd.read_pickle(ROOT / "exp/results/harshini_cached_xgb/rows.pkl")
# Deterministic broad pilot, independent of targets.
all_wells = np.array(sorted(d.well.unique()))
rng = np.random.RandomState(73126)
keep = np.sort(rng.choice(all_wells, N_WELLS, replace=False))
d = d[d.well.isin(keep)].reset_index(drop=True)
features = [c for c in d if c not in ("well", "flat", "target")]
X = d[features].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
y = d.target.to_numpy(np.float32)
g = d.well.to_numpy()
warm = 1 - np.exp(-np.maximum(d.md_since.to_numpy(float), 0) / 85.)
physics = (warm * d.blend_d.to_numpy(float)).astype(np.float32)
resid = y - physics

target_names = [
    "resid_at_50", "resid_at_150", "resid_at_400",
    "resid_mean_50", "resid_mean_150", "resid_mean_400",
    "u_slope_150", "u_slope_400",
]
A = np.empty((len(d), len(target_names)), np.float32)
for _, q in d.groupby("well", sort=False):
    ix = q.index.to_numpy()
    # Cached suffix stations are sampled at one-foot MD spacing; md_since is
    # the legal relative measured-depth coordinate (absolute MD is unnecessary).
    md = q.md_since.to_numpy(float)
    rr = resid[ix].astype(float)
    # U = TVT + Z; flat + target reconstruct TVT and dZ is Z relative to PS.
    u = q.flat.to_numpy(float) + y[ix] + q.dZ.to_numpy(float)
    cs = np.r_[0., np.cumsum(rr)]
    for hi, h in enumerate((50., 150., 400.)):
        j = np.minimum(np.searchsorted(md, md + h), len(md)-1)
        A[ix, hi] = rr[j]
        # Future interval includes the current station and is MD-defined.
        cnt = np.maximum(1, j-np.arange(len(ix))+1)
        A[ix, 3+hi] = ((cs[j+1]-cs[np.arange(len(ix))])/cnt).astype(np.float32)
    for si, h in enumerate((150., 400.)):
        j = np.minimum(np.searchsorted(md, md+h), len(md)-1)
        A[ix, 6+si] = ((u[j]-u) / np.maximum(md[j]-md, 1.)).astype(np.float32)

def aux_model(seed):
    return lgb.LGBMRegressor(
        objective="huber", n_estimators=130, learning_rate=.05,
        num_leaves=20, max_depth=7, min_child_samples=100,
        max_bin=127, colsample_bytree=.65, subsample=.8, subsample_freq=1,
        reg_alpha=2., reg_lambda=18., verbosity=-1, n_jobs=12,
        random_state=seed)

def main_model(seed):
    return lgb.LGBMRegressor(
        objective="huber", n_estimators=320, learning_rate=.035,
        num_leaves=28, max_depth=8, min_child_samples=100,
        max_bin=127, colsample_bytree=.7, subsample=.85, subsample_freq=1,
        reg_alpha=2., reg_lambda=16., verbosity=-1, n_jobs=12,
        random_state=seed)

def score(p, ix):
    return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))

pred_plain = np.zeros(len(d), np.float32)
pred_aux = np.zeros(len(d), np.float32)
fold_rows = []
t0 = time.time()
outer = list(GroupKFold(3).split(X, groups=g))
for fold, (tr, va) in enumerate(outer):
    # Use fixed global row stride; entire wells remain isolated by both split levels.
    trfit = tr[tr % STRIDE == fold % STRIDE]
    atr = np.zeros((len(trfit), A.shape[1]), np.float32)
    ava = np.zeros((len(va), A.shape[1]), np.float32)
    inner = GroupKFold(3)
    for ai in range(A.shape[1]):
        for infold, (it, iv) in enumerate(inner.split(X[trfit], groups=g[trfit])):
            m = aux_model(10000*fold + 100*ai + infold)
            m.fit(X[trfit[it]], A[trfit[it], ai])
            atr[iv, ai] = m.predict(X[trfit[iv]])
        m = aux_model(50000 + 100*fold + ai)
        m.fit(X[trfit], A[trfit, ai])
        ava[:, ai] = m.predict(X[va])
    plain = main_model(70000+fold)
    plain.fit(X[trfit], resid[trfit])
    aux = main_model(80000+fold)
    aux.fit(np.c_[X[trfit], atr], resid[trfit])
    pred_plain[va] = physics[va] + plain.predict(X[va])
    pred_aux[va] = physics[va] + aux.predict(np.c_[X[va], ava])
    fold_rows.append({
        "fold": fold, "valid_wells": int(np.unique(g[va]).size),
        "physics": score(physics, va), "plain": score(pred_plain, va),
        "aux": score(pred_aux, va),
        "aux_gain_vs_plain": score(pred_plain, va)-score(pred_aux, va)})
    print(fold_rows[-1], flush=True)

allix = np.arange(len(d))
summary = {
    "protocol": "3 outer x 3 inner GroupKFold by complete well; stride-12 training; all validation suffix rows",
    "wells": int(np.unique(g).size), "rows": len(d), "features": len(features),
    "targets": target_names, "physics_rmse": score(physics, allix),
    "plain_stage2_rmse": score(pred_plain, allix),
    "aux_stage2_rmse": score(pred_aux, allix),
    "aux_gain_vs_plain": score(pred_plain, allix)-score(pred_aux, allix),
    "folds": fold_rows, "seconds": time.time()-t0,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
np.savez_compressed(OUT / "oof.npz", y=y, physics=physics,
                    plain=pred_plain, aux=pred_aux, groups=g)
print(json.dumps(summary, indent=2), flush=True)
