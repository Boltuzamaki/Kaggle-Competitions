"""Exact five-leg deployable meta add-one audit for heel GR datum path."""
from pathlib import Path
import contextlib, io, json, runpy
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"exp/results/self_prefix_gr_shift/full_meta"
OUT.mkdir(parents=True, exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):
    state = runpy.run_path(str(ROOT/"exp/meta_all_honest_oof.py"))
legs, y, wells, common = state["legs"], state["y"], state["wells"], state["common"]
base_names = ["har_physics", "har_lgb", "har_xgb",
              "pil_blend_oof_postprocessed", "v4_lgb7"]

q = np.load(ROOT/"exp/results/self_prefix_gr_shift/oof.npz")
gt = pd.read_parquet(ROOT/"exp/public_artifacts/pilkwang/oof/train_gt.parquet",
                     columns=["id", "target_delta_from_last_known"])
vf = pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")[["id", "target"]]
if len(vf) != len(gt) or not np.array_equal(vf.id.to_numpy(), gt.id.to_numpy()):
    raise RuntimeError("V4/train_gt ID ordering identity failed")
terr = float(np.max(np.abs(vf.target.to_numpy(float) -
                           gt.target_delta_from_last_known.to_numpy(float))))
if terr > .002:
    raise RuntimeError(f"V4/train_gt target identity failed {terr}")
if len(q["correction"]) != len(gt) or not np.isfinite(q["correction"]).all():
    raise RuntimeError("self-prefix OOF length/finiteness failed")
base_err = float(np.max(np.abs(q["base"][common]-legs["v4_lgb7"])))
if base_err > 1e-6:
    raise RuntimeError(f"self-prefix/V4 base alignment failed {base_err}")

new = legs["v4_lgb7"] + q["correction"][common].astype(float)
Z = np.column_stack([legs[k] for k in base_names])
splits = list(GroupKFold(5).split(Z, groups=wells))

def crossfit(A):
    p = np.zeros(len(y)); co = []
    for tr, va in splits:
        m = Ridge(alpha=100, positive=True, fit_intercept=False)
        m.fit(A[tr[::8]], y[tr[::8]])
        p[va] = m.predict(A[va]); co.append(m.coef_.tolist())
    return p, co

def rmse(p, ix=None):
    if ix is None: ix = np.arange(len(y))
    return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))

bp, bc = crossfit(Z)
ap, ac = crossfit(np.c_[Z, new])
fold_gains = [rmse(bp, va)-rmse(ap, va) for _, va in splits]
grid = [{"weight": float(w), "rmse": rmse((1-w)*bp+w*new)}
        for w in np.linspace(0, .3, 13)]
summary = {
    "rows": len(y), "wells": int(pd.Series(wells).nunique()),
    "identity": {"id_exact": True, "target_max_abs": terr,
                 "v4_base_max_abs": base_err, "global_rows": len(gt),
                 "common_rows": len(common)},
    "base_names": base_names, "base_crossfit": rmse(bp),
    "add_crossfit": rmse(ap), "gain": rmse(bp)-rmse(ap),
    "fold_gains": fold_gains, "fold_wins": int(sum(x > 0 for x in fold_gains)),
    "new_leg_rmse": rmse(new), "base_weights": bc, "add_weights": ac,
    "best_fixed_blend": min(grid, key=lambda x:x["rmse"]),
}
assert np.isfinite(ap).all()
pd.DataFrame(grid).to_csv(OUT/"fixed_blend.csv", index=False)
np.savez_compressed(OUT/"oof.npz", base=bp, add=ap, y=y, new=new,
                    global_indices=common)
(OUT/"summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
