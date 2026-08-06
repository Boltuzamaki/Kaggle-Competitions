"""Strict add-one-leg audit around the five-leg deployable meta stack."""
from pathlib import Path
import contextlib
import io
import json
import runpy

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/meta_deploy_addone"
OUT.mkdir(parents=True, exist_ok=True)

with contextlib.redirect_stdout(io.StringIO()):
    state = runpy.run_path(str(ROOT / "exp/meta_all_honest_oof.py"))
legs, y, wells = state["legs"], state["y"], state["wells"]
base_names = [
    "har_physics", "har_lgb", "har_xgb",
    "pil_blend_oof_postprocessed", "v4_lgb7",
]
candidates = [k for k in legs if k not in base_names]
splits = list(GroupKFold(5).split(y, groups=wells))


def crossfit(names):
    z = np.column_stack([legs[k] for k in names])
    pred = np.zeros(len(y))
    fold_rmse = []
    for tr, va in splits:
        model = Ridge(alpha=100, positive=True, fit_intercept=False)
        model.fit(z[tr[::8]], y[tr[::8]])
        pred[va] = model.predict(z[va])
        fold_rmse.append(float(np.sqrt(np.mean((pred[va] - y[va]) ** 2))))
    full = Ridge(alpha=100, positive=True, fit_intercept=False)
    full.fit(z[::8], y[::8])
    return {
        "rmse": float(np.sqrt(np.mean((pred-y) ** 2))),
        "fold_rmse": fold_rmse,
        "weights": dict(zip(names, full.coef_.tolist())),
    }


results = {"base": crossfit(base_names), "add_one": {}}
for candidate in candidates:
    results["add_one"][candidate] = crossfit(base_names + [candidate])
results["best"] = min(
    results["add_one"], key=lambda k: results["add_one"][k]["rmse"])
print(json.dumps(results, indent=2), flush=True)
(OUT / "summary.json").write_text(json.dumps(results, indent=2))
