"""Fold-0 ablation of the rebuilt feature set, tuned LGBM params."""
import os, sys
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)
os.chdir(_SRC)   # so the ../data and ../artifacts paths below resolve
import json, os, numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import common as C, features as F

P = dict(objective="binary", metric="auc", n_jobs=14, verbosity=-1, seed=42,
         **json.load(open("../artifacts/tuned_lgb.json"))["params"])
X, Xt, y, tid = F.base_frame()
itr, iva = C.get_folds(y)[0]
A, (B,) = F.fold_transform(X.iloc[itr], y[itr], [X.iloc[iva]])
print("full matrix:", A.shape, flush=True)

GROUPS = {
    "hard rules (cliff/dead/spike)": ["is_income_cliff", "is_dead_zone", "is_income_spike", "dist_to_cliff"],
    "real-survey anchors (org_*)":   [c for c in A.columns if c.startswith("org_")],
    "frequency enc (freq_*)":        [c for c in A.columns if c.startswith("freq_")],
    "digits":                        [c for c in A.columns if c[3:5] in ("_d",) and c[:3] in ("ann","dai","age")],
    "multi-scale keys (k_*)":        [c for c in A.columns if c.startswith("k_")],
    "TE smooth=100 only":            [c for c in A.columns if c.endswith("_100")],
    "TE inc_1000 (all smooth)":      [c for c in A.columns if c.startswith("te_inc_1000")],
}

def run(a, b, tag):
    d = lgb.Dataset(a, y[itr]); dv = lgb.Dataset(b, y[iva], reference=d)
    m = lgb.train(P, d, 30000, valid_sets=[dv], callbacks=[lgb.early_stopping(300, verbose=False)])
    s = roc_auc_score(y[iva], m.predict(b))
    print(f"{tag:42s} nf={a.shape[1]:>3} iter={m.best_iteration:>5} fold0={s:.6f}", flush=True)
    return s

full = run(A, B, "FULL (all new features)")
for name, cols in GROUPS.items():
    cols = [c for c in cols if c in A.columns]
    if not cols: continue
    s = run(A.drop(columns=cols), B.drop(columns=cols), f"  minus {name}")
    print(f"      -> contribution {full - s:+.6f}", flush=True)
