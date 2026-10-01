"""Digits have no ordinal meaning - does declaring them categorical help LGBM?"""
import os, sys
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)
os.chdir(_SRC)   # so the ../data and ../artifacts paths below resolve
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import common as C, features as F

X, Xt, y, tid = F.base_frame()
itr, iva = C.get_folds(y)[0]
P = dict(objective="binary", metric="auc", learning_rate=.03, num_leaves=64,
         min_child_samples=200, feature_fraction=.8, bagging_fraction=.8, bagging_freq=1,
         lambda_l2=1.0, n_jobs=4, verbosity=-1, seed=42)
A, (B,) = F.fold_transform(X.iloc[itr], y[itr], [X.iloc[iva]])

def run(cats, tag, extra=None):
    p = dict(P, **(extra or {}))
    a, b = A.copy(), B.copy()
    for c in cats:
        a[c] = a[c].astype("category")
        b[c] = pd.Categorical(b[c], categories=a[c].cat.categories)
    d = lgb.Dataset(a, y[itr]); dv = lgb.Dataset(b, y[iva], reference=d)
    m = lgb.train(p, d, 20000, valid_sets=[dv], callbacks=[lgb.early_stopping(200, verbose=False)])
    print(f"{tag:52s} iter={m.best_iteration:>5} fold0={roc_auc_score(y[iva], m.predict(b)):.6f}", flush=True)

DIG = [c for c in A.columns if c.startswith("inc_d_")] + ["km_dec"]
run([], "all numeric (reference)")
run(DIG, "income digits + km_dec as categorical")
run(DIG + C.CAT_COLS, "digits + native categoricals as categorical")
run(DIG + ["inc_mod100"], "digits + inc_mod100 categorical",
    {"max_cat_threshold": 64, "cat_smooth": 20})
