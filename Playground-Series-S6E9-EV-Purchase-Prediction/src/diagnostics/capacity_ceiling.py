"""Capacity sweep on a single fold: is the baseline under- or over-fitting?"""
import os, sys
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)
os.chdir(_SRC)   # so the ../data and ../artifacts paths below resolve
import lightgbm as lgb, numpy as np
from sklearn.metrics import roc_auc_score
import common as C

X, Xt, y, tid = C.build(level="raw")
CATS = C.CAT_COLS
for c in CATS: X[c] = X[c].astype("category")
folds = C.get_folds(y); itr, iva = folds[0]
Xtr, ytr, Xva, yva = X.iloc[itr], y[itr], X.iloc[iva], y[iva]
dtr = lgb.Dataset(Xtr, ytr); dva = lgb.Dataset(Xva, yva, reference=dtr)

grid = [
    ("lr.05 leaves64 mcs100  (baseline)", dict(learning_rate=.05, num_leaves=64,  min_child_samples=100)),
    ("lr.03 leaves31 mcs200",             dict(learning_rate=.03, num_leaves=31,  min_child_samples=200)),
    ("lr.02 leaves16 mcs500",             dict(learning_rate=.02, num_leaves=16,  min_child_samples=500)),
    ("lr.02 leaves256 mcs2000",           dict(learning_rate=.02, num_leaves=256, min_child_samples=2000)),
    ("lr.01 leaves128 mcs1000",           dict(learning_rate=.01, num_leaves=128, min_child_samples=1000)),
    ("lr.02 depth4  mcs500",              dict(learning_rate=.02, num_leaves=16, max_depth=4, min_child_samples=500)),
]
for name, over in grid:
    p = dict(objective="binary", metric="auc", feature_fraction=.8, bagging_fraction=.8,
             bagging_freq=1, lambda_l2=1.0, n_jobs=16, verbosity=-1, seed=C.SEED)
    p.update(over)
    m = lgb.train(p, dtr, num_boost_round=20000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    va = roc_auc_score(yva, m.predict(Xva))
    tr_auc = roc_auc_score(ytr, m.predict(Xtr))
    print(f"{name:38s} best_iter={m.best_iteration:>6}  val={va:.6f}  train={tr_auc:.6f}  gap={tr_auc-va:.4f}", flush=True)
