"""Do the window encodings pay? Folds 0-1."""
import os, sys
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)
os.chdir(_SRC)   # so the ../data and ../artifacts paths below resolve
import numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import common as C, features as F

X, Xt, y, tid = F.base_frame()
PRE = []
for itr, iva in C.get_folds(y)[:2]:
    a, (b,) = F.fold_transform(X.iloc[itr], y[itr], [X.iloc[iva]])
    PRE.append((a, y[itr], b, y[iva]))
print("matrix", PRE[0][0].shape, flush=True)
WIN = [c for c in PRE[0][0].columns if c.startswith("win_")]

P = dict(objective="binary", metric="auc", n_jobs=8, verbosity=-1, seed=42,
         learning_rate=0.02, max_depth=5, num_leaves=32, min_child_samples=10,
         bagging_fraction=0.8128, feature_fraction=0.20, lambda_l1=0.0709,
         lambda_l2=2.033, max_bin=1024, bagging_freq=1)

def run(tag, drop=None, **over):
    p = dict(P, **over); s = []
    for a, ya, b, yb in PRE:
        if drop: a, b = a.drop(columns=drop), b.drop(columns=drop)
        d = lgb.Dataset(a, ya); dv = lgb.Dataset(b, yb, reference=d)
        m = lgb.train(p, d, 30000, valid_sets=[dv], callbacks=[lgb.early_stopping(400, verbose=False)])
        s.append(roc_auc_score(yb, m.predict(b)))
    print(f"{tag:38s} nf={a.shape[1]:>3} iter={m.best_iteration:>5} mean={np.mean(s):.6f} {[round(x,6) for x in s]}", flush=True)

run("WITHOUT window encodings", drop=WIN)
run("WITH window encodings")
run("WITH windows, ff=0.30", feature_fraction=0.30)
run("WITH windows, depth3 leaves8 lr.015", max_depth=3, num_leaves=8, learning_rate=0.015)
