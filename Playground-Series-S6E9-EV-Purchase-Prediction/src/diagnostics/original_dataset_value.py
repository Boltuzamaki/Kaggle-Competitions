"""How informative is the original 10k dataset?"""
import os, sys
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)
os.chdir(_SRC)   # so the ../data and ../artifacts paths below resolve
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import common as C

orig = pd.read_csv("../data/original/EV_Adoption_and_Range_Anxiety_Dataset.csv")
print("original shape", orig.shape)
print("pos rate original:", (orig.Will_Buy_EV == "Yes").mean().round(4), " synthetic:", 0.1746)
print("\ndtypes / uniques vs synthetic:")
tr = pd.read_csv("../data/train.csv")
for c in C.NUM_COLS + C.CAT_COLS:
    print(f"  {c:30s} orig_nuniq={orig[c].nunique():>6} syn_nuniq={tr[c].nunique():>6} "
          f"orig_na={orig[c].isna().sum():>4} orig_range={str(orig[c].min())[:9]}..{str(orig[c].max())[:9]}")

yo = (orig.Will_Buy_EV == "Yes").astype(int).values
Xo = C.add_features(orig.drop(columns=["Buyer_ID", "Will_Buy_EV"]), "raw")

X, Xt, y, tid = C.build(level="raw")
Xo = Xo[X.columns]
folds = C.get_folds(y); itr, iva = folds[0]
P = dict(objective="binary", metric="auc", learning_rate=.02, num_leaves=16, max_depth=4,
         min_child_samples=500, feature_fraction=.8, bagging_fraction=.8, bagging_freq=1,
         lambda_l2=1.0, n_jobs=16, verbosity=-1, seed=42)

# A) model trained ONLY on original 10k -> score the synthetic val fold
d = lgb.Dataset(Xo, yo)
m = lgb.train(P, d, num_boost_round=800)
print("\nA) trained on ORIGINAL only  -> AUC on synthetic fold0 val:",
      round(roc_auc_score(y[iva], m.predict(X.iloc[iva])), 6))

# B) how self-predictable is the original? 5-fold CV on original alone
from sklearn.model_selection import StratifiedKFold
oof = np.zeros(len(yo))
for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xo, yo):
    mm = lgb.train(P, lgb.Dataset(Xo.iloc[a], yo[a]), num_boost_round=800)
    oof[b] = mm.predict(Xo.iloc[b])
print("B) original 10k self-CV AUC :", round(roc_auc_score(yo, oof), 6))

# C) synthetic-trained model scored on original
dtr = lgb.Dataset(X.iloc[itr], y[itr]); dva = lgb.Dataset(X.iloc[iva], y[iva], reference=dtr)
ms = lgb.train(P, dtr, num_boost_round=20000, valid_sets=[dva],
               callbacks=[lgb.early_stopping(200, verbose=False)])
print("C) synthetic-trained -> AUC on original 10k:", round(roc_auc_score(yo, ms.predict(Xo)), 6))
print("   (same model, fold0 val:", round(roc_auc_score(y[iva], ms.predict(X.iloc[iva])), 6), ")")

# D) train on synthetic + original
Xc = pd.concat([X.iloc[itr], Xo], ignore_index=True)
yc = np.concatenate([y[itr], yo])
for w in [1, 5, 20]:
    sw = np.concatenate([np.ones(len(itr)), np.full(len(yo), w)])
    md = lgb.train(P, lgb.Dataset(Xc, yc, weight=sw), num_boost_round=ms.best_iteration)
    print(f"D) synthetic+original (orig weight {w:>2}) fold0 val:",
          round(roc_auc_score(y[iva], md.predict(X.iloc[iva])), 6))
