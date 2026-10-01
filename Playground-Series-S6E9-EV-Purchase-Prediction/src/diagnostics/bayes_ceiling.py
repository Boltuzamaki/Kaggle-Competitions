"""(a) Bayes-AUC ceiling implied by our calibrated model.  (b) id-ordering leakage."""
import os, sys
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)
os.chdir(_SRC)   # so the ../data and ../artifacts paths below resolve
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import common as C

X, Xt, y, tid = C.build(level="raw")
oof = np.load("../artifacts/oof/exp01_lgbm_baseline.npy")
print("baseline OOF AUC:", round(roc_auc_score(y, oof), 6))

# (a) If our OOF probs WERE the true P(y=1|x), what AUC would a perfect model get?
rng = np.random.default_rng(0)
sims = [roc_auc_score((rng.random(len(oof)) < oof).astype(int), oof) for _ in range(5)]
print(f"Bayes-AUC ceiling implied by these probs: {np.mean(sims):.6f} +/- {np.std(sims):.6f}")

# (b) id leakage
tr = pd.read_csv("../data/train.csv")
ids = tr.id.values
print("\nAUC of raw id as a score        :", round(roc_auc_score(y, ids), 6))
print("corr(id, y)                     :", round(float(np.corrcoef(ids, y)[0, 1]), 6))
# positive rate across id blocks
blocks = pd.qcut(ids, 20, labels=False)
pr = pd.Series(y).groupby(blocks).mean()
print("pos-rate by id-ventile: min %.4f max %.4f std %.5f" % (pr.min(), pr.max(), pr.std()))
print("(binomial noise std for n=%d, p=%.3f: %.5f)" % (len(y)//20, y.mean(),
      np.sqrt(y.mean()*(1-y.mean())/(len(y)//20))))

# (c) residual signal in id: does id improve on top of the model?
resid = y - oof
print("corr(id, residual)              :", round(float(np.corrcoef(ids, resid)[0, 1]), 6))
