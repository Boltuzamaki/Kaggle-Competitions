"""Quick local (robust) leave-one-well-out CV of the offset-well structural
surface prior S=TVT+Z. Light: spatial + interp, ~10s."""
import sys, os, glob, time
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wellbore_lib import list_wells, ps_index

t0 = time.time()
wells = list_wells("data/train")
W = {}
for w in wells:
    try:
        h = pd.read_csv(f"data/train/{w}__horizontal_well.csv", usecols=["MD","X","Y","Z","TVT","TVT_input"])
    except Exception:
        continue
    ps = int(h["TVT_input"].notna().sum())
    if ps < 60 or ps >= len(h) - 5:
        continue
    W[w] = dict(X=h["X"].to_numpy(float), Y=h["Y"].to_numpy(float), Z=h["Z"].to_numpy(float),
                TVT=h["TVT"].to_numpy(float), ps=ps)
print("usable wells", len(W))

STRIDE = 12
px, py, ps_, pw = [], [], [], []
for w, d in W.items():
    sl = slice(0, len(d["X"]), STRIDE)
    px.append(d["X"][sl]); py.append(d["Y"][sl])
    ps_.append(d["TVT"][sl] + d["Z"][sl]); pw.append(np.full(len(d["X"][sl]), w))
PX, PY, PSURF, PW = map(np.concatenate, (px, py, ps_, pw))
sc = np.array([PX.std(), PY.std()]); sc = np.where(sc < 1e-6, 1, sc)
tree = cKDTree(np.column_stack([PX/sc[0], PY/sc[1]]))
print("point cloud", len(PX), "built %.0fs" % (time.time()-t0))


def predict(w, K, FETCH=600):
    d = W[w]; ps = d["ps"]; n = len(d["X"])
    q = np.column_stack([d["X"]/sc[0], d["Y"]/sc[1]])
    kk = min(FETCH, len(PX))
    dist, idx = tree.query(q, k=kk, workers=-1)
    self_mask = (PW[idx] == w)
    dist = np.where(self_mask, np.inf, dist)
    order = np.argsort(dist, axis=1)[:, :K]
    dk = np.take_along_axis(dist, order, 1); ik = np.take_along_axis(idx, order, 1)
    valid = np.isfinite(dk)
    wt = np.where(valid, 1.0/(dk + 1.0), 0.0)
    wsum = wt.sum(1)
    S_hat = np.where(wsum > 0, (PSURF[ik]*wt).sum(1) / np.where(wsum > 0, wsum, 1), np.nan)
    tvt_hat = S_hat - d["Z"]
    kn = slice(0, ps); ev = slice(ps, n)
    bias = np.nanmedian(d["TVT"][kn] - tvt_hat[kn])
    if not np.isfinite(bias):
        bias = 0.0
    pred = tvt_hat[ev] + bias
    pred = np.where(np.isfinite(pred), pred, d["TVT"][ps-1])   # fallback -> const
    return d["TVT"][ev], pred, np.full(n - ps, d["TVT"][ps-1])


def pooled(pairs, key):
    e = [(y - (p if key == "ow" else c))**2 for y, p, c in pairs]
    return float(np.sqrt(np.mean(np.concatenate(e))))


for K in [1, 3, 5, 8, 15]:
    pairs = [predict(w, K) for w in W]
    nnan = sum(int(np.isnan(p).any()) for _, p, _ in pairs)
    print("K=%2d  const %.3f   offset-well-S %.3f   (nan-wells %d)  %.0fs"
          % (K, pooled(pairs, "const"), pooled(pairs, "ow"), nnan, time.time()-t0), flush=True)
