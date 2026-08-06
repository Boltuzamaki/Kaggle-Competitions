"""Outer-GKF function-valued kernel regression on immutable causal-prefix v1.

Implements the separable operator-valued kernel K(x,x') I described by Kadri
et al. (JMLR 2016).  A scalar RBF covariance compares legal prefix/query
fingerprints while every response is the complete 128-point residual function.
Kernel bandwidth and ridge/noise are selected by GCV on each outer-training
fold only; held-out curves never influence either choice.
"""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz"
OUT = ROOT / "exp/results/functional_operator_kernel_causal_prefix_v2"
OUT.mkdir(parents=True, exist_ok=True)


def sqdist(a, b):
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    return np.maximum(aa + bb - 2 * a @ b.T, 0.0)


def select_gcv(d2, y):
    """Training-only GCV for RBF scale and KRR regularization."""
    nz = d2[np.triu_indices(len(d2), 1)]
    med = float(np.median(nz[nz > 0]))
    candidates = []
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        gamma = scale / max(med, 1e-12)
        k = np.exp(-gamma * d2)
        eig, u = np.linalg.eigh(k)
        eig = np.maximum(eig, 0.0)
        uy = u.T @ y
        for lam in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
            shrink_resid = lam / (eig + lam)
            sse = float(np.sum((shrink_resid[:, None] * uy) ** 2))
            denom = max(len(y) - float(np.sum(eig / (eig + lam))), 1e-6)
            candidates.append((sse / (denom * denom * y.shape[1]), gamma, lam))
    return min(candidates), med


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


z = np.load(SRC, allow_pickle=True)
wells = z["wells"].astype(str)
q = z["query"].astype(float)
c = z["true"].astype(float)
assert sha256(SRC) == "669f86aaca1fc9f86038826d4206cb45796b5a19ead79ab67eb55e75a32b2076"

pred = np.zeros_like(c)
fold_id = np.full(len(wells), -1, int)
folds = []
for fold, (tr, va) in enumerate(GroupKFold(5).split(q, groups=wells)):
    sc = StandardScaler().fit(q[tr])
    a0, b0 = sc.transform(q[tr]), sc.transform(q[va])
    # Unsupervised compression is fitted inside the outer fold. Whitening gives
    # the RBF a functional Mahalanobis distance without response leakage.
    pc = PCA(n_components=min(48, len(tr) - 1), whiten=True, random_state=812).fit(a0)
    a, b = pc.transform(a0), pc.transform(b0)
    mu = c[tr].mean(axis=0)
    yc = c[tr] - mu
    dtt = sqdist(a, a)
    best, median_d2 = select_gcv(dtt, yc)
    _, gamma, lam = best
    k = np.exp(-gamma * dtt)
    alpha = np.linalg.solve(k + lam * np.eye(len(tr)), yc)
    pred[va] = mu + np.exp(-gamma * sqdist(b, a)) @ alpha
    fold_id[va] = fold
    folds.append({"fold": fold, "train_wells": len(tr), "valid_wells": len(va),
                  "gamma": gamma, "lambda": lam, "median_d2": median_d2,
                  "gcv": best[0]})
    print(folds[-1], flush=True)

# Recreate exact rows in the same sorted-well order used by causal-prefix v1.
f = pd.read_pickle(ROOT / "r_v4b/train_feats.pkl")
zz = np.load(ROOT / "exp/results/heel_calibrated_gr_datum/full_meta/oof.npz")
f = f.iloc[zz["global_indices"].astype(int)].reset_index(drop=True)
s = np.load(ROOT / "exp/results/pf_student10_full_oof/meta_oof.npz", allow_pickle=True)
assert np.allclose(s["y"], f.target)
base = 0.1 * s["accepted"] + 0.9 * s["replacement"]
f["_base"] = base

row_pred, row_fold, row_y, row_base = [], [], [], []
groups = list(f.groupby("well", sort=True))
assert [str(w) for w, _ in groups] == wells.tolist()
for i, (_, g) in enumerate(groups):
    row_pred.append(np.interp(np.linspace(0, 1, len(g)), np.linspace(0, 1, 128), pred[i]))
    row_fold.append(np.full(len(g), fold_id[i], int))
    row_y.append(g.target.to_numpy(float))
    row_base.append(g._base.to_numpy(float))
row_pred = np.concatenate(row_pred)
row_fold = np.concatenate(row_fold)
y = np.concatenate(row_y)
base = np.concatenate(row_base)

grid = []
for blend in (0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
    p = base + blend * row_pred
    rec = {"blend": blend, "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}
    for k in range(5):
        m = row_fold == k
        rec[f"fold_{k}"] = float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
    grid.append(rec)
grid = pd.DataFrame(grid)
best = grid.iloc[int(grid.rmse.argmin())].to_dict()
et_verification = float(np.sqrt(np.mean((y - (base + np.concatenate([
    np.interp(np.linspace(0, 1, len(g)), np.linspace(0, 1, 128), z["et"][i])
    for i, (_, g) in enumerate(groups)]))) ** 2)))
expected_et = 8.113131030261947
expected_fold_base = np.array([7.645085532544145, 9.461489012436388,
                               7.619660958395028, 8.697194190199744,
                               7.214437333708504])
actual_fold_base = grid.loc[grid.blend == 0, [f"fold_{k}" for k in range(5)]].to_numpy()[0]
assert abs(et_verification - expected_et) < 1e-10, (et_verification, expected_et)
assert np.allclose(actual_fold_base, expected_fold_base, atol=1e-10), (actual_fold_base, expected_fold_base)
summary = {"method": "separable operator-valued RBF kernel ridge",
           "source_npz": str(SRC.relative_to(ROOT)), "source_sha256": sha256(SRC),
           "wells": len(wells), "rows": len(y),
           "baseline": float(np.sqrt(np.mean((y - base) ** 2))),
           "immutable_et_full_blend_verification": et_verification,
           "best": best, "gain_vs_et": et_verification - best["rmse"], "folds": folds,
           "references": [
             "Kadri et al. (2016), Operator-valued Kernels for Learning from Functional Response Data, JMLR 17(20), https://jmlr.org/papers/v17/11-315.html",
             "Shi and Choi (2011), Gaussian Process Regression Analysis for Functional Data, https://doi.org/10.1201/b11038"
           ]}
grid.to_csv(OUT / "blend_grid.csv", index=False)
np.savez_compressed(OUT / "oof_curves.npz", wells=wells, prediction=pred,
                    fold=fold_id, row_prediction=row_pred, row_fold=row_fold)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
