"""Blend the experiment OOFs.

Only models trained in this repo are blended - every OOF comes from the same
StratifiedKFold(5, seed=42) split, so the weights are chosen on honest
out-of-fold predictions rather than on the leaderboard.

Three blenders are compared on the OOF matrix:
  * simple rank average
  * non-negative hill climbing on ranks (greedy, with replacement)
  * logistic-regression stack on rank features, itself cross-validated
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import common as C


def load(names=None):
    if not names:
        avail = sorted(f[:-4] for f in os.listdir(C.OOF) if f.endswith(".npy"))
        # Default to the final 10-fold suite; the 5-fold exploration runs are kept on
        # disk for the write-up but are superseded and stay out of the blend.
        names = [n for n in avail if n.startswith("final_")] or avail
    names = [n for n in names if os.path.exists(os.path.join(C.PRED, n + ".npy"))]
    O = np.column_stack([np.load(os.path.join(C.OOF, n + ".npy")) for n in names])
    T = np.column_stack([np.load(os.path.join(C.PRED, n + ".npy")) for n in names])
    return names, O, T


def to_rank(M):
    return np.column_stack([rankdata(M[:, i]) / len(M) for i in range(M.shape[1])])


def hill_climb(R, y, n_iter=400, step=1.0):
    """Greedy forward selection with replacement.

    Each round adds one more unit of whichever model improves OOF AUC most, so the
    weights are a simple count vector. Stops as soon as no single addition helps,
    which keeps it from chasing OOF noise.
    """
    n = R.shape[1]
    scores = [roc_auc_score(y, R[:, i]) for i in range(n)]
    start = int(np.argmax(scores))
    w = np.zeros(n); w[start] = step
    cur = R[:, start].copy(); best = scores[start]
    for _ in range(n_iter):
        k = w.sum()
        cand = [(roc_auc_score(y, (cur * k + R[:, i] * step) / (k + step)), i) for i in range(n)]
        s, i = max(cand)
        if s <= best + 1e-9:
            break
        cur = (cur * k + R[:, i] * step) / (k + step); w[i] += step; best = s
    return w / w.sum(), best


def optimise_weights(R, y, w0):
    """Polish the hill-climb weights with a continuous non-negative search."""
    from scipy.optimize import minimize

    def neg(v):
        v = np.abs(v)
        if v.sum() < 1e-9:
            return 0.0
        return -roc_auc_score(y, R @ (v / v.sum()))

    r = minimize(neg, w0, method="Nelder-Mead",
                 options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-8})
    v = np.abs(r.x); v = v / v.sum()
    return v, -neg(r.x)


def main():
    names, O, T = load(sys.argv[1:] or None)
    y = (pd.read_csv(os.path.join(C.DATA, "train.csv"), usecols=[C.TARGET])[C.TARGET] == "Yes").astype(int).values
    print("models in the blend:")
    for i, n in enumerate(names):
        print(f"  {n:28s} OOF AUC {roc_auc_score(y, O[:, i]):.6f}")

    Cm = np.corrcoef(to_rank(O).T)
    print("\nrank correlation between model OOFs (lower = more to gain from blending):")
    print(pd.DataFrame(Cm, index=names, columns=[n[:9] for n in names]).round(4).to_string())

    Ro, Rt = to_rank(O), to_rank(T)
    res = {}
    res["rank_mean"] = (roc_auc_score(y, Ro.mean(1)), Rt.mean(1))

    w, s = hill_climb(Ro, y)
    print("\nhill-climb weights:", {n: round(float(x), 4) for n, x in zip(names, w) if x > 0})
    res["hill_climb"] = (s, Rt @ w)

    wo, so = optimise_weights(Ro, y, w)
    print("polished weights   :", {n: round(float(x), 4) for n, x in zip(names, wo) if x > 1e-3})
    res["weighted"] = (so, Rt @ wo)

    # cross-validated logistic stack so the reported number is not in-sample
    oof_stack = np.zeros(len(y)); test_stack = np.zeros(len(Rt))
    for itr, iva in C.get_folds(y):
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Ro[itr], y[itr])
        oof_stack[iva] = lr.decision_function(Ro[iva])
        test_stack += lr.decision_function(Rt) / C.N_SPLITS
    res["logistic_stack"] = (roc_auc_score(y, oof_stack), test_stack)

    print("\nblend results (all measured out-of-fold):")
    for k, (s, _) in res.items():
        print(f"  {k:16s} {s:.6f}")
    best = max(res, key=lambda k: res[k][0])
    print(f"\nbest blender: {best}  OOF AUC {res[best][0]:.6f}")

    tid = pd.read_csv(os.path.join(C.DATA, "test.csv"), usecols=["id"])["id"].values
    p = res[best][1]
    p = (rankdata(p) - 0.5) / len(p)            # AUC only needs the ordering
    C.make_submission(p, tid, f"blend_{best}.csv")
    json.dump({"models": names, "best": best,
               "scores": {k: float(v[0]) for k, v in res.items()},
               "hill_climb_weights": dict(zip(names, map(float, w)))},
              open(os.path.join(C.ROOT, "artifacts", "blend.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
