"""Test the writeup's core claim: weak-GR / strong-continuity beats explicit GR
matching. Sweep a GR-likelihood-weakening factor gs_mult on the holdout."""
import sys, os, time
import numpy as np
from joblib import Parallel, delayed
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H); sys.path.insert(0, os.path.dirname(_H))
from pf_decorr import pf_predict_decorr
from wellbore_lib import load_well, list_wells, ps_index

MULTS = [1.0, 1.2, 1.3, 1.4, 1.5]


def one(w):
    h, tw = load_well("data/train", w); ps = ps_index(h)
    if ps >= len(h) - 5 or ps < 10:
        return None
    ev = h["TVT_input"].isna().to_numpy(); y = h["TVT"].to_numpy()[ev]
    tps = float(h["TVT"].iloc[ps - 1])
    out = {"y": y, "const": np.full(len(y), tps)}
    for m in MULTS:
        out[f"gs{m}"] = pf_predict_decorr(h, tw, 400, 60, scale=10.0,
                                          decorrelate=True, gs_mult=m)[ev]
    return out


def pooled(res, k): return float(np.sqrt(np.mean(np.concatenate([(r["y"] - r[k]) ** 2 for r in res]))))

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    wells = list_wells("data/train"); rng = np.random.RandomState(7)
    samp = [wells[i] for i in rng.permutation(len(wells))[:n]]
    t0 = time.time()
    res = [r for r in Parallel(n_jobs=14)(delayed(one)(w) for w in samp) if r]
    print("holdout wells=%d time=%.0fs" % (len(res), time.time() - t0))
    print("%-14s %8s" % ("gs_mult (GR weak->)", "pooled"))
    print("%-14s %8.3f" % ("const", pooled(res, "const")))
    for m in MULTS:
        print("gs_mult=%-6s %8.3f" % (m, pooled(res, f"gs{m}")))
