"""Whole-well holdout: does the DECORRELATED ensemble beat the correlated one?
Metric = pooled RMSE (the competition metric)."""
import sys, os, time, argparse
import numpy as np
from joblib import Parallel, delayed
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H); sys.path.insert(0, os.path.dirname(_H))
from pf_decorr import pf_predict_decorr
from wellbore_lib import load_well, list_wells, ps_index


def one(w, n_particles, n_seeds):
    h, tw = load_well("data/train", w); ps = ps_index(h)
    if ps >= len(h) - 5 or ps < 10:
        return None
    ev = h["TVT_input"].isna().to_numpy(); y = h["TVT"].to_numpy()[ev]
    tps = float(h["TVT"].iloc[ps - 1])
    corr = pf_predict_decorr(h, tw, n_particles, n_seeds, scale=10.0, decorrelate=False)[ev]
    deco = pf_predict_decorr(h, tw, n_particles, n_seeds, scale=10.0, decorrelate=True)[ev]
    return {"y": y, "const": np.full(len(y), tps), "corr": corr, "deco": deco}


def pooled(res, k): return float(np.sqrt(np.mean(np.concatenate([(r["y"] - r[k]) ** 2 for r in res]))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--particles", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=120)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    wells = list_wells("data/train")
    rng = np.random.RandomState(7)              # fixed holdout (matches CV protocol)
    samp = [wells[i] for i in rng.permutation(len(wells))[:a.n]]
    t0 = time.time()
    res = [r for r in Parallel(n_jobs=a.jobs)(
        delayed(one)(w, a.particles, a.seeds) for w in samp) if r]
    dt = time.time() - t0
    print("holdout wells=%d  seeds=%d particles=%d  time=%.0fs"
          % (len(res), a.seeds, a.particles, dt))
    print("  const                pooled %.3f" % pooled(res, "const"))
    print("  PF correlated        pooled %.3f" % pooled(res, "corr"))
    print("  PF DECORRELATED      pooled %.3f" % pooled(res, "deco"))
    d = pooled(res, "corr") - pooled(res, "deco")
    print("  --> decorrelation gain: %+.3f ft pooled" % d)
