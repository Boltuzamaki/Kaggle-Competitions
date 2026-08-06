"""Decomposition-based post-process test on the whole-well holdout.

TVT = -Z (exact wiggle) + smooth trend.  So smooth the TREND g=TVT+Z (removes PF
noise, keeps -Z exact), NOT TVT directly (which damages the real wiggle).
Also tests a weak-GR variant (wider GR sigma -> stronger continuity).
Metric = pooled RMSE.
"""
import sys, os, time, argparse
import numpy as np
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H); sys.path.insert(0, os.path.dirname(_H))
from pf_decorr import pf_predict_decorr
from wellbore_lib import load_well, list_wells, ps_index


def sg(v, w, p=3):
    n = len(v); wl = min(w, n if n % 2 == 1 else n - 1)
    return savgol_filter(v, wl, p) if wl >= p + 2 else v


def one(w, n_particles, n_seeds):
    h, tw = load_well("data/train", w); ps = ps_index(h)
    if ps >= len(h) - 5 or ps < 10:
        return None
    ev = h["TVT_input"].isna().to_numpy()
    y = h["TVT"].to_numpy()[ev]; z = h["Z"].to_numpy()[ev]
    pf = pf_predict_decorr(h, tw, n_particles, n_seeds, scale=10.0, decorrelate=True)[ev]
    return {"y": y, "z": z, "pf": pf}


def pooled(res, fn):
    return float(np.sqrt(np.mean(np.concatenate([(r["y"] - fn(r)) ** 2 for r in res]))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--particles", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=80); ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()
    wells = list_wells("data/train"); rng = np.random.RandomState(7)
    samp = [wells[i] for i in rng.permutation(len(wells))[:a.n]]
    t0 = time.time()
    res = [r for r in Parallel(n_jobs=a.jobs)(
        delayed(one)(w, a.particles, a.seeds) for w in samp) if r]
    print("holdout wells=%d seeds=%d time=%.0fs" % (len(res), a.seeds, time.time() - t0))

    variants = {"raw PF": lambda r: r["pf"]}
    for wsm in [31, 61, 101, 201, 401]:
        variants[f"smoothTVT w{wsm}"] = (lambda r, W=wsm: sg(r["pf"], W))
        variants[f"smoothTREND w{wsm}"] = (lambda r, W=wsm: sg(r["pf"] + r["z"], W) - r["z"])
    print("%-20s %8s" % ("variant", "pooled"))
    for name, fn in variants.items():
        print("%-20s %8.3f" % (name, pooled(res, fn)))
