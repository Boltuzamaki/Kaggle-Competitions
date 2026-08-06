"""Confirm the tuned honest config (scale=12, heavier particles/seeds) on 80 wells.
Compares pure PF vs PF+beam+hold (CV-fixed weights, no LB tuning, no leak)."""
import sys, os, time, argparse
import numpy as np
from joblib import Parallel, delayed
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H); sys.path.insert(0, os.path.dirname(_H))
from pf_tracker import pf_predict
from beam_tracker import beam_predict
from wellbore_lib import load_well, list_wells, ps_index


def one(w, N, seeds, scale):
    h, tw = load_well("data/train", w); ps = ps_index(h)
    if ps >= len(h) - 5 or ps < 10:
        return None
    ev = h["TVT_input"].isna().to_numpy(); y = h["TVT"].to_numpy()[ev]
    tps = float(h["TVT"].iloc[ps - 1])
    pf = pf_predict(h, tw, n_particles=N, n_seeds=seeds, scale=scale)[ev]
    bm = beam_predict(h, tw)[ev]
    return {"y": y, "pf": pf, "bm": bm, "hold": np.full(len(y), tps)}


def pooled(res, f): return float(np.sqrt(np.mean(np.concatenate([(r["y"] - f(r)) ** 2 for r in res]))))
def perwell(res, f): return float(np.mean([np.sqrt(np.mean((r["y"] - f(r)) ** 2)) for r in res]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80); ap.add_argument("--particles", type=int, default=700)
    ap.add_argument("--seeds", type=int, default=48); ap.add_argument("--scale", type=float, default=12.0)
    ap.add_argument("--jobs", type=int, default=10); a = ap.parse_args()
    wells = list_wells("data/train"); rng = np.random.RandomState(7)
    samp = [wells[i] for i in rng.permutation(len(wells))[:a.n]]
    t0 = time.time()
    res = [r for r in Parallel(n_jobs=a.jobs)(
        delayed(one)(w, a.particles, a.seeds, a.scale) for w in samp) if r]
    print("wells=%d time=%.0fs N=%d seeds=%d scale=%.0f" % (len(res), time.time()-t0, a.particles, a.seeds, a.scale))
    variants = {
        "pf": lambda r: r["pf"],
        "pf+beam .85/.15": lambda r: .85*r["pf"] + .15*r["bm"],
        "pf+beam+hold .72/.18/.10": lambda r: .72*r["pf"] + .18*r["bm"] + .10*r["hold"],
        "pf+hold .9/.1": lambda r: .9*r["pf"] + .1*r["hold"],
    }
    print("%-26s %8s %8s" % ("variant", "pooled", "perWell"))
    for name, f in variants.items():
        print("%-26s %8.3f %8.3f" % (name, pooled(res, f), perwell(res, f)))
