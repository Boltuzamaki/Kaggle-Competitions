"""Honest CV comparing baseline PF vs pf2 variants (no leak)."""
import sys, os, time, argparse
import numpy as np
from joblib import Parallel, delayed
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H); sys.path.insert(0, os.path.dirname(_H))
from pf_tracker import pf_predict
from pf2 import pf2_predict
from beam_tracker import beam_predict
from wellbore_lib import load_well, list_wells, ps_index


def one(w, N, seeds, scale):
    h, tw = load_well("data/train", w); ps = ps_index(h)
    if ps >= len(h) - 5 or ps < 10:
        return None
    ev = h["TVT_input"].isna().to_numpy(); y = h["TVT"].to_numpy()[ev]
    tps = float(h["TVT"].iloc[ps - 1])
    r = {"y": y, "const": np.full(ev.sum(), tps)}
    r["pf_base"] = pf_predict(h, tw, n_particles=N, n_seeds=seeds, scale=scale)[ev]
    r["pf_anc2"] = pf2_predict(h, tw, N=N, n_seeds=seeds, scale=scale, use_z=False)[ev]
    r["pf_z"]    = pf2_predict(h, tw, N=N, n_seeds=seeds, scale=scale, use_z=True, w_ancc=0.0)[ev]
    r["pf_ens"]  = pf2_predict(h, tw, N=N, n_seeds=seeds, scale=scale, use_z=True, w_ancc=0.5)[ev]
    r["beam"]    = beam_predict(h, tw)[ev]
    return r


def pooled(res, k): return float(np.sqrt(np.mean(np.concatenate([(r["y"] - r[k]) ** 2 for r in res]))))
def perwell(res, k): return float(np.mean([np.sqrt(np.mean((r["y"] - r[k]) ** 2)) for r in res]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--particles", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--scale", type=float, default=5.0)
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    wells = list_wells("data/train")
    rng = np.random.RandomState(7)
    samp = [wells[i] for i in rng.permutation(len(wells))[:a.n]]
    t0 = time.time()
    res = [r for r in Parallel(n_jobs=a.jobs)(
        delayed(one)(w, a.particles, a.seeds, a.scale) for w in samp) if r]
    print("wells=%d time=%.0fs (N=%d seeds=%d scale=%.1f)"
          % (len(res), time.time() - t0, a.particles, a.seeds, a.scale))
    print("%-10s %8s %8s" % ("variant", "pooled", "perWell"))
    for k in ["const", "beam", "pf_base", "pf_anc2", "pf_z", "pf_ens"]:
        print("%-10s %8.3f %8.3f" % (k, pooled(res, k), perwell(res, k)))
    # ensemble weight sweep pf_ens = w*anc + (1-w)*z  (recompute cheaply)
    for wA in [0.3, 0.4, 0.5, 0.6, 0.7]:
        blend = [{"y": r["y"], "b": wA * r["pf_anc2"] + (1 - wA) * r["pf_z"]} for r in res]
        p = np.sqrt(np.mean(np.concatenate([(r["y"] - r["b"]) ** 2 for r in blend])))
        print("  ens w_anc=%.1f pooled %.3f" % (wA, p))
