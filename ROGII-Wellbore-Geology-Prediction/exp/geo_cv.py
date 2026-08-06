"""Parallel CV of geosteering variants on a train-well sample.
Reports pooled + per-well RMSE (pooled ~ the competition metric)."""
import sys, os, time, argparse
import numpy as np
from joblib import Parallel, delayed
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from pf_tracker import pf_predict
from beam_tracker import beam_predict
from wellbore_lib import load_well, list_wells, ps_index


def eval_well(w, n_particles, n_seeds, scale):
    h, tw = load_well("data/train", w)
    ps = ps_index(h)
    if ps >= len(h) - 5 or ps < 5:
        return None
    true = h["TVT"].to_numpy()
    ev = h["TVT_input"].isna().to_numpy()
    tps = float(h["TVT"].iloc[ps - 1])
    pf = pf_predict(h, tw, n_particles=n_particles, n_seeds=n_seeds, scale=scale)
    bm = beam_predict(h, tw)
    y = true[ev]
    def rmse(p): return float(np.sqrt(np.mean((y - p[ev]) ** 2)))
    return dict(well=w, n=int(ev.sum()), y=y,
                const=np.full(ev.sum(), tps),
                pf=pf[ev], beam=bm[ev])


def pooled(res, key_or_arr):
    errs = []
    for r in res:
        p = r[key_or_arr] if isinstance(key_or_arr, str) else key_or_arr(r)
        errs.append((r["y"] - p) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(errs))))


def per_well(res, key_or_arr):
    rs = []
    for r in res:
        p = r[key_or_arr] if isinstance(key_or_arr, str) else key_or_arr(r)
        rs.append(np.sqrt(np.mean((r["y"] - p) ** 2)))
    return float(np.mean(rs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--particles", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--scale", type=float, default=5.0)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    wells = list_wells("data/train")
    rng = np.random.RandomState(7)
    samp = [wells[i] for i in rng.permutation(len(wells))[:a.n]]
    t0 = time.time()
    res = Parallel(n_jobs=a.jobs)(
        delayed(eval_well)(w, a.particles, a.seeds, a.scale) for w in samp)
    res = [r for r in res if r is not None]
    print("wells=%d  time=%.0fs  (particles=%d seeds=%d scale=%.1f)"
          % (len(res), time.time() - t0, a.particles, a.seeds, a.scale))

    variants = {
        "const": "const", "pf": "pf", "beam": "beam",
        "pf+beam .8/.2": lambda r: 0.8 * r["pf"] + 0.2 * r["beam"],
        "pf+beam .7/.3": lambda r: 0.7 * r["pf"] + 0.3 * r["beam"],
        "pf+hold .9": lambda r: 0.9 * r["pf"] + 0.1 * r["const"],
        "pf+beam+hold": lambda r: 0.72 * r["pf"] + 0.18 * r["beam"] + 0.10 * r["const"],
    }
    print("%-16s %8s %8s" % ("variant", "pooled", "perWell"))
    for name, f in variants.items():
        print("%-16s %8.3f %8.3f" % (name, pooled(res, f), per_well(res, f)))
