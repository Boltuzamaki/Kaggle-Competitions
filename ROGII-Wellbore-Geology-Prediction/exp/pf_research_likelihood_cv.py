"""Paper-derived PF likelihood pilot: GR noise floor and Student-t tails."""
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from pf_decorr import pf_predict_decorr
from wellbore_lib import list_wells, load_well, ps_index

# (label, floor, Student-t degrees of freedom); nu=0 is Gaussian.
CONFIGS = (
    ("gauss_adaptive", 0.0, 0.0),
    ("gauss_floor30", 30.0, 0.0),
    ("gauss_floor45", 45.0, 0.0),
    ("gauss_floor60", 60.0, 0.0),
    ("student3_floor30", 30.0, 3.0),
    ("student3_floor45", 45.0, 3.0),
    ("student5_floor30", 30.0, 5.0),
    ("student5_floor45", 45.0, 5.0),
    ("student10_floor45", 45.0, 10.0),
)


def one(well):
    h, tw = load_well("data/train", well)
    ps = ps_index(h)
    if ps < 10 or ps >= len(h)-5:
        return None
    ev = h["TVT_input"].isna().to_numpy()
    out = {"y": h["TVT"].to_numpy(float)[ev]}
    for label, floor, nu in CONFIGS:
        out[label] = pf_predict_decorr(
            h, tw, n_particles=250, n_seeds=40, scale=10.0,
            decorrelate=True, seed0=31000, gs_mult=1.0,
            gs_floor=floor, student_nu=nu)[ev]
    return out


def pooled(rows, key):
    return float(np.sqrt(np.mean(np.concatenate(
        [(r["y"]-r[key])**2 for r in rows]))))


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    wells = list_wells("data/train")
    sample = [wells[i] for i in
              np.random.RandomState(37).permutation(len(wells))[:count]]
    started = time.time()
    rows = [r for r in Parallel(n_jobs=12)(delayed(one)(w) for w in sample)
            if r is not None]
    print(f"wells={len(rows)} seconds={time.time()-started:.0f}")
    for label, _, _ in CONFIGS:
        print(f"{label:22s} {pooled(rows,label):.6f}")
