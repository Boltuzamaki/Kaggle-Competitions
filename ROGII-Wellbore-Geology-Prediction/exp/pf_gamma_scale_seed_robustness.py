"""Robustness audit for the V48 PF gamma-scale clue across seed banks."""
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

SCALES = (1.0, 1.15, 1.2, 1.25, 1.3)
SEED_BANKS = (1000, 10000, 20000)


def evaluate(well):
    horizontal, typewell = load_well("data/train", well)
    ps = ps_index(horizontal)
    if ps < 10 or ps >= len(horizontal) - 5:
        return None
    mask = horizontal["TVT_input"].isna().to_numpy()
    result = {"y": horizontal["TVT"].to_numpy(float)[mask]}
    for seed0 in SEED_BANKS:
        for scale in SCALES:
            result[f"s{seed0}_g{scale}"] = pf_predict_decorr(
                horizontal, typewell, n_particles=250, n_seeds=40,
                scale=10.0, decorrelate=True, seed0=seed0,
                gs_mult=scale)[mask]
    return result


def pooled(rows, key):
    errors = np.concatenate([(row["y"] - row[key]) ** 2 for row in rows])
    return float(np.sqrt(np.mean(errors)))


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    wells = list_wells("data/train")
    sample = [wells[i] for i in np.random.RandomState(19).permutation(len(wells))[:count]]
    started = time.time()
    rows = [row for row in Parallel(n_jobs=10)(delayed(evaluate)(w) for w in sample)
            if row is not None]
    print(f"wells={len(rows)} seconds={time.time()-started:.0f}")
    scores = {}
    for seed0 in SEED_BANKS:
        for scale in SCALES:
            key = f"s{seed0}_g{scale}"
            scores[key] = pooled(rows, key)
            print(f"seed0={seed0:5d} gamma={scale:4.2f} rmse={scores[key]:.6f}")
    for scale in SCALES:
        vals = [scores[f"s{seed0}_g{scale}"] for seed0 in SEED_BANKS]
        print(f"gamma={scale:4.2f} seed_mean={np.mean(vals):.6f} "
              f"seed_worst={np.max(vals):.6f}")
