"""Hyperparameter sweep of the momentum particle filter (honest, no leak).
One-at-a-time around the baseline, pooled RMSE on a held-out CV sample."""
import sys, os, time, itertools
import numpy as np
from joblib import Parallel, delayed
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H); sys.path.insert(0, os.path.dirname(_H))
from pf2 import _pf_ancc, _gr_sigma
from wellbore_lib import load_well, list_wells, ps_index
import pandas as pd


def prep(w):
    h, tw = load_well("data/train", w); ps = ps_index(h)
    if ps >= len(h) - 5 or ps < 10:
        return None
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
    kn = h[h["TVT_input"].notna()]; ev = h[h["TVT_input"].isna()]
    last = kn.iloc[-1]; last_z = float(last["Z"]); last_tvt = float(last["TVT_input"])
    gs = _gr_sigma(kn, tw_tvt, tw_gr)
    gr_all = h["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).to_numpy()
    idx = ev.index.to_numpy()
    md = np.concatenate([[float(last["MD"])], ev["MD"].to_numpy(float)])
    z = np.concatenate([[last_z], ev["Z"].to_numpy(float)])
    gr = np.concatenate([[np.nan], gr_all[idx]])
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].to_numpy()); dz = np.diff(tail["Z"].to_numpy())
    dmv = np.diff(tail["MD"].to_numpy()); m = dmv > 0
    ir = float(np.median((dt + dz)[m] / dmv[m])) if m.sum() >= 3 else 0.0
    return dict(md=md, z=z, gr=gr, tw_tvt=tw_tvt, tw_gr=tw_gr, gs=gs,
                ls=last_tvt + last_z, ir=ir, y=h["TVT"].to_numpy()[ev.index.to_numpy()])


def run_cfg(d, N, seeds, scale, MOM, VN, PN):
    preds = []; lls = []
    for s in range(seeds):
        p, ll = _pf_ancc(d["md"], d["z"], d["gr"], d["tw_tvt"], d["tw_gr"],
                         d["ls"], d["ir"], d["gs"], N, s, MOM=MOM, VN=VN, PN=PN)
        preds.append(p[1:]); lls.append(ll)
    lls = np.array(lls); lls -= lls.max(); wt = np.exp(lls / scale); wt /= wt.sum()
    est = (wt[:, None] * np.stack(preds, 0)).sum(0)
    return (d["y"] - est) ** 2


if __name__ == "__main__":
    NW, N, SEEDS, JOBS = 60, 500, 32, 10
    wells = list_wells("data/train"); rng = np.random.RandomState(7)
    samp = [wells[i] for i in rng.permutation(len(wells))[:NW]]
    data = [d for d in (prep(w) for w in samp) if d is not None]
    print(f"prepared {len(data)} wells; N={N} seeds={SEEDS}")

    def pooled(cfg):
        errs = Parallel(n_jobs=JOBS)(delayed(run_cfg)(d, N, SEEDS, *cfg) for d in data)
        return float(np.sqrt(np.mean(np.concatenate(errs))))

    base = (5.0, 0.998, 0.002, 0.005)  # scale, MOM, VN, PN
    t0 = time.time()
    print("baseline (scale5 MOM.998 VN.002 PN.005): pooled %.3f" % pooled(base))
    grid = {
        "scale": [(s, 0.998, 0.002, 0.005) for s in (3.0, 8.0, 12.0)],
        "MOM":   [(5.0, m, 0.002, 0.005) for m in (0.99, 0.995, 0.999)],
        "PN":    [(5.0, 0.998, 0.002, p) for p in (0.002, 0.01, 0.02)],
        "VN":    [(5.0, 0.998, v, 0.005) for v in (0.001, 0.004, 0.008)],
    }
    for name, cfgs in grid.items():
        for c in cfgs:
            print(f"  {name} {c}: pooled %.3f" % pooled(c), flush=True)
    print("done %.0fs" % (time.time() - t0))
