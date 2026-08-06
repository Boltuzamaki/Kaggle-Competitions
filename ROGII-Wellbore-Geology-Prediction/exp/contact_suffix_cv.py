"""Rolling-suffix validation of the target-free contact formula.

For each train well (which HAS its formation columns, exactly like the test
wells' train twins), reconstruct  raw = ref_tvt - (Z - formation_col),
fit the bias on the VISIBLE PREFIX only, predict the HIDDEN SUFFIX, and score.
This tests whether the contact formula is an accurate hidden-suffix predictor
(the basis of the ~6.7 estimate) and how it generalizes across all 773 wells.

Also tests multi-contact consensus: pick the formation whose prefix-audit RMSE
is lowest (held-out inside the prefix), never using suffix TVT.
"""
import sys, os, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_H))
from wellbore_lib import load_well, list_wells, ps_index

FORMS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def raw_contact(h, tw, ref):
    if ref not in h.columns:
        return None
    g = tw.dropna(subset=["Geology", "TVT"])
    r = g.loc[g["Geology"].astype(str) == ref, "TVT"]
    if r.empty:
        return None
    ref_tvt = float(r.min())
    raw = ref_tvt - (h["Z"].to_numpy(float) - h[ref].to_numpy(float))
    return raw if np.isfinite(raw).sum() >= 100 else None


def one(w):
    h, tw = load_well("data/train", w); ps = ps_index(h)
    if ps < 60 or ps >= len(h) - 5:
        return None
    tvt = h["TVT"].to_numpy(float)
    pre = np.arange(ps); suf = np.arange(ps, len(h))
    y = tvt[suf]; tps = tvt[ps - 1]
    out = {"y": y, "const": np.full(len(y), tps)}
    # inner prefix split for held-out audit (never touches suffix)
    isp = int(0.7 * ps)
    per_ref = {}
    for ref in FORMS:
        raw = raw_contact(h, tw, ref)
        if raw is None:
            continue
        bias = np.nanmedian(tvt[:isp] - raw[:isp])          # fit on inner-prefix
        audit_rmse = np.sqrt(np.nanmean((tvt[isp:ps] - (raw[isp:ps] + bias)) ** 2))
        # refit bias on FULL prefix for the actual suffix prediction
        bias_full = np.nanmedian(tvt[:ps] - raw[:ps])
        pred = raw[suf] + bias_full
        per_ref[ref] = (audit_rmse, pred)
    if not per_ref:
        return None
    out["EGFDU"] = per_ref["EGFDU"][1] if "EGFDU" in per_ref else out["const"]
    # multi-contact consensus: pick lowest prefix-audit RMSE
    best_ref = min(per_ref, key=lambda k: per_ref[k][0])
    out["best_prefix"] = per_ref[best_ref][1]
    # mean of the 3 best-audit refs
    top = sorted(per_ref, key=lambda k: per_ref[k][0])[:3]
    out["consensus3"] = np.nanmean([per_ref[k][1] for k in top], axis=0)
    return out


def pooled(res, k):
    e = [(r["y"] - r[k]) ** 2 for r in res if k in r]
    return float(np.sqrt(np.mean(np.concatenate(e))))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    wells = list_wells("data/train"); rng = np.random.RandomState(7)
    samp = [wells[i] for i in rng.permutation(len(wells))[:n]]
    t0 = time.time()
    res = [r for r in Parallel(n_jobs=6)(delayed(one)(w) for w in samp) if r]
    print("wells=%d time=%.0fs" % (len(res), time.time() - t0))
    for k in ["const", "EGFDU", "best_prefix", "consensus3"]:
        print("  %-12s pooled %.3f" % (k, pooled(res, k)))
    # how often EGFDU is the prefix-best
