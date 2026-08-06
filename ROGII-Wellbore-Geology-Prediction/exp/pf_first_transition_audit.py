"""Strict raw-input audit of the PF first hidden-row motion assumption.

The public tracker lineage initializes ``prev_md = eval_md[0] - 1``, which
incorrectly treats the known-to-hidden gap as one foot.  Our promoted PF
prepends the last known station, preserving the real gap but applying one
artificial one-foot state transition.  This audit compares both with the clean
implementation: initialize prev_md directly at the last known MD.
"""
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "exp")]
from wellbore_lib import list_wells, load_well, ps_index
from pf_decorr import _run_pf


def predict(hw, tw, mode, n_particles=100, n_seeds=12):
    tw = tw.sort_values("TVT")
    tt = tw.TVT.to_numpy(float)
    tg = tw.GR.fillna(tw.GR.mean()).to_numpy(float)
    kn = hw[hw.TVT_input.notna()]
    ev = hw[hw.TVT_input.isna()]
    last = kn.iloc[-1]
    at = np.interp(kn.TVT_input, tt, tg)
    gs0 = max(float(np.clip(np.nanstd(kn.GR.fillna(0).to_numpy() - at), 10, 60)), 45.0)
    tail = kn.tail(30)
    dm = np.diff(tail.MD); ok = dm > 0
    ir = float(np.median((np.diff(tail.TVT_input) + np.diff(tail.Z))[ok] / dm[ok]))
    ls = float(last.TVT_input + last.Z)
    gr = hw.GR.interpolate(limit_direction="both").fillna(tg.mean()).to_numpy()[ev.index]
    md0, z0 = ev.MD.to_numpy(float), ev.Z.to_numpy(float)
    if mode == "prepend":
        md = np.r_[float(last.MD), md0]; z = np.r_[float(last.Z), z0]; gr = np.r_[np.nan, gr]
    else:  # public workhorse behavior
        md, z = md0, z0
    preds = []
    lls = []
    for s in range(n_seeds):
        pr = np.random.default_rng(41000 + s)
        p, ll = _run_pf(md, z, gr, tt, tg, ls, ir, gs0 * pr.uniform(.92, 1.12),
                        n_particles, s, .998, .002 * pr.uniform(.93, 1.08),
                        .005 * pr.uniform(.88, 1.15), .1 * pr.uniform(.85, 1.2),
                        .001, pr.uniform(.47, .55), pr.uniform(1.6, 2.4), 10.,
                        float(last.MD) if mode == "explicit" else None)
        preds.append(p); lls.append(ll)
    lls = np.asarray(lls); w = np.exp((lls - lls.max()) / 10); w /= w.sum()
    p = (w[:, None] * np.asarray(preds)).sum(0)
    return p[1:] if mode == "prepend" else p


def one(wid):
    hw, tw = load_well("data/train", wid)
    ps = ps_index(hw)
    if ps < 10 or ps >= len(hw) - 5:
        return None
    y = hw.loc[hw.TVT_input.isna(), "TVT"].to_numpy(float)
    return wid, y, {m: predict(hw, tw, m) for m in ("buggy", "prepend", "explicit")}


if __name__ == "__main__":
    # Fixed deterministic, error-agnostic sample; truth is touched only in scoring.
    wells = list_wells("data/train")
    sample = [wells[i] for i in np.random.default_rng(7381).permutation(len(wells))[:160]]
    rows = [r for r in Parallel(n_jobs=12, verbose=5)(delayed(one)(w) for w in sample) if r]
    summary = {"wells": len(rows), "rows": int(sum(len(r[1]) for r in rows))}
    for mode in ("buggy", "prepend", "explicit"):
        e = np.concatenate([r[2][mode] - r[1] for r in rows])
        summary[mode] = float(np.sqrt(np.mean(e * e)))
    summary["explicit_gain_vs_prepend"] = summary["prepend"] - summary["explicit"]
    summary["explicit_gain_vs_buggy"] = summary["buggy"] - summary["explicit"]
    out = ROOT / "exp/results/pf_first_transition_audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
