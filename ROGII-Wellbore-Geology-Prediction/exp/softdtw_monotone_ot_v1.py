"""Target-blind multiscale monotone soft-DTW corridor around legal level2.

The alignment posterior uses complete horizontal GR (legal at inference) and
typewell GR. Targets are accessed only after predictions for both frozen,
disjoint evaluation sets have been materialized.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/softdtw_monotone_ot_v1"
OUT.mkdir(parents=True, exist_ok=True)

lev = np.load(ROOT / "exp/results/legal_level2_all_oof_v1/oof.npz")
stu = np.load(ROOT / "exp/results/pf_student10_full_oof/meta_oof.npz", allow_pickle=True)
groups = stu["groups"].astype(str)
y = stu["y"].astype(float)
center = .1 * stu["accepted"] + .9 * stu["replacement"] + lev["anchored_difference"]
wells = np.array(sorted(set(groups)))
rng = np.random.RandomState(1601)
order = rng.permutation(wells)
pilot, confirm = set(order[:120]), set(order[120:360])

# Frozen target-blind hyperparameters.
OFF = np.arange(-12., 13., 1.)
STRIDE = 8
SCALES = (0., 2., 8., 24.)
TEMP = .45
TRANS = .16
PRIOR = .035
BLEND = .15

def robust(x):
    x = pd.Series(np.asarray(x, float)).interpolate(limit_direction="both").to_numpy()
    lo, hi = np.quantile(x, [.01, .99])
    return np.clip((x-lo)/(hi-lo+1e-6), 0, 1)

def pyramid(x):
    z = robust(x)
    return np.stack([z if s == 0 else gaussian_filter1d(z, s) for s in SCALES], 1)

def one(w):
    mask = groups == w
    n = int(mask.sum())
    h = pd.read_csv(ROOT / "data/train" / f"{w}__horizontal_well.csv")
    tw = pd.read_csv(ROOT / "data/train" / f"{w}__typewell.csv").dropna().sort_values("TVT")
    ps = int(h.TVT_input.notna().sum())
    anchor = float(h.TVT_input.iloc[ps-1])
    idx = np.arange(0, n, STRIDE)
    if idx[-1] != n-1: idx = np.r_[idx, n-1]
    H = pyramid(h.GR.iloc[ps:].to_numpy())[idx]
    tt = tw.TVT.to_numpy(float)
    T = pyramid(tw.GR.to_numpy(float))
    baseq = anchor + center[mask][idx]
    ns, no = len(idx), len(OFF)

    # Robust multiscale local transport cost inside a narrow physical corridor.
    emit = np.empty((ns, no))
    for j, d in enumerate(OFF):
        v = np.stack([np.interp(baseq+d, tt, T[:, k], left=np.nan, right=np.nan)
                      for k in range(T.shape[1])], 1)
        r = np.abs(H-v)
        emit[:, j] = np.nanmean(np.minimum(r/.18, 3.)**2, axis=1)
        emit[~np.isfinite(emit[:, j]), j] = 50.

    # Entropic forward/backward recursion. Sequence time advances at every
    # transition and displacement can move by at most two corridor cells; TVT
    # itself may reverse because real horizontal paths cross beds repeatedly.
    # The resulting marginal mean is the
    # differentiable soft-DTW/causal OT displacement rather than a hard path.
    A = np.full((ns, no), -np.inf)
    A[0] = -(emit[0] + PRIOR*OFF**2) / TEMP
    trans = []
    for t in range(1, ns):
        mat = np.full((no, no), -np.inf)
        for a in range(no):
            legal = np.abs(OFF-OFF[a]) <= 2.
            mat[a, legal] = -TRANS*(OFF[legal]-OFF[a])**2/TEMP
        trans.append(mat)
        A[t] = -emit[t]/TEMP + logsumexp(A[t-1][:, None]+mat, axis=0)
    B = np.zeros((ns, no))
    for t in range(ns-2, -1, -1):
        B[t] = logsumexp(trans[t] + (-emit[t+1]/TEMP+B[t+1])[None, :], axis=1)
    logp = A+B
    logp -= logsumexp(logp, axis=1)[:, None]
    prob = np.exp(logp)
    shift_grid = prob @ OFF
    entropy = -np.sum(prob*np.log(prob+1e-12), axis=1)
    shift = np.interp(np.arange(n), idx, shift_grid)
    # Datum is known at the join; allow alignment influence to grow gradually.
    shift *= np.minimum(1., np.arange(n)/200.)
    pred = center[mask] + BLEND*shift
    return dict(w=w, n=n, pred=pred, shift=shift,
                mean_abs_shift=float(np.mean(np.abs(shift))),
                mean_entropy=float(np.mean(entropy)))

# Crucially generate both sets without reading y.
rows = [one(w) for w in order[:360]]
np.savez_compressed(OUT/"frozen_predictions.npz",
                    wells=np.array([r["w"] for r in rows]),
                    pred=np.array([r["pred"] for r in rows], dtype=object),
                    shift=np.array([r["shift"] for r in rows], dtype=object))

def report(which):
    rr = [r for r in rows if r["w"] in which]
    se0=se1=nn=0; wins=0
    details=[]
    for r in rr:
        m=groups==r["w"]; e0=float(np.sum((y[m]-center[m])**2));e1=float(np.sum((y[m]-r["pred"])**2))
        se0+=e0;se1+=e1;nn+=r["n"];wins += e1<e0
        details.append(dict(w=r["w"], n=r["n"], se_base=e0, se_softdtw=e1,
                            mean_abs_shift=r["mean_abs_shift"], mean_entropy=r["mean_entropy"]))
    return dict(wells=len(rr), rows=nn, baseline=float(np.sqrt(se0/nn)),
                softdtw=float(np.sqrt(se1/nn)), gain=float(np.sqrt(se0/nn)-np.sqrt(se1/nn)),
                well_wins=wins, details=details)

rp, rc = report(pilot), report(confirm)
detail = pd.DataFrame(rp.pop("details") + rc.pop("details"))
detail["split"] = ["pilot"]*120 + ["confirmation"]*240
detail.to_csv(OUT/"well_rows.csv", index=False)
summary = dict(protocol="seeded disjoint 120/240; both prediction sets frozen before target scoring",
               params=dict(offset=[-12,12,1], stride=STRIDE, scales=SCALES, temperature=TEMP,
                           transition=TRANS, boundary_prior=PRIOR, blend=BLEND),
               pilot=rp, confirmation=rc)
summary["prediction_sha256"] = hashlib.sha256((OUT/"frozen_predictions.npz").read_bytes()).hexdigest()
(OUT/"summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
