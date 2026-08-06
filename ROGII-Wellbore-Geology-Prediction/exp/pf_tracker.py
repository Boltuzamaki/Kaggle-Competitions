"""
Particle-filter geosteering tracker (numpy, vectorised over particles).

State per particle: (pos, rate) where pos = TVT + Z  ("structural" position that
is ~constant when geology is flat), rate = d(pos)/dMD. At each eval step we
propagate with momentum, convert to TVT = pos - Z, score against the typewell
GR(TVT) curve via a Gaussian likelihood, resample, and take the weighted-mean TVT.

Predictions are made only for the eval rows (TVT_input is NaN). A seed ensemble
is combined with weights ∝ exp(loglik/scale). This is the core signal that gets
geosteering solutions to single-digit RMSE.
"""
import numpy as np

# tuned defaults (close to strong public solutions)
MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5


def _run_pf(md, z, gr, tw_tvt, tw_gr, last_tvt, last_Z, ir, gs,
            n_particles=500, seed=0):
    rng = np.random.default_rng(seed)
    N = n_particles
    ls = last_tvt + last_Z
    pos = ls + 2.0 * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    lo, hi = tw_tvt[0] - 100, tw_tvt[-1] + 100
    out = np.empty(len(md)); loglik = 0.0
    prev_md = md[0] - 1.0
    for i in range(len(md)):
        dm = max(md[i] - prev_md, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm + PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - z[i], lo, hi)
        pos = tvt_p + z[i]
        if np.isfinite(gr[i]):
            eg = np.interp(tvt_p, tw_tvt, tw_gr)
            d = (gr[i] - eg) / gs
            lk = np.maximum(np.exp(-0.5 * np.minimum(d * d, 600.0)), 1e-300)
            avg = float((w * lk).sum()); loglik += np.log(max(avg, 1e-300))
            w = w * lk; s = w.sum(); w = w / s if s > 0 else np.ones(N) / N
        neff = 1.0 / (w * w).sum()
        if neff < RESAMP * N:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N
        out[i] = float(np.dot(w, pos - z[i]))
        prev_md = md[i]
    return out, loglik


def pf_predict(hw, tw, n_particles=500, n_seeds=48, scale=5.0):
    """Return a full-length TVT array (known rows kept, eval rows predicted)."""
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)

    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    out = hw["TVT_input"].to_numpy(float).copy()
    if len(ev) == 0:
        return out

    last = kn.iloc[-1]
    last_tvt = float(last["TVT_input"]); last_Z = float(last["Z"])
    # GR noise sigma from known section
    tw_at_k = np.interp(kn["TVT_input"].to_numpy(), tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).to_numpy() - tw_at_k), 10., 60.))
    # initial structural rate from last 30 known points
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy()); dz = np.diff(tail["Z"].to_numpy())
    dmv = np.diff(tail["MD"].to_numpy()); m = dmv > 0
    ir = float(np.median((dt + dz)[m] / dmv[m])) if m.sum() >= 3 else 0.0

    gr_all = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).to_numpy(float)
    md = ev["MD"].to_numpy(float); z = ev["Z"].to_numpy(float)
    gr = gr_all[ev.index.to_numpy()]
    # prepend last known so momentum starts from the right place
    md = np.concatenate([[float(last["MD"])], md])
    z = np.concatenate([[last_Z], z])
    gr = np.concatenate([[np.nan], gr])

    preds = np.empty((n_seeds, len(md))); liks = np.empty(n_seeds)
    for s in range(n_seeds):
        preds[s], liks[s] = _run_pf(md, z, gr, tw_tvt, tw_gr, last_tvt, last_Z,
                                    ir, gs, n_particles, seed=s)
    liks -= liks.max()
    wts = np.exp(liks / scale); wts /= wts.sum()
    est = (wts[:, None] * preds).sum(0)[1:]        # drop the prepended known row
    out[ev.index.to_numpy()] = est
    return out
