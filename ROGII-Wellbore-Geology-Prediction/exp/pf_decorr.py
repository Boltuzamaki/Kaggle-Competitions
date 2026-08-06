"""
Decorrelated particle-filter ensemble (variance reduction).

Standard ensemble = N seeds sharing the SAME dynamics -> errors correlated, so
averaging barely reduces variance. Here each candidate draws its own dynamics
(process noise, GR-likelihood width, resample threshold, init spread, momentum
jitter) from tight ranges around the tuned optimum, so candidate errors are more
independent and their mean reduces the seed-noise floor.

Per the writeup ("The Wiggle Is Free, the Trend Is the Wall"), this variance
reduction is the real, transferable gain over an isolated PF. Only test-time data
is used (no leak). Combine = likelihood-weighted mean (high scale ~ simple mean).
"""
import numpy as np
import pandas as pd


def _run_pf(md, z, gr, tw_tvt, tw_gr, ls, ir, gs, N, seed,
            MOM, VN, PN, RP, RR, RESAMP, init_spr, student_nu=0.0,
            prev_md_init=None):
    rng = np.random.default_rng(seed)
    pos = ls + init_spr * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    lo, hi = tw_tvt[0] - 60, tw_tvt[-1] + 60
    out = np.empty(len(md)); ll = 0.0
    pm = md[0] - 1 if prev_md_init is None else float(prev_md_init)
    for i in range(len(md)):
        dm = max(md[i] - pm, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm + PN * rng.standard_normal(N)
        tvt = np.clip(pos - z[i], lo, hi); pos = tvt + z[i]
        if np.isfinite(gr[i]):
            d = (gr[i] - np.interp(tvt, tw_tvt, tw_gr)) / gs
            if student_nu > 0:
                # Robust measurement model: preserve alternatives when a GR
                # station is an outlier or the lateral/typewell amplitudes differ.
                lk = np.power(1.0 + d*d/student_nu,
                              -0.5*(student_nu+1.0))
                lk = np.maximum(lk, 1e-300)
            else:
                lk = np.maximum(np.exp(-0.5 * np.minimum(d * d, 600)), 1e-300)
            avg = float((w * lk).sum()); ll += np.log(max(avg, 1e-300))
            w *= lk; s = w.sum(); w = w / s if s > 0 else np.ones(N) / N
        if 1.0 / (w * w).sum() < RESAMP * N:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N); w = np.ones(N) / N
        out[i] = float(np.dot(w, pos - z[i])); pm = md[i]
    return out, ll


def pf_predict_decorr(hw, tw, n_particles=400, n_seeds=200, scale=10.0,
                      decorrelate=True, seed0=1000, gs_mult=1.0,
                      gs_floor=0.0, student_nu=0.0):
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    out = hw["TVT_input"].to_numpy(float).copy()
    if len(ev) == 0 or len(kn) < 5:
        return out
    last = kn.iloc[-1]; last_tvt = float(last["TVT_input"]); last_z = float(last["Z"])
    at = np.interp(kn["TVT_input"].to_numpy(), tw_tvt, tw_gr)
    gs0 = max(float(np.clip(np.nanstd(kn["GR"].fillna(0).to_numpy() - at),
                            10., 60.)), float(gs_floor)) * gs_mult
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy()); dz = np.diff(tail["Z"].to_numpy())
    dmv = np.diff(tail["MD"].to_numpy()); m = dmv > 0
    ir = float(np.median((dt + dz)[m] / dmv[m])) if m.sum() >= 3 else 0.0
    ls = last_tvt + last_z

    gr_all = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).to_numpy()
    idx = ev.index.to_numpy()
    md = np.concatenate([[float(last["MD"])], ev["MD"].to_numpy(float)])
    z = np.concatenate([[last_z], ev["Z"].to_numpy(float)])
    gr = np.concatenate([[np.nan], gr_all[idx]])

    preds = np.empty((n_seeds, len(md))); lls = np.empty(n_seeds)
    for s in range(n_seeds):
        if decorrelate:
            pr = np.random.default_rng(seed0 + s)   # param RNG (separate from particle RNG)
            MOM = 0.998
            VN = 0.002 * pr.uniform(0.93, 1.08)
            PN = 0.005 * pr.uniform(0.88, 1.15)
            RESAMP = pr.uniform(0.47, 0.55)
            gs = gs0 * pr.uniform(0.92, 1.12)
            init_spr = pr.uniform(1.6, 2.4)
            RP = 0.1 * pr.uniform(0.85, 1.2)
        else:
            MOM, VN, PN, RESAMP, gs, init_spr, RP = 0.998, 0.002, 0.005, 0.5, gs0, 2.0, 0.1
        preds[s], lls[s] = _run_pf(md, z, gr, tw_tvt, tw_gr, ls, ir, gs,
                                   n_particles, s, MOM, VN, PN, RP, 0.001,
                                   RESAMP, init_spr, student_nu)
    lls -= lls.max()
    wts = np.exp(lls / scale); wts /= wts.sum()
    est = (wts[:, None] * preds).sum(0)[1:]
    out[idx] = est
    return out
