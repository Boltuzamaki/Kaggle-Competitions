"""
Stronger honest geosteering particle filters (numpy, vectorised over particles).
NO leak: uses only test-available data (horizontal MD/X/Y/Z/GR + typewell GR(TVT)).

Two complementary filters, ensembled:
  pf_ancc : state (pos=TVT+Z, rate).  Momentum on the structural position.
  pf_z    : state (pos=TVT, vel=dTVT/dMD).  Velocity is pulled toward the dip
            predicted from the well's Z-trajectory (beta*dz/dmd + icpt fit on the
            known section), plus a dual GR likelihood (raw + smoothed typewell).

Both run a likelihood-weighted seed ensemble. Final = weighted mean of the two.
"""
import numpy as np
import pandas as pd


# ----------------------------- momentum PF (pos=TVT+Z) -----------------------
def _pf_ancc(md, z, gr, tw_tvt, tw_gr, ls, ir, gs, N, seed,
             MOM=0.998, VN=0.002, PN=0.005, RP=0.1, RR=0.001, RESAMP=0.5):
    rng = np.random.default_rng(seed)
    pos = ls + 0.5 * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    lo, hi = tw_tvt[0] - 50, tw_tvt[-1] + 50
    out = np.empty(len(md)); ll = 0.0; pm = md[0] - 1
    for i in range(len(md)):
        dm = max(md[i] - pm, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm + PN * rng.standard_normal(N)
        tvt = np.clip(pos - z[i], lo, hi); pos = tvt + z[i]
        if np.isfinite(gr[i]):
            d = (gr[i] - np.interp(tvt, tw_tvt, tw_gr)) / gs
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


# --------------------- velocity/dip PF (pos=TVT, vel=dTVT/dMD) ----------------
def _pf_z(md, z, gr, gr_sm, tw_tvt, tw_gr, tw_gr_sm, ip, iv, beta, icpt, zsig,
         gs, N, seed, MOM=0.993, VN=0.005, PN=0.01, GR_WT=0.3,
         RP=0.2, RV=0.003, RESAMP=0.5):
    rng = np.random.default_rng(seed)
    pos = ip + 0.5 * rng.standard_normal(N)
    vel = iv + 0.02 * rng.standard_normal(N)
    w = np.ones(N) / N
    lo, hi = tw_tvt[0] - 50, tw_tvt[-1] + 50
    out = np.empty(len(md)); ll = 0.0; pm = md[0] - 1; pz = z[0] - 1
    zs = max(zsig * 2.0, 0.005)
    for i in range(len(md)):
        dm = max(md[i] - pm, 1.0)
        ve = beta * ((z[i] - pz) / dm) + icpt          # dip-predicted velocity
        vel = MOM * vel + VN * rng.standard_normal(N)
        pos = np.clip(pos + vel * dm + PN * rng.standard_normal(N), lo, hi)
        if np.isfinite(gr[i]):
            dp = (gr[i] - np.interp(pos, tw_tvt, tw_gr)) / gs
            lp = np.maximum(np.exp(-0.5 * np.minimum(dp * dp, 600)), 1e-300)
            if np.isfinite(gr_sm[i]):
                ds = (gr_sm[i] - np.interp(pos, tw_tvt, tw_gr_sm)) / (gs * 1.5)
                ls_ = np.maximum(np.exp(-0.5 * np.minimum(ds * ds, 600)), 1e-300)
                lk = (1 - GR_WT) * lp + GR_WT * ls_
            else:
                lk = lp
            avg = float((w * lk).sum()); ll += np.log(max(avg, 1e-300))
            w *= lk; s = w.sum(); w = w / s if s > 0 else np.ones(N) / N
        dv = (vel - ve) / zs                            # dip prior on velocity
        lz = np.maximum(np.exp(-0.5 * np.minimum(dv * dv, 600)), 1e-300)
        w *= lz; s = w.sum(); w = w / s if s > 0 else np.ones(N) / N
        if 1.0 / (w * w).sum() < RESAMP * N:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            vel = vel[idx] + RV * rng.standard_normal(N); w = np.ones(N) / N
        out[i] = float(np.dot(w, pos)); pm = md[i]; pz = z[i]
    return out, ll


def _gr_sigma(kn, tw_tvt, tw_gr):
    if len(kn) < 20:
        return 30.0
    at = np.interp(kn["TVT_input"].to_numpy(), tw_tvt, tw_gr)
    return float(np.clip(np.nanstd(kn["GR"].fillna(0).to_numpy() - at), 10., 60.))


def _ens(fn, args, n_seeds, scale):
    preds = []; lls = []
    for s in range(n_seeds):
        p, ll = fn(*args, seed=s)
        preds.append(p); lls.append(ll)
    lls = np.array(lls); lls -= lls.max()
    wt = np.exp(lls / scale); wt /= wt.sum()
    return (wt[:, None] * np.stack(preds, 0)).sum(0)


def pf2_predict(hw, tw, N=500, n_seeds=48, scale=5.0, w_ancc=0.5,
                use_z=True, params=None):
    p = params or {}
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
    tw_gr_sm = pd.Series(tw_gr).rolling(5, center=True, min_periods=1).mean().to_numpy()

    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    out = hw["TVT_input"].to_numpy(float).copy()
    if len(ev) == 0 or len(kn) < 5:
        return out
    last = kn.iloc[-1]; last_tvt = float(last["TVT_input"]); last_z = float(last["Z"])
    gs = _gr_sigma(kn, tw_tvt, tw_gr)

    gr_all = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).to_numpy()
    gr_sm_all = hw["GR"].rolling(5, center=True, min_periods=1).mean() \
                  .interpolate(limit_direction="both").fillna(tw_gr.mean()).to_numpy()
    idx = ev.index.to_numpy()
    md = np.concatenate([[float(last["MD"])], ev["MD"].to_numpy(float)])
    z = np.concatenate([[last_z], ev["Z"].to_numpy(float)])
    gr = np.concatenate([[np.nan], gr_all[idx]])
    gr_sm = np.concatenate([[np.nan], gr_sm_all[idx]])

    # momentum init rate
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy()); dz = np.diff(tail["Z"].to_numpy())
    dmv = np.diff(tail["MD"].to_numpy()); m = dmv > 0
    ir = float(np.median((dt + dz)[m] / dmv[m])) if m.sum() >= 3 else 0.0
    ls = last_tvt + last_z
    pa = {k: p[k] for k in ("MOM", "VN", "PN", "RP", "RR") if k in p}
    pred_a = _ens(_pf_ancc, (md, z, gr, tw_tvt, tw_gr, ls, ir, gs, N),
                  n_seeds, scale) if True else None

    if use_z:
        dzk = np.diff(kn["Z"].to_numpy()); dvt = np.diff(kn["TVT_input"].to_numpy())
        dmk = np.diff(kn["MD"].to_numpy()); m2 = dmk > 0
        if m2.sum() >= 10:
            vz = dzk[m2] / dmk[m2]; vt = dvt[m2] / dmk[m2]
            A = np.column_stack([vz, np.ones_like(vz)])
            c, *_ = np.linalg.lstsq(A, vt, rcond=None)
            beta, icpt = float(c[0]), float(c[1])
            zsig = max(float(np.std(vt - (c[0] * vz + c[1]))), 0.001)
        else:
            beta, icpt, zsig = -1.0, 0.0, 0.1
        t2 = kn.tail(20); dvt2 = np.diff(t2["TVT_input"].to_numpy())
        dm2 = np.diff(t2["MD"].to_numpy()); m3 = dm2 > 0
        iv = float(np.median(dvt2[m3] / dm2[m3])) if m3.sum() >= 3 else 0.0
        pred_z = _ens(_pf_z, (md, z, gr, gr_sm, tw_tvt, tw_gr, tw_gr_sm,
                              last_tvt, iv, beta, icpt, zsig, gs, N), n_seeds, scale)
        est = w_ancc * pred_a + (1 - w_ancc) * pred_z
    else:
        est = pred_a
    out[idx] = est[1:]
    return out
