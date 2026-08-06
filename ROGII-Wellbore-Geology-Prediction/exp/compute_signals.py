"""
Compute honest domain signals per well (post-PS rows) for the LightGBM residual
stack. Target-free (no hidden TVT used). Saves parquet for train and test.

Signals: PF path (delta), beam path (delta), multi-scale NCC vs typewell,
self-correlation vs the lateral's own known section, Q-3D tortuosity, relative
trajectory geometry, GR texture, typewell GR residuals at PF/anchor baselines.
Target: dTVT = TVT - last_known_TVT (train only).
"""
import sys, os, glob, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H); sys.path.insert(0, os.path.dirname(_H))
from pf_tracker import pf_predict
from beam_tracker import beam_predict
from wellbore_lib import ps_index

ANCH_OFF = [-20, -10, -5, 0, 5, 10, 20]


def multiscale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    for hw in hws:
        win = 2 * hw + 1; nk = len(kgr); nh = len(hgr)
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1] if len(ktvt) else 0.0, np.float32), np.zeros(nh, np.float32)))
            continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().to_numpy(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().to_numpy(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32); M = len(sts)
        if M == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]]
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode="edge")
        Hm = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]]
        Hn = (Hm - Hm.mean(1, keepdims=True)) / (Hm.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win
        best = ncc.argmax(1); score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32), score))
    return out


def tortuosity(x, y, z, md, win=50):
    d = np.stack([np.gradient(x), np.gradient(y), np.gradient(z)], 1)
    dm = np.gradient(md)[:, None] + 1e-9
    t = d / dm
    n = np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    u = t / n
    kappa = np.linalg.norm(np.gradient(u, axis=0), axis=1)   # curvature proxy
    return pd.Series(kappa).rolling(win, min_periods=1).sum().to_numpy(np.float32)


def build_well(path, is_train):
    wid = os.path.basename(path).split("__")[0]
    base = os.path.dirname(path)
    try:
        h = pd.read_csv(path)
        tw = pd.read_csv(f"{base}/{wid}__typewell.csv")
    except Exception:
        return None
    ps = ps_index(h)
    if ps < 10 or ps >= len(h) - 3:
        return None
    if is_train and "TVT" not in h.columns:
        return None
    tw_s = tw.sort_values("TVT"); tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
    if len(tw_tvt) < 3:
        return None

    tps = float(h["TVT_input"].iloc[ps - 1])
    md = h["MD"].to_numpy(float); x = h["X"].to_numpy(float); y = h["Y"].to_numpy(float); z = h["Z"].to_numpy(float)
    gr_all = h["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).to_numpy(float)
    ev = np.arange(ps, len(h)); nev = len(ev)

    pf = pf_predict(h, tw, n_particles=300, n_seeds=24, scale=12.0)
    bm = beam_predict(h, tw)
    ktvt = h["TVT_input"].to_numpy(float)[:ps]
    kgr = gr_all[:ps]; hgr = gr_all[ps:]
    ncc = multiscale_ncc(kgr, ktvt, hgr)
    tort = tortuosity(x, y, z, md)

    md_ps, x_ps, y_ps, z_ps = md[ps - 1], x[ps - 1], y[ps - 1], z[ps - 1]
    gr_s = pd.Series(gr_all)
    f = {
        "well": wid, "id": [f"{wid}_{i}" for i in ev],
        "last_known_tvt": tps,
        "pf_d": (pf[ev] - tps).astype(np.float32),
        "beam_d": (bm[ev] - tps).astype(np.float32),
        "pf_vs_beam": (pf[ev] - bm[ev]).astype(np.float32),
        "ncc8_d": np.clip(ncc[0][0] - tps, -80, 80), "ncc8_s": ncc[0][1],
        "ncc15_d": np.clip(ncc[1][0] - tps, -80, 80), "ncc15_s": ncc[1][1],
        "ncc25_d": np.clip(ncc[2][0] - tps, -80, 80), "ncc25_s": ncc[2][1],
        "ncc_mean_d": np.clip((ncc[0][0] + ncc[1][0] + ncc[2][0]) / 3 - tps, -80, 80),
        "tort": tort[ev],
        "d_md": (md[ev] - md_ps).astype(np.float32),
        "d_z": (z[ev] - z_ps).astype(np.float32),
        "d_xy": np.hypot(x[ev] - x_ps, y[ev] - y_ps).astype(np.float32),
        "dz_dmd": (np.gradient(z) / (np.gradient(md) + 1e-9))[ev].astype(np.float32),
        "gr": gr_all[ev].astype(np.float32),
        "gr_m21": gr_s.rolling(21, center=True, min_periods=1).mean().to_numpy()[ev].astype(np.float32),
        "gr_s21": gr_s.rolling(21, center=True, min_periods=1).std().fillna(0).to_numpy()[ev].astype(np.float32),
        "gr_d1": gr_s.diff().fillna(0).to_numpy()[ev].astype(np.float32),
        "frac": (np.arange(nev) / max(nev - 1, 1)).astype(np.float32),
    }
    for o in ANCH_OFF:
        f[f"twres_pf{o}"] = (gr_all[ev] - np.interp(pf[ev] + o, tw_tvt, tw_gr)).astype(np.float32)
    if is_train:
        f["target"] = (h["TVT"].to_numpy(float)[ev] - tps).astype(np.float32)
    return pd.DataFrame(f)


def run(split, is_train, out):
    paths = sorted(glob.glob(f"data/{split}/*__horizontal_well.csv"))
    t0 = time.time()
    res = Parallel(n_jobs=10)(delayed(build_well)(p, is_train) for p in paths)
    df = pd.concat([r for r in res if r is not None], ignore_index=True)
    df.to_pickle(out)
    print(f"{split}: {df['well'].nunique()} wells, {len(df)} rows, {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    os.makedirs("exp/signals", exist_ok=True)
    run("test", False, "exp/signals/test.pkl")
    run("train", True, "exp/signals/train.pkl")
    print("DONE", flush=True)
