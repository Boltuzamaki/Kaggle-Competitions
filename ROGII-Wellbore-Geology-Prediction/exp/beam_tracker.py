"""Beam-search geosteering tracker (numba JIT). Complements the particle filter:
finds a low-cost monotone-ish path through the typewell GR(TVT) grid that matches
the horizontal GR signature. Ensembled over several stiffness configs."""
import numpy as np
from numba import njit
from scipy.signal import savgol_filter

# (beam_size, move_cost, err_scale, savgol_radius)
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2), (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5), (20, 4.0, 36.0, 3), (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
]


@njit(cache=True)
def _beam(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS * 6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.0; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64); cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt:
                    continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni:
                        fnd = ci; break
                if fnd >= 0:
                    if tot < cC[fnd]:
                        cC[fnd] = tot; cP[fnd] = bi
                elif nc < MAX:
                    cI[nc] = ni; cC[nc] = tot; cP[nc] = bi; nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]:
                    mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]:
            best = b
    path = np.zeros(n, np.int64); b = best
    for s in range(n - 1, -1, -1):
        path[s] = hI[s, b]; b = hP[s, b]
    return path


def beam_predict(hw, tw):
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    out = hw["TVT_input"].to_numpy(float).copy()
    if len(ev) == 0:
        return out
    last_tvt = float(kn.iloc[-1]["TVT_input"])
    gr_all = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).to_numpy(float)
    hgr = gr_all[ev.index.to_numpy()]
    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))
    paths = []
    for BS, mc, es, r in BEAM_CONFIGS:
        if r > 0 and len(hgr) > max(3, 2 * r + 1):
            win = min(2 * r + 1, len(hgr) if len(hgr) % 2 == 1 else len(hgr) - 1)
            sgr = savgol_filter(hgr, win, min(2, win - 1))
        else:
            sgr = hgr.copy()
        paths.append(tw_tvt[_beam(sgr.astype(np.float64), tw_gr, si, BS, float(mc), float(es))])
    out[ev.index.to_numpy()] = np.stack(paths, 0).mean(0)
    return out
