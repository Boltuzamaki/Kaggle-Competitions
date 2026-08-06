"""
Build per-well feature sequences (full length) + fixed 5-fold GroupKFold split.
Cached to exp/cache/ so every experiment (tabular & sequence) reuses identical
features and folds. Sequences are stored at STRIDE-ft resolution; predictions are
interpolated back to the 1-ft grid at scoring time.

Channels (per position):
  0 gr_norm         robust-normalised smoothed GR
  1 gr_grad         GR gradient (scaled)
  2 gr_std          local GR std (scaled)
  3 d_z             elevation change from PS / 50
  4 dz_dmd          local dZ/dMD * 100
  5 pos             (row - ps) / len_post  (negative before PS)
  6 gr_impl_d       typewell GR->TVT match candidate / 20
  7 gr_vs_tw        GR - typewell GR@PS / 20
  8 tvt_in_rel      (TVT_input - TVT_PS)/20 where known, else 0
  9 known           1 before PS, 0 after
 10 d_md            MD change from PS / 1000
Target: dTVT = (TVT - TVT_PS)   (feet; NN training scales by /20 internally)
"""
import os, sys, glob, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wellbore_lib import (list_wells, load_well, ps_index,
                          build_typewell_lookup, gr_implied_tvt)

STRIDE = 2
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)
N_FOLDS = 5
SEED = 42
CHANNELS = ["gr_norm", "gr_grad", "gr_std", "d_z", "dz_dmd", "pos",
            "gr_impl_d", "gr_vs_tw", "tvt_in_rel", "known", "d_md"]


def build_well_sequence(h, tw, well):
    ps = ps_index(h)
    if ps < 5 or ps >= len(h):
        return None
    n = len(h)
    md = h["MD"].to_numpy(); z = h["Z"].to_numpy()
    md_ps, z_ps = md[ps - 1], z[ps - 1]
    tvt = h["TVT"].to_numpy() if "TVT" in h.columns else None
    tvt_ps = (tvt[ps - 1] if tvt is not None else h["TVT_input"].iloc[ps - 1])

    gr = h["GR"].interpolate(limit_direction="both").to_numpy()
    gr_s15 = pd.Series(gr).rolling(15, center=True, min_periods=1).median().to_numpy()
    gr_grad = np.gradient(gr_s15)
    gr_std = pd.Series(gr).rolling(31, center=True, min_periods=1).std().fillna(0).to_numpy()
    med = np.nanmedian(gr_s15); iqr = np.subtract(*np.nanpercentile(gr_s15, [75, 25])) + 1e-6

    tw_tvt, tw_gr = build_typewell_lookup(tw)
    gr_impl = gr_implied_tvt(gr_s15, tw_tvt, tw_gr, tvt_ps, window=45.0)
    gr_impl = pd.Series(gr_impl).rolling(25, center=True, min_periods=1).median().to_numpy()
    gr_impl_d = np.nan_to_num(gr_impl - tvt_ps)
    tw_gr_at_ps = np.interp(tvt_ps, tw_tvt, tw_gr) if len(tw_tvt) > 1 else np.nanmedian(gr_s15)

    tvt_in = h["TVT_input"].to_numpy()
    known = np.isfinite(tvt_in).astype(np.float32)
    tvt_in_rel = np.where(known > 0, np.nan_to_num(tvt_in - tvt_ps), 0.0)

    dz_dmd = np.gradient(z) / (np.gradient(md) + 1e-9)
    pos = (np.arange(n) - ps) / max(1, n - ps)

    X = np.stack([
        (gr_s15 - med) / iqr,
        gr_grad / 5.0,
        gr_std / 20.0,
        (z - z_ps) / 50.0,
        dz_dmd * 100.0,
        pos,
        gr_impl_d / 20.0,
        (gr_s15 - tw_gr_at_ps) / 20.0,
        tvt_in_rel / 20.0,
        known,
        (md - md_ps) / 1000.0,
    ], axis=1).astype(np.float32)                       # (n, C)

    y = (tvt - tvt_ps).astype(np.float32) if tvt is not None else np.zeros(n, np.float32)
    post = (np.arange(n) >= ps)
    rows = np.arange(n)

    sl = slice(0, n, STRIDE)
    return dict(well=well, X=X[sl], y=y[sl], post=post[sl], rows=rows[sl],
                tvt_ps=float(tvt_ps), n=n, ps=ps)


def build_split(split_dir, has_target=True):
    seqs = []
    for w in list_wells(split_dir):
        try:
            h, tw = load_well(split_dir, w)
        except Exception:
            continue
        s = build_well_sequence(h, tw, w)
        if s is not None:
            seqs.append(s)
    return seqs


def assign_folds(seqs):
    rng = np.random.RandomState(SEED)
    wells = sorted(s["well"] for s in seqs)
    order = rng.permutation(len(wells))
    fold_of = {wells[order[i]]: i % N_FOLDS for i in range(len(wells))}
    return fold_of


if __name__ == "__main__":
    import time
    DATA = "data"
    t0 = time.time()
    print("Building train sequences (stride=%d)..." % STRIDE)
    train = build_split(f"{DATA}/train", has_target=True)
    print("  %d wells in %.0fs" % (len(train), time.time() - t0))
    folds = assign_folds(train)
    for s in train:
        s["fold"] = folds[s["well"]]
    test = build_split(f"{DATA}/test", has_target=False)
    print("  %d test wells" % len(test))

    with open(os.path.join(CACHE, "train_seq.pkl"), "wb") as f:
        pickle.dump(train, f, protocol=4)
    with open(os.path.join(CACHE, "test_seq.pkl"), "wb") as f:
        pickle.dump(test, f, protocol=4)
    lengths = [len(s["y"]) for s in train]
    print("seq len (strided): min %d med %d max %d" %
          (min(lengths), int(np.median(lengths)), max(lengths)))
    print("channels:", CHANNELS)
    print("cached to", CACHE)
