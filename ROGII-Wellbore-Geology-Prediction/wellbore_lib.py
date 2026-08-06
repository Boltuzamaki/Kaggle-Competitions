"""
Shared library for the ROGII Wellbore Geology Prediction competition.

Task: predict TVT (true vertical thickness = stratigraphic depth in the typewell
frame) for every point of a horizontal well beyond the Prediction Start (PS) point.
TVT is known up to PS (column TVT_input, which is NaN after PS).

Model target is dTVT = TVT - TVT_PS (anchored at PS -> well-agnostic, mean ~0).
Final prediction = TVT_PS + predicted dTVT.

This module is imported by both the local runner and the Kaggle notebook.
"""
import os
import glob
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------
def list_wells(split_dir):
    return sorted({os.path.basename(f).split("__")[0]
                   for f in glob.glob(os.path.join(split_dir, "*__horizontal_well.csv"))})


def load_well(split_dir, well):
    h = pd.read_csv(os.path.join(split_dir, f"{well}__horizontal_well.csv"))
    tw = pd.read_csv(os.path.join(split_dir, f"{well}__typewell.csv"))
    return h, tw


def ps_index(h):
    """First row index with no TVT_input == Prediction Start point."""
    return int(h["TVT_input"].notna().sum())


# ----------------------------------------------------------------------------
# Typewell GR(TVT) -> implied TVT candidate
# ----------------------------------------------------------------------------
def build_typewell_lookup(tw):
    t = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    return t["TVT"].to_numpy(), t["GR"].to_numpy()


def gr_implied_tvt(gr_vals, tw_tvt, tw_gr, center, window=45.0):
    """For each GR, nearest-GR TVT in the typewell within +/-window of center."""
    out = np.full(len(gr_vals), np.nan)
    if len(tw_tvt) < 2:
        return out
    m = (tw_tvt >= center - window) & (tw_tvt <= center + window)
    if m.sum() < 2:
        return out
    t, g = tw_tvt[m], tw_gr[m]
    for i, gv in enumerate(gr_vals):
        if np.isfinite(gv):
            out[i] = t[np.argmin(np.abs(g - gv))]
    return out


# ----------------------------------------------------------------------------
# Feature engineering (per well, post-PS rows only)
# ----------------------------------------------------------------------------
def make_features(h, tw, well, has_target=True):
    h = h.copy()
    ps = ps_index(h)
    if ps < 5 or ps >= len(h):
        return None

    # anchor values at PS
    md_ps = h["MD"].iloc[ps - 1]
    z_ps = h["Z"].iloc[ps - 1]
    x_ps = h["X"].iloc[ps - 1]
    y_ps = h["Y"].iloc[ps - 1]
    tvt_ps = h["TVT"].iloc[ps - 1] if has_target else h["TVT_input"].iloc[ps - 1]

    # clean / fill GR over the full well, then smooth
    gr = h["GR"].interpolate(limit_direction="both").to_numpy()
    gr_s15 = pd.Series(gr).rolling(15, center=True, min_periods=1).median().to_numpy()
    gr_s51 = pd.Series(gr).rolling(51, center=True, min_periods=1).median().to_numpy()
    gr_grad = np.gradient(gr_s15)
    gr_std = pd.Series(gr).rolling(31, center=True, min_periods=1).std().to_numpy()

    # GR-implied TVT (geosteering correlation candidate), centred at PS TVT
    tw_tvt, tw_gr = build_typewell_lookup(tw)
    gr_impl = gr_implied_tvt(gr_s15, tw_tvt, tw_gr, tvt_ps, window=45.0)
    gr_impl = pd.Series(gr_impl).rolling(25, center=True, min_periods=1).median().to_numpy()
    gr_impl_d = gr_impl - tvt_ps  # candidate dTVT

    # typewell GR slope near the anchor (steerability)
    if len(tw_tvt) > 5:
        near = (tw_tvt >= tvt_ps - 20) & (tw_tvt <= tvt_ps + 20)
        tw_gr_at_ps = np.interp(tvt_ps, tw_tvt, tw_gr)
        tw_gr_slope = (np.polyfit(tw_tvt[near], tw_gr[near], 1)[0]
                       if near.sum() > 3 else 0.0)
    else:
        tw_gr_at_ps, tw_gr_slope = np.nan, 0.0

    n = len(h)
    idx = np.arange(n)
    df = pd.DataFrame({
        "well": well,
        "row": idx,
        "d_md": h["MD"].to_numpy() - md_ps,
        "d_z": h["Z"].to_numpy() - z_ps,
        "d_x": h["X"].to_numpy() - x_ps,
        "d_y": h["Y"].to_numpy() - y_ps,
        "horiz_disp": np.hypot(h["X"].to_numpy() - x_ps, h["Y"].to_numpy() - y_ps),
        "dz_dmd": np.gradient(h["Z"].to_numpy()) / (np.gradient(h["MD"].to_numpy()) + 1e-9),
        "gr": gr,
        "gr_s15": gr_s15,
        "gr_s51": gr_s51,
        "gr_grad": gr_grad,
        "gr_std": gr_std,
        "gr_minus_ps": gr_s15 - np.interp(0, [0], [gr_s15[ps - 1]]) if ps >= 1 else gr_s15,
        "gr_impl_d": gr_impl_d,
        "gr_vs_typewell": gr_s15 - tw_gr_at_ps,
        "tw_gr_slope": tw_gr_slope,
        "tvt_ps": tvt_ps,
    })
    df["frac_along"] = (df["row"] - ps) / max(1, (n - ps))
    df["gr_minus_ps"] = df["gr_s15"] - df["gr_s15"].iloc[ps - 1]

    if has_target:
        df["d_tvt"] = h["TVT"].to_numpy() - tvt_ps

    post = df.iloc[ps:].reset_index(drop=True)
    post["id"] = post["well"] + "_" + post["row"].astype(str)
    return post


FEATURES = [
    "d_md", "d_z", "d_x", "d_y", "horiz_disp", "dz_dmd",
    "gr", "gr_s15", "gr_s51", "gr_grad", "gr_std",
    "gr_minus_ps", "gr_impl_d", "gr_vs_typewell", "tw_gr_slope",
    "frac_along",
]


def build_dataset(split_dir, wells=None, has_target=True):
    if wells is None:
        wells = list_wells(split_dir)
    parts = []
    for w in wells:
        try:
            h, tw = load_well(split_dir, w)
        except Exception:
            continue
        f = make_features(h, tw, w, has_target=has_target)
        if f is not None and len(f):
            parts.append(f)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    n_estimators=900,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=200,   # heavy regularisation -> shrink toward constant
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=1.0,
    max_depth=-1,
    verbose=-1,
)


def postprocess(pred_dtvt, clip=40.0, shrink=0.50):
    """Shrink toward the constant prior and clip to a plausible window.

    Tuned on 5-fold GroupKFold OOF: shrink=0.5, clip=40 minimised per-well RMSE
    (geosteering keeps the well in-zone, so corrections to the PS anchor are small).
    """
    return np.clip(pred_dtvt * shrink, -clip, clip)
