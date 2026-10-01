"""Feature construction for PS S6E9.

The competition data is synthetic, and the generator left three separable
fingerprints on `Annual_Income_USD`:

  1. DIGITS      - the base-10 digits of income carry target signal at 16-25x the
                   binomial null corridor. Absent from the real survey, so it is an
                   artifact of the generator, not of the world.
  2. JAGGEDNESS  - the income -> P(buy) curve is not smooth. Binned at $1000 it
                   deviates from its own local trend by up to 32 sigma, so income
                   needs encoding at several resolutions at once.
  3. HARD RULES  - income >= 170537 is deterministically a buyer (393/393 training
                   rows, and it overrides both gates); $38-42k is deterministically
                   not (0.16%); $30000 is a mode-collapse spike holding 9.2% of rows.

Two stages:
  base_frame()     fold-independent columns
  fold_transform() fold-DEPENDENT columns (target encodings), rebuilt inside every
                   fold from that fold's training rows only.
"""
import numpy as np
import pandas as pd

import common as C

CLIFF = 170537          # lowest income of the perfect all-Yes run
DEAD_LO, DEAD_HI = 38000, 42000
INCOME_SPIKE = 30000
COMMUTE_FLOOR = 5.0

# Target-encoding keys: income and commute at several resolutions, each at three
# smoothing strengths so the model can pick its own bias/variance trade-off.
# Target encoding is applied broadly: every raw column, every digit column and every
# multi-resolution income/commute key, each at three smoothing strengths. The digit
# columns matter most - a target encoding hands the model the digit artifact directly
# as a number instead of making it find the splits.
TE_SMOOTH = [5.0, 20.0, 100.0]
MULTI_KEYS = ["inc_exact", "inc_100", "inc_1000", "km_exact", "km_int"]
FREQ_KEYS = MULTI_KEYS


def _keys(d):
    """Integer keys used for both target and frequency encoding."""
    inc = d["Annual_Income_USD"].values.astype(np.int64)
    km10 = np.rint(d["Daily_Commute_km"].values * 10).astype(np.int64)
    return {
        "inc_exact": inc,
        "inc_100": inc // 100,
        "inc_1000": inc // 1000,
        "km_exact": km10,
        "km_int": km10 // 10,
    }


def _digits(d):
    """Base-10 digits of every genuinely continuous column, 10^-1 .. 10^4.

    Constant columns are dropped by the caller, so positions that carry no
    resolution for a given column cost nothing.
    """
    out = {}
    for col, scale in [("Annual_Income_USD", 1), ("Daily_Commute_km", 10), ("Age", 1)]:
        v = np.rint(d[col].values * scale).astype(np.int64)
        for k in range(0, 6):
            out[f"{col[:3].lower()}_d{k}"] = ((v // 10 ** k) % 10).astype(np.int8)
    return out


def _orig_means():
    """Anchor: the purchase rate each value shows in the REAL 10k survey.

    This is external data, carries no competition label, and gives the model the
    smooth real-world curve alongside the generator's jagged one.
    """
    import os
    p = os.path.join(C.DATA, "original", "EV_Adoption_and_Range_Anxiety_Dataset.csv")
    if not os.path.exists(p):
        return None, None
    o = pd.read_csv(p)
    oy = (o["Will_Buy_EV"] == "Yes").astype(int)
    maps = {}
    for col in ["Age", "Number_of_Cars_Owned", "Charging_Stations_Near_Home",
                "Charging_Stations_Near_Work", "Environmental_Concern_Level",
                "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
                "Subsidy_Available", "Range_Anxiety_Level"]:
        maps[col] = oy.groupby(o[col]).mean()
    maps["_inc5k"] = oy.groupby((o["Annual_Income_USD"] // 5000)).mean()
    maps["_km5"] = oy.groupby((o["Daily_Commute_km"] // 5)).mean()
    return maps, float(oy.mean())


def base_frame(drop_cars=True):
    tr, te, y = C.load_raw()
    test_id = te[C.ID].values
    raw_tr = tr.drop(columns=[C.ID])
    raw_te = te.drop(columns=[C.ID])

    X = C.add_features(raw_tr, level="raw")      # ordinal encoding only
    Xt = C.add_features(raw_te, level="raw")

    for d, src in ((X, raw_tr), (Xt, raw_te)):
        inc = src["Annual_Income_USD"].values
        km = src["Daily_Commute_km"].values
        # --- hard generator rules -------------------------------------------
        d["is_income_cliff"] = (inc >= CLIFF).astype(np.int8)
        d["is_dead_zone"] = ((inc >= DEAD_LO) & (inc <= DEAD_HI)).astype(np.int8)
        d["is_income_spike"] = (inc == INCOME_SPIKE).astype(np.int8)
        d["is_commute_floor"] = (km == COMMUTE_FLOOR).astype(np.int8)
        d["dist_to_cliff"] = (CLIFF - inc).astype(np.float32)
        # --- multi-resolution income / commute keys --------------------------
        for k, v in _keys(src).items():
            d["k_" + k] = v
        # --- digits ----------------------------------------------------------
        for k, v in _digits(src).items():
            d[k] = v
        d["log_income"] = np.log1p(inc).astype(np.float32)
        d["is_env_hater"] = (src["Environmental_Concern_Level"] == 1).astype(np.int8)

    # --- real-survey anchors -------------------------------------------------
    maps, prior = _orig_means()
    if maps is not None:
        for d, src in ((X, raw_tr), (Xt, raw_te)):
            for col, m in maps.items():
                if col.startswith("_"):
                    continue
                d[f"org_{col}"] = src[col].map(m).fillna(prior).astype(np.float32)
            d["org_inc5k"] = (src["Annual_Income_USD"] // 5000).map(
                maps["_inc5k"]).fillna(prior).astype(np.float32)
            d["org_km5"] = (src["Daily_Commute_km"] // 5).map(
                maps["_km5"]).fillna(prior).astype(np.float32)

    # --- frequency encoding over train+test together -------------------------
    # Unsupervised, so pooling the two sets is legitimate and gives a sharper
    # estimate of how often each exact income value was emitted.
    ktr, kte = _keys(raw_tr), _keys(raw_te)
    for name in FREQ_KEYS:
        both = pd.Series(np.concatenate([ktr[name], kte[name]]))
        vc = both.value_counts()
        X[f"freq_{name}"] = pd.Series(ktr[name]).map(vc).astype(np.float32).values
        Xt[f"freq_{name}"] = pd.Series(kte[name]).map(vc).astype(np.float32).values

    if drop_cars:
        X = X.drop(columns=["Number_of_Cars_Owned"])
        Xt = Xt.drop(columns=["Number_of_Cars_Owned"])

    # drop columns with no resolution (e.g. sub-unit digits of an integer column)
    const = [c for c in X.columns if X[c].nunique() <= 1 and Xt[c].nunique() <= 1]
    if const:
        X = X.drop(columns=const); Xt = Xt.drop(columns=const)

    Xt = Xt[X.columns]
    return X, Xt, y, test_id


WIN_INC = [2, 5, 10, 25, 50, 200]      # half-widths in *distinct income values*
WIN_KM = [1, 3, 10]
WIN_PRIOR = 10.0
# Measured on folds 0-1: window encodings cost -0.00008 against the same matrix
# without them (0.945401 vs 0.945483). The three smoothing strengths of the plain
# target encodings already supply multi-scale smoothing, so the windows are
# redundant. Kept, switched off, as evidence rather than deleted.
USE_WINDOWS = False


def _window_encode(ktr, ytr, kos, widths, prior):
    """Rolling target mean over neighbouring DISTINCT key values.

    Exact-value target encoding is noisy (~50 rows per income) and fixed bins
    (//100, //1000) put arbitrary walls through a jagged curve. A window centred on
    each distinct value smooths at a chosen scale without boundary artifacts, and
    several widths together let the model pick the resolution it wants.

    Unseen values are mapped to their nearest observed value via searchsorted, so
    test income never falls off the end of the table.
    """
    g = pd.DataFrame({"k": ktr, "y": ytr}).groupby("k")["y"].agg(["sum", "count"])
    g = g.sort_index()
    vals = g.index.values.astype(np.float64)
    ssum = g["sum"].values.astype(np.float64)
    scnt = g["count"].values.astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(ssum)])
    cc = np.concatenate([[0.0], np.cumsum(scnt)])
    n = len(vals)

    # nearest observed value for every row we need to encode
    idx_of = []
    for k in kos:
        j = np.searchsorted(vals, k)
        j = np.clip(j, 0, n - 1)
        jm = np.clip(j - 1, 0, n - 1)
        pick = np.where(np.abs(vals[jm] - k) <= np.abs(vals[j] - k), jm, j)
        idx_of.append(pick)

    out = {}
    for w in widths:
        lo = np.maximum(np.arange(n) - w, 0)
        hi = np.minimum(np.arange(n) + w + 1, n)
        wsum = cs[hi] - cs[lo]
        wcnt = cc[hi] - cc[lo]
        enc = (wsum + prior * float(ytr.mean())) / (wcnt + prior)
        out[w] = [enc[i].astype(np.float32) for i in idx_of]
    return out


def te_columns(X):
    """Every column that gets target-encoded."""
    digits = [c for c in X.columns if len(c) > 4 and c[3:5] == "_d" and c[:3] in ("ann", "dai", "age")]
    keys = [c for c in X.columns if c.startswith("k_")]
    base = [c for c in C.CAT_COLS + C.NUM_COLS if c in X.columns]
    return base + digits + keys


def fold_transform(Xtr, ytr, others, te=True, counts=False):
    """others: frames to receive encodings fitted on (Xtr, ytr) only."""
    Xtr = Xtr.copy()
    others = [o.copy() for o in others]
    if not te:
        return Xtr, others

    for col in te_columns(Xtr):
        ktr = Xtr[col].values
        kos = [o[col].values for o in others]
        etr, eos = C.oof_target_encode_multi(ktr, ytr, kos, TE_SMOOTH)
        for sm in TE_SMOOTH:
            Xtr[f"te_{col}_{sm:g}"] = etr[sm]
            for o, e in zip(others, eos[sm]):
                o[f"te_{col}_{sm:g}"] = e

    # --- neighbourhood (window) encodings -------------------------------------
    # Built from the fold's training rows only. The training copy is encoded with an
    # inner KFold, exactly as the plain target encodings are, so no row contributes
    # its own label to its own window.
    from sklearn.model_selection import KFold
    for col, widths in ((("k_inc_exact", WIN_INC), ("k_km_exact", WIN_KM))
                        if USE_WINDOWS else ()):
        ktr = Xtr[col].values
        kos = [o[col].values for o in others]
        # apply-side: one table fitted on all of the fold's training rows
        outs = _window_encode(ktr, ytr, kos, widths, WIN_PRIOR)
        for w in widths:
            for o, e in zip(others, outs[w]):
                o[f"win_{col}_{w}"] = e
        # train-side: inner out-of-fold
        acc = {w: np.zeros(len(ktr), dtype=np.float32) for w in widths}
        for a, b in KFold(5, shuffle=True, random_state=C.SEED).split(ktr):
            inner = _window_encode(ktr[a], ytr[a], [ktr[b]], widths, WIN_PRIOR)
            for w in widths:
                acc[w][b] = inner[w][0]
        for w in widths:
            Xtr[f"win_{col}_{w}"] = acc[w]
    return Xtr, others
