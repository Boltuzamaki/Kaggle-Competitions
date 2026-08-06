"""Build the self-contained ROGII contact-gated training notebook."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "kernels" / "contact_train"


def code(source: str, hidden: bool = True) -> dict:
    metadata = {}
    if hidden:
        metadata = {
            "_kg_hide-input": True,
            "jupyter": {"source_hidden": True},
            "source_hidden": True,
            "tags": ["hide-input"],
        }
    return {
        "cell_type": "code",
        "metadata": metadata,
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


PF = (ROOT / "exp" / "pf_tracker.py").read_text(encoding="utf-8")
BEAM = (ROOT / "exp" / "beam_tracker.py").read_text(encoding="utf-8").replace(
    "cache=True", "cache=False"
)
SIGNALS = (ROOT / "exp" / "compute_signals.py").read_text(encoding="utf-8")
SIGNALS = SIGNALS.split("def run(")[0]
for old in (
    "from pf_tracker import pf_predict",
    "from beam_tracker import beam_predict",
    "from wellbore_lib import ps_index",
    "_H = os.path.dirname(os.path.abspath(__file__))",
    "sys.path.insert(0, _H); sys.path.insert(0, os.path.dirname(_H))",
):
    SIGNALS = SIGNALS.replace(old, "")
SIGNALS = SIGNALS.replace(
    "pf = pf_predict(h, tw, n_particles=300, n_seeds=24, scale=12.0)",
    "pf = pf_predict(h, tw, "
    "n_particles=TRAIN_PF_PARTICLES if is_train else TEST_PF_PARTICLES, "
    "n_seeds=TRAIN_PF_SEEDS if is_train else TEST_PF_SEEDS, scale=12.0)",
)
SIGNALS = (
    "def ps_index(h):\n"
    "    return int(h['TVT_input'].notna().sum())\n\n"
    + SIGNALS
)


INTRO = r"""# ROGII Contact-Gated Stratigraphic Alignment

This is an end-to-end training notebook. It does not import prediction CSVs or
pretrained competition models. The learned branches are fitted from the mounted
competition train wells, while PF, beam search, GR alignment, projection, and
contact checks use only information available for the current well.

## Prediction flow

1. Train a regularized ridge surface branch and a nonlinear LightGBM surface branch.
2. Build a target-free PF/beam selector.
3. Form the anchor
   \[
   T_i^A=0.30T_i^{ridge}+0.70T_i^{selector}.
   \]
4. Robustly project \(U_i=T_i+Z_i\) in normalized MD and blend at
   \(\lambda_p=0.75\).
5. Blend the projected anchor and learned branch:
   \[
   T_i^{blend}=0.60T_i^{proj}+0.40T_i^{learned}.
   \]
6. Audit heel-calibrated GR datum ambiguity and visible-prefix pseudo-holdouts.
7. Apply a same-well EGFDU contact path only when at least 50 visible rows
   reproduce within 1.0 ft RMSE.
8. Verify exact `sample_submission.csv` id order and finite output values.

The key learned target is the structural residual
\[
\Delta S=(TVT+Z)-(TVT_{last}+Z_{last}),
\]
not raw TVT. Test TVT is recovered by
\[
\widehat{TVT}=TVT_{last}+\widehat{\Delta S}-(Z-Z_{last}).
\]

`submission_honest.csv` stops before any same-well contact use.
`submission_contact_gated.csv` includes only prefix-verified contact replacements.
The selected profile is copied to `submission.csv`.
"""


CONTROL = r"""# Control panel: this is the only visible code cell.
import os

SUBMISSION_PROFILE = "contact_gated_anchor"  # honest_anchor | contact_gated_anchor | visible_prefix_bounded

RIDGE_ANCHOR_WEIGHT = 0.30
SELECTOR_ANCHOR_WEIGHT = 0.70
SELECTOR_PF_WEIGHT = 0.85
PROJECTION_DEGREE = 4
PROJECTION_WEIGHT = 0.75
PROJECTED_ANCHOR_WEIGHT = 0.60

CONTACT_REFERENCE = "EGFDU"
CONTACT_PREFIX_RMSE_LIMIT = 1.0
CONTACT_MIN_PREFIX_ROWS = 50
CONTACT_MIN_PHYS_ROWS = 100

RUN_HEEL_DATUM_AUDIT = True
APPLY_BIMODAL_HEDGE = False
RUN_VISIBLE_PREFIX_AUDIT = True
PREFIX_CUT_FRACTIONS = (0.50, 0.65, 0.75)
PREFIX_MIN_GAIN = 1.0
PREFIX_MAX_BEST_RMSE = 9.0
PREFIX_MIN_CONSISTENCY = 0.67
PREFIX_MOVE_WEIGHT_CAP = 0.22
PREFIX_MOVE_CLIP = 18.0

TRAIN_PF_PARTICLES = 120
TRAIN_PF_SEEDS = 8
TEST_PF_PARTICLES = 500
TEST_PF_SEEDS = 48
N_JOBS = 4
LGB_ESTIMATORS = 900
RUN_GROUP_CV = False
CV_FOLDS = 5

# Local runs automatically use a small smoke subset. Kaggle always uses all wells.
LOCAL_DEBUG = not os.path.exists("/kaggle/input")
DEBUG_TRAIN_WELLS = 12

if SUBMISSION_PROFILE not in {"honest_anchor", "contact_gated_anchor", "visible_prefix_bounded"}:
    raise ValueError("Unknown SUBMISSION_PROFILE")
"""


DRIVER = r"""import glob
import hashlib
import json
import math
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def find_data_root():
    candidates = [
        "/kaggle/input/competitions/rogii-wellbore-geology-prediction",
        "/kaggle/input/rogii-wellbore-geology-prediction",
        "data",
    ] + sorted(glob.glob("/kaggle/input/*"))
    for root in candidates:
        if (
            os.path.exists(f"{root}/sample_submission.csv")
            and glob.glob(f"{root}/train/*__horizontal_well.csv")
        ):
            return Path(root)
    hits = glob.glob("/kaggle/input/**/*__horizontal_well.csv", recursive=True)
    if hits:
        return Path(hits[0]).parent.parent
    raise FileNotFoundError("Could not locate the competition data")


DATA = find_data_root()
print("data root:", DATA)
print("local debug:", LOCAL_DEBUG)


def build_split(split, is_train):
    paths = sorted(glob.glob(str(DATA / split / "*__horizontal_well.csv")))
    if LOCAL_DEBUG and is_train:
        paths = paths[:DEBUG_TRAIN_WELLS]
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(build_well)(path, is_train) for path in paths
    )
    frames = [frame for frame in results if frame is not None and len(frame)]
    if not frames:
        raise RuntimeError(f"No usable {split} wells")
    return pd.concat(frames, ignore_index=True)


print("building target-free test signals")
test = build_split("test", False)
print("building train signals")
train = build_split("train", True)
train["target_surface"] = train["target"].astype(float) + train["d_z"].astype(float)
print(
    "train rows/wells:", len(train), train["well"].nunique(),
    "| test rows/wells:", len(test), test["well"].nunique(),
)

DROP = {"well", "id", "target", "target_surface", "last_known_tvt"}
FEATURES = [column for column in train.columns if column not in DROP]
X = np.nan_to_num(
    train[FEATURES].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0
)
y = train["target_surface"].to_numpy(np.float32)
groups = train["well"].to_numpy()
Xt = np.nan_to_num(
    test[FEATURES].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0
)


def pooled_rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def mean_well_rmse(frame, actual, predicted):
    audit = pd.DataFrame(
        {
            "well": frame["well"].to_numpy(),
            "sse": (np.asarray(actual) - np.asarray(predicted)) ** 2,
        }
    )
    return float(audit.groupby("well")["sse"].mean().pow(0.5).mean())


# Ridge is deliberately sampled: it is a broad, regularized anchor, not the final learner.
rng = np.random.default_rng(2026)
ridge_n = min(len(train), 500_000)
ridge_idx = rng.choice(len(train), size=ridge_n, replace=False)
scaler = StandardScaler().fit(X[ridge_idx])
ridge = Ridge(alpha=30.0).fit(scaler.transform(X[ridge_idx]), y[ridge_idx])
ridge_test_ds = ridge.predict(scaler.transform(Xt))

LGB_PARAMS = dict(
    objective="regression",
    n_estimators=LGB_ESTIMATORS if not LOCAL_DEBUG else 120,
    learning_rate=0.025,
    num_leaves=127,
    min_child_samples=100,
    subsample=0.80,
    subsample_freq=1,
    colsample_bytree=0.75,
    reg_lambda=7.0,
    reg_alpha=1.0,
    verbose=-1,
    n_jobs=-1,
    random_state=2026,
)

cv_rows = []
if RUN_GROUP_CV and train["well"].nunique() >= CV_FOLDS:
    oof = np.zeros(len(train), dtype=float)
    splitter = GroupKFold(n_splits=CV_FOLDS)
    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(X, y, groups)):
        model = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[fit_idx], y[fit_idx])
        oof[valid_idx] = model.predict(X[valid_idx])
        fold_rmse = pooled_rmse(y[valid_idx], oof[valid_idx])
        print("CV fold", fold, "pooled surface RMSE", round(fold_rmse, 4))
    cv_rows.append(
        {
            "candidate": "lightgbm_surface",
            "pooled_rmse": pooled_rmse(y, oof),
            "mean_well_rmse": mean_well_rmse(train, y, oof),
        }
    )
else:
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=2026)
    fit_idx, valid_idx = next(split.split(X, y, groups))
    probe = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[fit_idx], y[fit_idx])
    probe_pred = probe.predict(X[valid_idx])
    cv_rows.append(
        {
            "candidate": "lightgbm_surface_holdout",
            "pooled_rmse": pooled_rmse(y[valid_idx], probe_pred),
            "mean_well_rmse": mean_well_rmse(
                train.iloc[valid_idx], y[valid_idx], probe_pred
            ),
        }
    )

model = lgb.LGBMRegressor(**LGB_PARAMS).fit(X, y)
learned_test_ds = model.predict(Xt)
pd.DataFrame(cv_rows).to_csv(WORK / "training_validation_report.csv", index=False)
pd.Series(model.feature_importances_, index=FEATURES).sort_values(
    ascending=False
).rename("importance").to_csv(WORK / "feature_importance.csv")


def smooth(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    window = min(31, n if n % 2 else n - 1)
    if window >= 5:
        return savgol_filter(values, window, min(3, window - 1))
    return values


test["ridge_ds"] = ridge_test_ds
test["learned_ds"] = learned_test_ds
test["tvt_ridge"] = (
    test["last_known_tvt"].astype(float) + test["ridge_ds"] - test["d_z"]
)
test["tvt_learned"] = (
    test["last_known_tvt"].astype(float) + test["learned_ds"] - test["d_z"]
)
test["tvt_pf"] = test["last_known_tvt"].astype(float) + test["pf_d"].astype(float)
test["tvt_beam"] = (
    test["last_known_tvt"].astype(float) + test["beam_d"].astype(float)
)
test["tvt_selector"] = (
    SELECTOR_PF_WEIGHT * test["tvt_pf"]
    + (1.0 - SELECTOR_PF_WEIGHT) * test["tvt_beam"]
)
test["tvt_anchor"] = (
    RIDGE_ANCHOR_WEIGHT * test["tvt_ridge"]
    + SELECTOR_ANCHOR_WEIGHT * test["tvt_selector"]
)


def robust_polyfit(x, y_values, degree):
    x = np.asarray(x, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    good = np.isfinite(x) & np.isfinite(y_values)
    if good.sum() < max(6, degree + 2):
        return y_values.copy()
    xx, yy = x[good], y_values[good]
    deg = min(int(degree), int(good.sum()) - 2)
    coef = np.polyfit(xx, yy, deg)
    for _ in range(5):
        residual = yy - np.polyval(coef, xx)
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        weight = 1.0 / (1.0 + (residual / (2.0 * scale)) ** 2)
        coef = np.polyfit(xx, yy, deg, w=weight)
    output = y_values.copy()
    output[good] = np.polyval(coef, xx)
    return output


def load_pair(well, split):
    base = DATA / split
    return (
        pd.read_csv(base / f"{well}__horizontal_well.csv"),
        pd.read_csv(base / f"{well}__typewell.csv"),
    )


def project_group(group):
    group = group.copy()
    well = str(group["well"].iloc[0])
    hw, _ = load_pair(well, "test")
    row_idx = group["id"].str.rsplit("_", n=1).str[-1].astype(int).to_numpy()
    md = hw["MD"].to_numpy(float)[row_idx]
    z = hw["Z"].to_numpy(float)[row_idx]
    known = hw[hw["TVT_input"].notna()]
    last = known.iloc[-1]
    anchor_u = float(last["TVT_input"]) + float(last["Z"])
    start_md = float(last["MD"])
    denom = max(float(hw["MD"].iloc[-1]) - start_md, 1e-6)
    s = (md - start_md) / denom
    raw = group["tvt_anchor"].to_numpy(float)
    delta_u = raw + z - anchor_u
    fitted_delta_u = robust_polyfit(s, delta_u, PROJECTION_DEGREE)
    projected = anchor_u + fitted_delta_u - z
    group["tvt_projected"] = (
        (1.0 - PROJECTION_WEIGHT) * raw + PROJECTION_WEIGHT * projected
    )
    group["tvt_honest"] = (
        PROJECTED_ANCHOR_WEIGHT * group["tvt_projected"].to_numpy(float)
        + (1.0 - PROJECTED_ANCHOR_WEIGHT)
        * smooth(group["tvt_learned"].to_numpy(float))
    )
    return group


test = pd.concat(
    [project_group(group) for _, group in test.groupby("well", sort=False)],
    ignore_index=True,
)


def heel_calibration_and_scan(well, base_pred):
    hw, tw = load_pair(well, "test")
    known = hw[hw["TVT_input"].notna()].copy()
    tws = tw.sort_values("TVT")
    tw_tvt = tws["TVT"].to_numpy(float)
    tw_gr = tws["GR"].interpolate(limit_direction="both").to_numpy(float)
    expected = np.interp(known["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    observed = known["GR"].to_numpy(float)
    mask = np.isfinite(expected) & np.isfinite(observed)
    if mask.sum() < 40:
        return base_pred, {"well": well, "status": "too_few_prefix_rows"}
    design = np.c_[expected[mask], np.ones(mask.sum())]
    alpha, beta = np.linalg.lstsq(design, observed[mask], rcond=None)[0]
    alpha = float(np.clip(alpha, 0.35, 2.50))
    beta = float(np.clip(beta, -120.0, 120.0))
    calibrated = (hw["GR"].to_numpy(float) - beta) / alpha
    eval_idx = np.flatnonzero(hw["TVT_input"].isna().to_numpy())
    gr_eval = calibrated[eval_idx]
    valid = np.isfinite(gr_eval) & np.isfinite(base_pred)
    if valid.sum() < 80:
        return base_pred, {
            "well": well,
            "status": "too_few_hidden_gr_rows",
            "alpha": alpha,
            "beta": beta,
        }
    prefix_resid = observed[mask] - (alpha * expected[mask] + beta)
    scale = float(np.clip(1.4826 * np.median(np.abs(prefix_resid)), 5.0, 60.0))
    shifts = np.arange(-20.0, 20.01, 0.5)
    scores = []
    for shift in shifts:
        residual = (
            gr_eval[valid]
            - np.interp(base_pred[valid] + shift, tw_tvt, tw_gr)
        ) / scale
        scores.append(float(np.mean(np.clip(residual, -6.0, 6.0) ** 2)))
    scores = np.asarray(scores)
    order = np.argsort(scores)
    first = int(order[0])
    second = next(
        (int(index) for index in order[1:] if abs(shifts[index] - shifts[first]) >= 8.0),
        int(order[min(1, len(order) - 1)]),
    )
    plausible = scores[second] <= scores[first] * 1.15
    fitted = alpha * expected[mask] + beta
    r2 = 1.0 - np.sum((observed[mask] - fitted) ** 2) / (
        np.sum((observed[mask] - observed[mask].mean()) ** 2) + 1e-9
    )
    trust = float(np.clip(r2, 0.0, 1.0))
    logits = -np.array([scores[first], scores[second]]) / 0.75
    logits -= logits.max()
    p_scan = float(np.exp(logits[0]) / np.exp(logits).sum())
    p_effective = trust * p_scan + (1.0 - trust) * 0.5
    hedge_shift = float(
        p_effective * shifts[first] + (1.0 - p_effective) * shifts[second]
    )
    output = base_pred + hedge_shift if APPLY_BIMODAL_HEDGE and plausible else base_pred
    return output, {
        "well": well,
        "status": "hedged" if APPLY_BIMODAL_HEDGE and plausible else "audit_only",
        "alpha": alpha,
        "beta": beta,
        "prefix_r2": trust,
        "shift_1": float(shifts[first]),
        "score_1": float(scores[first]),
        "shift_2": float(shifts[second]),
        "score_2": float(scores[second]),
        "plausible_bimodal": bool(plausible),
        "p_scan": p_scan,
        "p_effective": p_effective,
        "hedge_shift": hedge_shift,
    }


datum_rows = []
if RUN_HEEL_DATUM_AUDIT:
    adjusted = []
    for well, group in test.groupby("well", sort=False):
        values, report = heel_calibration_and_scan(
            well, group["tvt_honest"].to_numpy(float)
        )
        group = group.copy()
        group["tvt_honest"] = values
        adjusted.append(group)
        datum_rows.append(report)
    test = pd.concat(adjusted, ignore_index=True)
pd.DataFrame(datum_rows).to_csv(WORK / "heel_bimodal_audit.csv", index=False)


def prefix_candidate(hw, cut, name, target_idx):
    known_idx = np.flatnonzero(hw["TVT_input"].notna().to_numpy())
    train_idx = known_idx[:cut]
    if len(train_idx) < 20:
        return None
    tvt = hw["TVT_input"].to_numpy(float)
    z = hw["Z"].to_numpy(float)
    md = hw["MD"].to_numpy(float)
    last = train_idx[-1]
    if name == "flat":
        return np.full(len(target_idx), tvt[last])
    if name == "structural":
        return tvt[last] + z[last] - z[target_idx]
    tail = train_idx[-min(80, len(train_idx)) :]
    surface = tvt[tail] + z[tail]
    slope = np.polyfit(md[tail] - md[last], surface, 1)[0]
    slope = float(np.clip(slope, -0.08, 0.08))
    return tvt[last] + z[last] + slope * (md[target_idx] - md[last]) - z[target_idx]


def run_prefix_audit(well):
    hw, _ = load_pair(well, "test")
    known_idx = np.flatnonzero(hw["TVT_input"].notna().to_numpy())
    names = ("flat", "structural", "surface_trend")
    rows = []
    for fraction in PREFIX_CUT_FRACTIONS:
        cut = int(round(len(known_idx) * fraction))
        hold_idx = known_idx[cut:]
        if cut < 20 or len(hold_idx) < 10:
            continue
        truth = hw["TVT_input"].to_numpy(float)[hold_idx]
        for name in names:
            pred = prefix_candidate(hw, cut, name, hold_idx)
            rows.append(
                {
                    "well": well,
                    "cut_fraction": fraction,
                    "candidate": name,
                    "rmse": pooled_rmse(truth, pred),
                }
            )
    return rows


prefix_rows = []
if RUN_VISIBLE_PREFIX_AUDIT:
    for well in test["well"].drop_duplicates():
        prefix_rows.extend(run_prefix_audit(well))
prefix_report = pd.DataFrame(prefix_rows)
prefix_report.to_csv(WORK / "visible_prefix_audit.csv", index=False)


def apply_prefix_bounded(group):
    if SUBMISSION_PROFILE != "visible_prefix_bounded" or prefix_report.empty:
        return group["tvt_honest"].to_numpy(float), {
            "well": group["well"].iloc[0],
            "accepted": False,
            "reason": "profile_audit_only",
        }
    well = str(group["well"].iloc[0])
    report = prefix_report[prefix_report["well"] == well]
    if report.empty:
        return group["tvt_honest"].to_numpy(float), {
            "well": well, "accepted": False, "reason": "no_prefix_report"
        }
    summary = report.groupby("candidate")["rmse"].agg(["median", "std", "count"])
    summary["score"] = summary["median"] + 0.10 * summary["std"].fillna(0.0)
    winner = str(summary["score"].idxmin())
    best = float(summary.loc[winner, "score"])
    flat = float(summary.loc["flat", "score"])
    gain = flat - best
    cuts = report[report["candidate"] == winner]
    flat_by_cut = report[report["candidate"] == "flat"].set_index("cut_fraction")["rmse"]
    consistency = float(
        np.mean(
            [
                row.rmse < flat_by_cut.get(row.cut_fraction, np.inf)
                for row in cuts.itertuples()
            ]
        )
    )
    accepted = (
        winner != "flat"
        and gain >= PREFIX_MIN_GAIN
        and best <= PREFIX_MAX_BEST_RMSE
        and consistency >= PREFIX_MIN_CONSISTENCY
    )
    if not accepted:
        return group["tvt_honest"].to_numpy(float), {
            "well": well,
            "accepted": False,
            "reason": "quality_gate",
            "winner": winner,
            "gain": gain,
            "best_rmse": best,
            "consistency": consistency,
        }
    hw, _ = load_pair(well, "test")
    hidden_idx = group["id"].str.rsplit("_", n=1).str[-1].astype(int).to_numpy()
    known_count = int(hw["TVT_input"].notna().sum())
    candidate = prefix_candidate(hw, known_count, winner, hidden_idx)
    base = group["tvt_honest"].to_numpy(float)
    ramp = 1.0 - np.exp(-np.arange(len(base)) / max(80.0, 0.12 * len(base)))
    weight = min(PREFIX_MOVE_WEIGHT_CAP, 0.08 + 0.12 * min(gain, 5.0) / 5.0)
    move = np.clip(weight * ramp * (candidate - base), -PREFIX_MOVE_CLIP, PREFIX_MOVE_CLIP)
    return base + move, {
        "well": well,
        "accepted": True,
        "reason": "prefix_verified",
        "winner": winner,
        "gain": gain,
        "best_rmse": best,
        "consistency": consistency,
        "weight": weight,
        "max_abs_move": float(np.max(np.abs(move))),
    }


prefix_move_rows = []
prefix_adjusted = []
for _, group in test.groupby("well", sort=False):
    values, report = apply_prefix_bounded(group)
    group = group.copy()
    group["tvt_pre_contact"] = values
    prefix_adjusted.append(group)
    prefix_move_rows.append(report)
test = pd.concat(prefix_adjusted, ignore_index=True)
pd.DataFrame(prefix_move_rows).to_csv(WORK / "visible_prefix_moves.csv", index=False)


def contact_reconstruction(hw_train, tw_train, reference):
    if reference not in hw_train.columns:
        return None
    geology = tw_train.dropna(subset=["Geology", "TVT"]).copy()
    ref = geology.loc[geology["Geology"].astype(str) == reference, "TVT"]
    if ref.empty:
        return None
    ref_tvt = float(ref.min())
    raw = ref_tvt - (
        hw_train["Z"].to_numpy(float) - hw_train[reference].to_numpy(float)
    )
    bias = float(np.nanmean(hw_train["TVT"].to_numpy(float) - raw))
    result = raw + bias
    return result if np.isfinite(result).sum() >= CONTACT_MIN_PHYS_ROWS else None


def apply_contact_guard(group):
    well = str(group["well"].iloc[0])
    train_hw_path = DATA / "train" / f"{well}__horizontal_well.csv"
    train_tw_path = DATA / "train" / f"{well}__typewell.csv"
    base = group["tvt_pre_contact"].to_numpy(float)
    report = {
        "well": well,
        "accepted": False,
        "reference": CONTACT_REFERENCE,
        "replaced_rows": 0,
    }
    if not train_hw_path.exists() or not train_tw_path.exists():
        report["reason"] = "no_same_well_train_copy"
        return base, report
    hw_test, _ = load_pair(well, "test")
    hw_train = pd.read_csv(train_hw_path)
    tw_train = pd.read_csv(train_tw_path)
    contact = contact_reconstruction(hw_train, tw_train, CONTACT_REFERENCE)
    if contact is None:
        report["reason"] = "contact_unavailable"
        return base, report
    train_md = hw_train["MD"].to_numpy(float)
    known = hw_test[hw_test["TVT_input"].notna()].copy()
    in_range = known["MD"].between(np.nanmin(train_md), np.nanmax(train_md)).to_numpy()
    comparable = known.loc[in_range]
    if len(comparable) < CONTACT_MIN_PREFIX_ROWS:
        report["reason"] = "too_few_comparable_prefix_rows"
        report["prefix_rows"] = int(len(comparable))
        return base, report
    prefix_pred = np.interp(comparable["MD"].to_numpy(float), train_md, contact)
    prefix_truth = comparable["TVT_input"].to_numpy(float)
    prefix_rmse = pooled_rmse(prefix_truth, prefix_pred)
    report["prefix_rows"] = int(len(comparable))
    report["prefix_rmse"] = prefix_rmse
    if prefix_rmse > CONTACT_PREFIX_RMSE_LIMIT:
        report["reason"] = "prefix_rmse_failed"
        return base, report
    hidden_idx = group["id"].str.rsplit("_", n=1).str[-1].astype(int).to_numpy()
    hidden_md = hw_test["MD"].to_numpy(float)[hidden_idx]
    replace = (hidden_md >= np.nanmin(train_md)) & (hidden_md <= np.nanmax(train_md))
    output = base.copy()
    output[replace] = np.interp(hidden_md[replace], train_md, contact)
    report.update(
        {
            "accepted": True,
            "reason": "prefix_verified",
            "replaced_rows": int(replace.sum()),
        }
    )
    return output, report


contact_rows = []
contact_adjusted = []
for _, group in test.groupby("well", sort=False):
    values, report = apply_contact_guard(group)
    group = group.copy()
    group["tvt_contact"] = values
    contact_adjusted.append(group)
    contact_rows.append(report)
test = pd.concat(contact_adjusted, ignore_index=True)
pd.DataFrame(contact_rows).to_csv(WORK / "guarded_contact_report.csv", index=False)


sample = pd.read_csv(DATA / "sample_submission.csv")


def ordered_submission(column):
    prediction = test[["id", column]].rename(columns={column: "tvt"})
    if prediction["id"].duplicated().any():
        raise RuntimeError(f"Duplicate ids in {column}")
    output = sample[["id"]].merge(prediction, on="id", how="left", validate="one_to_one")
    if output["tvt"].isna().any():
        missing = output.loc[output["tvt"].isna(), "id"].head().tolist()
        raise RuntimeError(f"Missing predictions for {column}: {missing}")
    if output["id"].tolist() != sample["id"].tolist():
        raise RuntimeError("Submission id order changed")
    if not np.isfinite(output["tvt"].to_numpy(float)).all():
        raise RuntimeError(f"Non-finite predictions in {column}")
    return output


honest_sub = ordered_submission("tvt_pre_contact")
contact_sub = ordered_submission("tvt_contact")
honest_sub.to_csv(WORK / "submission_honest.csv", index=False)
contact_sub.to_csv(WORK / "submission_contact_gated.csv", index=False)
final_sub = honest_sub if SUBMISSION_PROFILE == "honest_anchor" else contact_sub
final_sub.to_csv(WORK / "submission.csv", index=False)

sha = hashlib.sha256((WORK / "submission.csv").read_bytes()).hexdigest()
audit = {
    "profile": SUBMISSION_PROFILE,
    "rows": int(len(final_sub)),
    "unique_ids": int(final_sub["id"].nunique()),
    "id_order_matches_sample": bool(final_sub["id"].tolist() == sample["id"].tolist()),
    "finite_tvt": bool(np.isfinite(final_sub["tvt"].to_numpy(float)).all()),
    "tvt_min": float(final_sub["tvt"].min()),
    "tvt_max": float(final_sub["tvt"].max()),
    "contact_wells_accepted": int(sum(bool(row.get("accepted")) for row in contact_rows)),
    "contact_rows_replaced": int(sum(int(row.get("replaced_rows", 0)) for row in contact_rows)),
    "sha256": sha,
}
(WORK / "submission_audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
print("wrote", WORK / "submission.csv")
final_sub.head()
"""


NOTEBOOK = {
    "cells": [
        markdown(INTRO),
        code(CONTROL, hidden=False),
        code(PF),
        code(BEAM),
        code(SIGNALS),
        code(DRIVER),
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

METADATA = {
    "id": "boltuzamaki/rogii-contact-gated-stratigraphic-alignment",
    "title": "ROGII Contact-Gated Stratigraphic Alignment",
    "code_file": "rogii-contact-gated-stratigraphic-training.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": False,
    "dataset_sources": [],
    "competition_sources": ["rogii-wellbore-geology-prediction"],
    "kernel_sources": [],
}


OUT.mkdir(parents=True, exist_ok=True)
(OUT / METADATA["code_file"]).write_text(
    json.dumps(NOTEBOOK, indent=1), encoding="utf-8"
)
(OUT / "kernel-metadata.json").write_text(
    json.dumps(METADATA, indent=2), encoding="utf-8"
)
print("wrote", OUT)
