from __future__ import annotations

import json
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "kernels" / "external_combined_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    md(
        """# ROGII Combined Geology Pretraining — Nested Group CV

This notebook tests whether the public **Geology Forecast Challenge** horizon
curves add signal beyond current-competition training wells. It is deliberately
CV-only: it does **not** create a competition submission.

## Scientific controls

- The older challenge is read from its original Kaggle competition mount.
- Only `train_raw/*.csv` is used. The processed realization columns are
  duplicated and are excluded.
- Outer validation splits by current-competition well.
- External windows are grouped by original horizon curve.
- External mixing weight and Ridge regularization are selected inside each
  outer fold using only inner training wells.
- Models forecast the residual around a local linear surface continuation,
  which reduces recursive drift.
- Artifacts are promoted only when predeclared CV gates pass.

## Outputs

`cv_summary.csv`, `cv_fold_metrics.csv`, `cv_well_metrics.csv`,
`hyperparameter_search.csv`, `quality_gate.json`, diagnostic plots, and—only
after a pass—`combined_surface_ridge.joblib` plus `model_card.json`.
"""
    ),
    code(
        """# Configuration and deterministic runtime
import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

SEED = 20260721
random.seed(SEED)
np.random.seed(SEED)

SMOKE_MODE = os.environ.get("ROGII_SMOKE", "0") == "1"
N_SPLITS = 2 if SMOKE_MODE else 5
WINDOW_STRIDE = 600 if SMOKE_MODE else 200
MAX_CURRENT_WELLS = 60 if SMOKE_MODE else None
PAST_FT = 300
FUTURE_FT = 300
N_GRID = 60
MAX_LOCAL_SLOPE = 0.60
ALPHA_GRID = [1.0, 10.0] if SMOKE_MODE else [0.1, 1.0, 10.0, 100.0]
EXTERNAL_WEIGHT_GRID = [0.25, 1.0] if SMOKE_MODE else [0.10, 0.25, 0.50, 1.00]
HORIZONS = [300, 600, 1200, 2400, 4800]
DECAY_TAU_GRID = [600.0, 1200.0, 2400.0]

# Conservative gates for deciding whether this branch deserves integration.
GATE_RELATIVE_300 = 0.995
GATE_RELATIVE_600 = 0.995
GATE_LONG_RANGE_TO_LINEAR = 1.005
GATE_MIN_WELL_WIN_RATE = 0.52

default_work = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
WORK = Path(os.environ.get("ROGII_OUTPUT_DIR", str(default_work)))
WORK.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rogii-combined-cv")
log.info("SMOKE_MODE=%s folds=%d stride=%d", SMOKE_MODE, N_SPLITS, WINDOW_STRIDE)
"""
    ),
    code(
        """# Discover both competition mounts without hard-coding one Kaggle layout
def find_current_data():
    local_candidates = [base / "data" for base in [Path.cwd(), *Path.cwd().parents]]
    candidates = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    ] + local_candidates
    for candidate in candidates:
        if (candidate / "train").exists():
            return candidate
    matches = list(Path("/kaggle/input").glob("**/train/*__horizontal_well.csv")) if Path("/kaggle/input").exists() else []
    if matches:
        return matches[0].parent.parent
    raise FileNotFoundError("Current ROGII competition data was not found")


def find_external_data():
    local_candidates = [
        base / "external_data/geology_forecast_challenge/extracted/data"
        for base in [Path.cwd(), *Path.cwd().parents]
    ]
    candidates = [
        Path("/kaggle/input/competitions/geology-forecast-challenge-open/data"),
        Path("/kaggle/input/geology-forecast-challenge-open/data"),
    ] + local_candidates
    for candidate in candidates:
        if (candidate / "train_raw").exists():
            return candidate
    matches = list(Path("/kaggle/input").glob("**/train_raw/*.csv")) if Path("/kaggle/input").exists() else []
    if matches:
        return matches[0].parent.parent
    raise FileNotFoundError("Geology Forecast Challenge train_raw data was not found")


CURRENT_DATA = find_current_data()
EXTERNAL_DATA = find_external_data()
log.info("Current data: %s", CURRENT_DATA)
log.info("External data: %s", EXTERNAL_DATA)

current_files = sorted((CURRENT_DATA / "train").glob("*__horizontal_well.csv"))
external_files = sorted((EXTERNAL_DATA / "train_raw").glob("*.csv"))
if MAX_CURRENT_WELLS:
    current_files = current_files[:MAX_CURRENT_WELLS]
log.info("Current wells=%d external curves=%d", len(current_files), len(external_files))
assert len(current_files) >= N_SPLITS * 5
assert len(external_files) >= 50
"""
    ),
    code(
        """# Window construction and leakage-safe feature representation
PAST_GRID = np.linspace(-(PAST_FT - 1), 0.0, N_GRID)
FUTURE_GRID = np.linspace(1.0, FUTURE_FT, N_GRID)


def clean_curve(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique = np.r_[True, np.diff(x) > 1e-8]
    return x[unique], y[unique]


def feature_target(past, future=None):
    past = np.asarray(past, float)
    coef = np.polyfit(PAST_GRID, past, 1)
    linear_past = np.polyval(coef, PAST_GRID)
    detrended = past - linear_past
    d1 = np.gradient(detrended)
    d2 = np.gradient(d1)
    extras = np.array([
        coef[0],
        np.std(detrended),
        np.mean(np.abs(d1)),
        np.mean(np.abs(d2)),
    ])
    x = np.concatenate([detrended, d1, d2, extras])
    if future is None:
        return x, np.polyval(coef, FUTURE_GRID)
    linear_future = np.polyval(coef, FUTURE_GRID)
    return x, np.asarray(future, float) - linear_future


def valid_window(past, future):
    seq = np.r_[past, future]
    if not np.all(np.isfinite(seq)):
        return False
    return np.quantile(np.abs(np.diff(seq)), 0.99) <= MAX_LOCAL_SLOPE


def curve_windows(name, x, y, stride=WINDOW_STRIDE):
    x, y = clean_curve(x, y)
    lo = int(np.ceil(x[0] + PAST_FT - 1))
    hi = int(np.floor(x[-1] - FUTURE_FT))
    xs, ys = [], []
    if hi < lo:
        return xs, ys
    for anchor in np.arange(lo, hi + 1, stride):
        past = np.interp(anchor + PAST_GRID, x, y)
        future = np.interp(anchor + FUTURE_GRID, x, y)
        if valid_window(past, future):
            xx, yy = feature_target(past, future)
            xs.append(xx)
            ys.append(yy)
    return xs, ys


def load_external_windows():
    xs, ys, groups = [], [], []
    for path in external_files:
        df = pd.read_csv(path, usecols=["VS_APPROX_adjusted", "HORIZON_Z_adjusted"]).dropna()
        xx, yy = curve_windows(
            path.stem,
            df["VS_APPROX_adjusted"].to_numpy(float),
            df["HORIZON_Z_adjusted"].to_numpy(float),
        )
        xs.extend(xx); ys.extend(yy); groups.extend([path.stem] * len(xx))
    return np.asarray(xs), np.asarray(ys), np.asarray(groups)


def load_current_wells_and_windows():
    wells = {}
    xs, ys, groups = [], [], []
    for number, path in enumerate(current_files, 1):
        well = path.name.split("__")[0]
        df = pd.read_csv(path, usecols=["MD", "Z", "TVT", "TVT_input"])
        md = df["MD"].to_numpy(float)
        z = df["Z"].to_numpy(float)
        tvt = df["TVT"].to_numpy(float)
        tvt_input = df["TVT_input"].to_numpy(float)
        u = tvt + z
        md_clean, u_clean = clean_curve(md, u)
        xx, yy = curve_windows(well, md_clean, u_clean)
        xs.extend(xx); ys.extend(yy); groups.extend([well] * len(xx))
        wells[well] = {"md": md, "z": z, "tvt": tvt, "tvt_input": tvt_input}
        if number % 100 == 0:
            log.info("Loaded %d/%d current wells", number, len(current_files))
    return wells, np.asarray(xs), np.asarray(ys), np.asarray(groups)


t0 = time.time()
X_EXT, Y_EXT, G_EXT = load_external_windows()
WELLS, X_CUR, Y_CUR, G_CUR = load_current_wells_and_windows()
log.info(
    "Windows external=%s from %d curves | current=%s from %d wells | %.1fs",
    X_EXT.shape, len(np.unique(G_EXT)), X_CUR.shape, len(np.unique(G_CUR)), time.time() - t0,
)
assert len(X_EXT) > 100 and len(X_CUR) > 100
"""
    ),
    code(
        """# Stable multi-output Ridge model and recursive forecasting
class ScaledRidge:
    def __init__(self, alpha):
        self.alpha = float(alpha)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha)

    def fit(self, x, y, sample_weight=None):
        xs = self.x_scaler.fit_transform(x)
        ys = self.y_scaler.fit_transform(y)
        self.model.fit(xs, ys, sample_weight=sample_weight)
        return self

    def predict(self, x):
        ys = self.model.predict(self.x_scaler.transform(x))
        return self.y_scaler.inverse_transform(ys)


def predict_next(model, past):
    features, linear_future = feature_target(past)
    return linear_future + model.predict(features[None, :])[0]


def recursive_forecast(model, past, deltas):
    deltas = np.asarray(deltas, float)
    max_delta = float(np.max(deltas))
    state = np.asarray(past, float)
    all_deltas, all_values = [], []
    start = 0.0
    while start < max_delta:
        future = predict_next(model, state)
        all_deltas.extend(start + FUTURE_GRID)
        all_values.extend(future)
        state = future
        start += FUTURE_FT
    return np.interp(deltas, np.asarray(all_deltas), np.asarray(all_values))


def official_cut_context(well_data):
    md = well_data["md"]
    visible = np.isfinite(well_data["tvt_input"])
    if not visible.any() or visible.all():
        return None
    last = np.flatnonzero(visible)[-1]
    anchor = md[last]
    if anchor - md[0] < PAST_FT - 1:
        return None
    hidden = np.arange(last + 1, len(md))
    if not len(hidden):
        return None
    u_input = well_data["tvt_input"] + well_data["z"]
    u_true = well_data["tvt"] + well_data["z"]
    past = np.interp(anchor + PAST_GRID, md[: last + 1], u_input[: last + 1])
    delta = md[hidden] - anchor
    truth = u_true[hidden]
    linear_coef = np.polyfit(PAST_GRID, past, 1)
    linear = np.polyval(linear_coef, delta)
    return past, delta, truth, linear


def evaluate_model(model, well_ids, fold, candidate, decay_tau=None):
    rows = []
    for well in well_ids:
        context = official_cut_context(WELLS[well])
        if context is None:
            continue
        past, delta, truth, linear = context
        raw = recursive_forecast(model, past, delta)
        if decay_tau is None:
            pred = raw
        else:
            weight = np.exp(-delta / float(decay_tau))
            pred = linear + weight * (raw - linear)
        for horizon in HORIZONS:
            mask = delta <= horizon
            if not mask.any():
                continue
            err2 = (pred[mask] - truth[mask]) ** 2
            linear_err2 = (linear[mask] - truth[mask]) ** 2
            rows.append({
                "fold": fold,
                "well": well,
                "candidate": candidate,
                "horizon": horizon,
                "n_rows": int(mask.sum()),
                "sse": float(err2.sum()),
                "rmse": float(np.sqrt(err2.mean())),
                "linear_sse": float(linear_err2.sum()),
                "linear_rmse": float(np.sqrt(linear_err2.mean())),
                "beats_linear": bool(err2.mean() < linear_err2.mean()),
            })
    return rows
"""
    ),
    code(
        """# Nested group CV: tune only on inner current-well windows
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def tune_combined(train_wells, fold):
    mask = np.isin(G_CUR, train_wells)
    x, y, groups = X_CUR[mask], Y_CUR[mask], G_CUR[mask]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=SEED + fold)
    inner_train, inner_valid = next(splitter.split(x, y, groups))
    records = []
    best = (np.inf, None, None)
    for alpha in ALPHA_GRID:
        for external_weight in EXTERNAL_WEIGHT_GRID:
            x_fit = np.vstack([x[inner_train], X_EXT])
            y_fit = np.vstack([y[inner_train], Y_EXT])
            weights = np.r_[
                np.ones(len(inner_train)),
                np.full(len(X_EXT), external_weight),
            ]
            model = ScaledRidge(alpha).fit(x_fit, y_fit, weights)
            score = rmse(y[inner_valid], model.predict(x[inner_valid]))
            records.append({
                "fold": fold,
                "alpha": alpha,
                "external_weight": external_weight,
                "inner_window_rmse": score,
                "inner_train_windows": len(inner_train),
                "inner_valid_windows": len(inner_valid),
            })
            if score < best[0]:
                best = (score, alpha, external_weight)
    log.info(
        "Fold %d best inner RMSE=%.4f alpha=%s external_weight=%s",
        fold, best[0], best[1], best[2],
    )
    return best[1], best[2], records


all_wells = np.asarray(sorted(WELLS))
outer = GroupKFold(n_splits=N_SPLITS)
fold_rows = []
search_rows = []
selected_params = []

# External-only benchmark is safe to fit once because it uses no current labels.
external_model = ScaledRidge(alpha=1.0).fit(X_EXT, Y_EXT)

for fold, (train_index, valid_index) in enumerate(outer.split(all_wells, groups=all_wells), 1):
    fold_start = time.time()
    train_wells = all_wells[train_index]
    valid_wells = all_wells[valid_index]
    alpha, external_weight, records = tune_combined(train_wells, fold)
    search_rows.extend(records)
    selected_params.append({"fold": fold, "alpha": alpha, "external_weight": external_weight})

    current_mask = np.isin(G_CUR, train_wells)
    current_model = ScaledRidge(alpha).fit(X_CUR[current_mask], Y_CUR[current_mask])
    combined_x = np.vstack([X_CUR[current_mask], X_EXT])
    combined_y = np.vstack([Y_CUR[current_mask], Y_EXT])
    combined_weights = np.r_[
        np.ones(current_mask.sum()),
        np.full(len(X_EXT), external_weight),
    ]
    combined_model = ScaledRidge(alpha).fit(combined_x, combined_y, combined_weights)

    fold_rows.extend(evaluate_model(external_model, valid_wells, fold, "external_only"))
    fold_rows.extend(evaluate_model(current_model, valid_wells, fold, "current_only"))
    fold_rows.extend(evaluate_model(combined_model, valid_wells, fold, "combined_raw"))
    for tau in DECAY_TAU_GRID:
        fold_rows.extend(
            evaluate_model(combined_model, valid_wells, fold, f"combined_decay_{int(tau)}", decay_tau=tau)
        )
    log.info(
        "Fold %d complete: train_wells=%d valid_wells=%d windows=%d elapsed=%.1fs",
        fold, len(train_wells), len(valid_wells), current_mask.sum(), time.time() - fold_start,
    )
    del current_model, combined_model, combined_x, combined_y
    gc.collect()

well_metrics = pd.DataFrame(fold_rows)
search = pd.DataFrame(search_rows)
selected = pd.DataFrame(selected_params)
well_metrics.to_csv(WORK / "cv_well_metrics.csv", index=False)
search.to_csv(WORK / "hyperparameter_search.csv", index=False)
selected.to_csv(WORK / "selected_hyperparameters.csv", index=False)
log.info("OOF well-metric rows=%d", len(well_metrics))
"""
    ),
    code(
        """# Aggregate OOF metrics and apply predeclared quality gates
def aggregate_metrics(frame):
    rows = []
    for (candidate, horizon), group in frame.groupby(["candidate", "horizon"]):
        rows.append({
            "candidate": candidate,
            "horizon": int(horizon),
            "wells": int(group["well"].nunique()),
            "rows": int(group["n_rows"].sum()),
            "row_weighted_rmse": float(np.sqrt(group["sse"].sum() / group["n_rows"].sum())),
            "mean_well_rmse": float(group["rmse"].mean()),
            "median_well_rmse": float(group["rmse"].median()),
            "well_win_rate_vs_linear": float(group["beats_linear"].mean()),
            "linear_row_weighted_rmse": float(np.sqrt(group["linear_sse"].sum() / group["n_rows"].sum())),
        })
    return pd.DataFrame(rows).sort_values(["horizon", "row_weighted_rmse"])


summary = aggregate_metrics(well_metrics)
summary.to_csv(WORK / "cv_summary.csv", index=False)
fold_metrics = aggregate_metrics(
    well_metrics.assign(candidate=well_metrics["candidate"] + "__fold" + well_metrics["fold"].astype(str))
)
fold_metrics.to_csv(WORK / "cv_fold_metrics.csv", index=False)
display(summary)


def metric(candidate, horizon, column="row_weighted_rmse"):
    row = summary[(summary.candidate == candidate) & (summary.horizon == horizon)]
    return float(row.iloc[0][column]) if len(row) else np.inf


# Choose the decay candidate by 1,200-ft OOF RMSE; this is a model-selection
# diagnostic only. A later stack-integration notebook must repeat the comparison
# against deployed stack OOF before any submission is created.
decay_candidates = [f"combined_decay_{int(t)}" for t in DECAY_TAU_GRID]
best_decay = min(decay_candidates, key=lambda name: metric(name, 1200))

combined_300 = metric("combined_raw", 300)
best_reference_300 = min(metric("current_only", 300), metric("external_only", 300))
combined_600 = metric("combined_raw", 600)
best_reference_600 = min(metric("current_only", 600), metric("external_only", 600))
long_4800 = metric(best_decay, 4800)
linear_4800 = metric(best_decay, 4800, "linear_row_weighted_rmse")
win_rate_600 = metric("combined_raw", 600, "well_win_rate_vs_linear")

checks = {
    "combined_beats_best_single_source_300": combined_300 <= GATE_RELATIVE_300 * best_reference_300,
    "combined_beats_best_single_source_600": combined_600 <= GATE_RELATIVE_600 * best_reference_600,
    "decayed_branch_long_range_safe": long_4800 <= GATE_LONG_RANGE_TO_LINEAR * linear_4800,
    "combined_well_win_rate_600": win_rate_600 >= GATE_MIN_WELL_WIN_RATE,
}
gate = {
    "passed": bool(all(checks.values())),
    "checks": checks,
    "best_decay_candidate": best_decay,
    "metrics": {
        "combined_300": combined_300,
        "best_single_source_300": best_reference_300,
        "combined_600": combined_600,
        "best_single_source_600": best_reference_600,
        "best_decay_4800": long_4800,
        "linear_4800": linear_4800,
        "combined_win_rate_600": win_rate_600,
    },
    "next_step": (
        "Run stack-OOF integration CV; do not submit yet."
        if all(checks.values())
        else "Do not integrate or submit; inspect failed checks."
    ),
}
(WORK / "quality_gate.json").write_text(json.dumps(gate, indent=2))
log.info("QUALITY GATE: %s", json.dumps(gate, indent=2))

# Plot concise horizon comparison.
fig, ax = plt.subplots(figsize=(10, 6))
plot_names = ["external_only", "current_only", "combined_raw", best_decay]
for name in plot_names:
    part = summary[summary.candidate == name].sort_values("horizon")
    ax.plot(part.horizon, part.row_weighted_rmse, marker="o", label=name)
linear = summary[summary.candidate == best_decay].sort_values("horizon")
ax.plot(linear.horizon, linear.linear_row_weighted_rmse, marker="o", linestyle="--", label="linear")
ax.set(xlabel="Forecast distance (ft)", ylabel="Row-weighted RMSE (ft)", title="Nested group CV by forecast horizon")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(WORK / "cv_horizon_rmse.png", dpi=160)
plt.show()
"""
    ),
    code(
        """# Retrain and export only after the CV gate passes
if gate["passed"]:
    final_alpha = float(selected["alpha"].mode().iloc[0])
    final_external_weight = float(selected["external_weight"].mode().iloc[0])
    final_x = np.vstack([X_CUR, X_EXT])
    final_y = np.vstack([Y_CUR, Y_EXT])
    final_weights = np.r_[np.ones(len(X_CUR)), np.full(len(X_EXT), final_external_weight)]
    final_model = ScaledRidge(final_alpha).fit(final_x, final_y, final_weights)
    joblib.dump(final_model, WORK / "combined_surface_ridge.joblib")
    model_card = {
        "model": "detrended multi-output Ridge surface forecaster",
        "alpha": final_alpha,
        "external_weight": final_external_weight,
        "past_ft": PAST_FT,
        "future_ft": FUTURE_FT,
        "grid_points": N_GRID,
        "current_wells": len(WELLS),
        "current_windows": len(X_CUR),
        "external_curves": int(len(np.unique(G_EXT))),
        "external_windows": len(X_EXT),
        "cv_gate": gate,
        "allowed_next_step": "stack OOF integration CV only",
        "submission_authorized": False,
    }
    (WORK / "model_card.json").write_text(json.dumps(model_card, indent=2))
    log.info("Model package exported. Submission remains disabled.")
else:
    log.warning("CV gate failed. No model package exported and no submission created.")

assert not (WORK / "submission.csv").exists(), "This training notebook must never create submission.csv"
print("Training notebook complete. Gate passed:", gate["passed"])
print("Next step:", gate["next_step"])
"""
    ),
    md(
        """## Interpretation rule

A passing result is permission to build a second **stack-integration CV**
notebook, not permission to submit to the leaderboard. The combined branch must
still demonstrate incremental value against the deployed stack's OOF errors,
including error correlation, worst-decile wells, and per-horizon safety.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

notebook_path = OUT_DIR / "rogii-combined-geology-cv-r1.ipynb"
notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

metadata = {
    "id": "boltuzmaki/rogii-combined-geology-cv-r1",
    "title": "ROGII Combined Geology Nested Group CV R1",
    "code_file": notebook_path.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": False,
    "dataset_sources": [],
    "competition_sources": [
        "rogii-wellbore-geology-prediction",
        "geology-forecast-challenge-open",
    ],
    "kernel_sources": [],
}
(OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print(notebook_path)
print(OUT_DIR / "kernel-metadata.json")
