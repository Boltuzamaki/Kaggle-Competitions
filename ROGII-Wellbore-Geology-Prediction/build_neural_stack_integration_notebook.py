from __future__ import annotations

import json
from pathlib import Path

import build_contact_gated_training_notebook as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "kernels" / "neural_stack_integration_r3"
OUT.mkdir(parents=True, exist_ok=True)


INTRO = r"""# ROGII Current-Only Neural Stack Integration — R3

This CV-only notebook tests the one useful result from R2: a trajectory MLP
trained from scratch on current-competition wells. It does **not** use external
weights and it does not create a competition submission.

The control is the existing target-free PF/beam/GR residual stack. Five outer
folds are grouped by well. Inside every outer fold, the trajectory MLP uses a
separate grouped early-stopping split. Its prediction is converted to TVT and
added only as a small distance-decayed correction to the OOF stack:

\[
T^{candidate}_i=T^{stack}_i + 0.10e^{-d_i/2400}
\left(T^{neural}_i-T^{stack}_i\right).
\]

Promotion requires a material pooled gain, at least four winning outer folds,
a majority of winning wells, no meaningful 600/1200-ft regression, and no
worst-decile degradation. Passing authorizes a separate full-training
inference notebook; it does not authorize a leaderboard submission by itself.
"""


CONTROL = r"""# Experiment control panel
import os

SEED = 20260721
N_SPLITS = 5
TRAIN_PF_PARTICLES = 120
TRAIN_PF_SEEDS = 8
N_JOBS = 4
LGB_ESTIMATORS = 800
STACK_SHRINK = 0.95
STACK_CLIP = 60.0

PAST_FT = 300
FUTURE_FT = 300
N_GRID = 60
CURRENT_STRIDE = 200
MAX_LOCAL_SLOPE = 0.60
MLP_EPOCHS = 25
MLP_PATIENCE = 5
BATCH_SIZE = 1024

PRIMARY_WEIGHT = 0.10
PRIMARY_TAU = 2400.0
HORIZONS = (300, 600, 1200, 2400, 4800)

LOCAL_DEBUG = not os.path.exists("/kaggle/input")
SMOKE_MODE = os.environ.get("ROGII_SMOKE", "0") == "1" or LOCAL_DEBUG
FORCE_CPU = os.environ.get("ROGII_FORCE_CPU", "0") == "1"
if SMOKE_MODE:
    N_SPLITS = 2
    TRAIN_PF_PARTICLES = 50
    TRAIN_PF_SEEDS = 2
    LGB_ESTIMATORS = 100
    CURRENT_STRIDE = 600
    MLP_EPOCHS = 2
    MLP_PATIENCE = 2
    BATCH_SIZE = 512
"""


DRIVER = r'''import copy
import gc
import glob
import json
import logging
import random
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

default_work = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
WORK = Path(os.environ.get("ROGII_OUTPUT_DIR", str(default_work)))
WORK.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rogii-neural-stack-r3")


def find_data_root():
    candidates = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("data"),
    ]
    for root in candidates:
        if (root / "train").exists() and (root / "sample_submission.csv").exists():
            return root
    if Path("/kaggle/input").exists():
        hits = list(Path("/kaggle/input").glob("**/train/*__horizontal_well.csv"))
        if hits:
            return hits[0].parent.parent
    raise FileNotFoundError("Could not locate competition data")


DATA = find_data_root()
all_paths = sorted((DATA / "train").glob("*__horizontal_well.csv"))
if SMOKE_MODE:
    all_paths = all_paths[:40]
selected_wells = np.asarray([p.name.split("__")[0] for p in all_paths])
assert len(selected_wells) >= N_SPLITS * 5

# Recent Kaggle PyTorch images can be incompatible with an assigned P100.
CUDA_VISIBLE = torch.cuda.is_available()
CUDA_COMPATIBLE = False
GPU_STATUS = "CUDA unavailable"
if CUDA_VISIBLE:
    try:
        capability = torch.cuda.get_device_capability(0)
        device_sm = 10 * capability[0] + capability[1]
        supported = [
            int(a.split("_")[1]) for a in torch.cuda.get_arch_list()
            if a.startswith("sm_") and a.split("_")[1].isdigit()
        ]
        minimum_sm = min(supported) if supported else 999
        CUDA_COMPATIBLE = device_sm >= minimum_sm
        GPU_STATUS = (
            f"{torch.cuda.get_device_name(0)} sm_{device_sm}; "
            f"PyTorch minimum sm_{minimum_sm}; compatible={CUDA_COMPATIBLE}"
        )
    except Exception as exc:
        GPU_STATUS = f"probe failed: {type(exc).__name__}: {exc}"
DEVICE = torch.device("cuda" if CUDA_COMPATIBLE and not FORCE_CPU else "cpu")
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.set_num_threads(min(8, os.cpu_count() or 4))
LGB_DEVICE = "gpu" if CUDA_VISIBLE and not FORCE_CPU and not LOCAL_DEBUG else "cpu"
log.info("data=%s wells=%d smoke=%s", DATA, len(selected_wells), SMOKE_MODE)
log.info("accelerator=%s torch_device=%s lgb_device=%s", GPU_STATUS, DEVICE, LGB_DEVICE)


def build_train_signals():
    t0 = time.time()
    frames = Parallel(
        n_jobs=N_JOBS,
        verbose=5,
        prefer="threads" if LOCAL_DEBUG else "processes",
    )(
        delayed(build_well)(str(path), True) for path in all_paths
    )
    frames = [f for f in frames if f is not None and len(f)]
    result = pd.concat(frames, ignore_index=True)
    log.info(
        "signals rows=%d wells=%d elapsed=%.1fs",
        len(result), result["well"].nunique(), time.time() - t0,
    )
    return result


train = build_train_signals()
selected_wells = np.asarray(sorted(train["well"].unique()))
DROP = {"well", "id", "target", "last_known_tvt"}
FEATURES = [c for c in train.columns if c not in DROP]
X_STACK = np.nan_to_num(
    train[FEATURES].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0
)
Y_STACK = train["target"].to_numpy(np.float32)
ROW_GROUPS = train["well"].to_numpy()


# Geometry-window representation used by the current-only R2 scratch control.
PAST_GRID = np.linspace(-(PAST_FT - 1), 0.0, N_GRID)
FUTURE_GRID = np.linspace(1.0, FUTURE_FT, N_GRID)


def clean_curve(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    order = np.argsort(x); x, y = x[order], y[order]
    unique = np.r_[True, np.diff(x) > 1e-8]
    return x[unique], y[unique]


def encode_window(past, future=None):
    past = np.asarray(past, float)
    coef = np.polyfit(PAST_GRID, past, 1)
    linear_past = np.polyval(coef, PAST_GRID)
    detrended = past - linear_past
    scale = max(float(np.std(detrended)), 0.50)
    level = detrended / scale
    d1 = np.gradient(level); d2 = np.gradient(d1)
    extras = np.array([
        np.clip(coef[0] / 0.10, -5.0, 5.0),
        np.log1p(scale), np.mean(np.abs(d1)), np.mean(np.abs(d2)),
    ])
    features = np.concatenate([level, d1, d2, extras]).astype(np.float32)
    linear_future = np.polyval(coef, FUTURE_GRID)
    if future is None:
        return features, linear_future, scale
    target = ((np.asarray(future, float) - linear_future) / scale).astype(np.float32)
    return features, target


def curve_windows(x, y):
    x, y = clean_curve(x, y)
    lo = int(np.ceil(x[0] + PAST_FT - 1))
    hi = int(np.floor(x[-1] - FUTURE_FT))
    xs, ys = [], []
    if hi < lo:
        return xs, ys
    for anchor in np.arange(lo, hi + 1, CURRENT_STRIDE):
        past = np.interp(anchor + PAST_GRID, x, y)
        future = np.interp(anchor + FUTURE_GRID, x, y)
        seq = np.r_[past, future]
        if np.all(np.isfinite(seq)) and np.quantile(np.abs(np.diff(seq)), 0.99) <= MAX_LOCAL_SLOPE:
            xx, yy = encode_window(past, future)
            xs.append(xx); ys.append(yy)
    return xs, ys


WELLS = {}
window_x, window_y, window_g = [], [], []
for number, path in enumerate(all_paths, 1):
    well = path.name.split("__")[0]
    if well not in set(selected_wells):
        continue
    df = pd.read_csv(path, usecols=["MD", "Z", "TVT", "TVT_input"])
    data = {c.lower(): df[c].to_numpy(float) for c in ["MD", "Z", "TVT", "TVT_input"]}
    WELLS[well] = data
    md_clean, u_clean = clean_curve(data["md"], data["tvt"] + data["z"])
    xx, yy = curve_windows(md_clean, u_clean)
    window_x.extend(xx); window_y.extend(yy); window_g.extend([well] * len(xx))
    if number % 100 == 0:
        log.info("geometry loaded %d/%d wells", number, len(all_paths))
X_GEOM = np.asarray(window_x, np.float32)
Y_GEOM = np.asarray(window_y, np.float32)
G_GEOM = np.asarray(window_g)
log.info("geometry windows=%s wells=%d", X_GEOM.shape, len(np.unique(G_GEOM)))
assert len(X_GEOM) > 100


class ResidualBlock(nn.Module):
    def __init__(self, width=256, dropout=0.08):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width * 2, width), nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.net(x)


class SurfaceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 256), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualBlock() for _ in range(4)])
        self.output = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, N_GRID))
    def forward(self, x):
        return self.output(self.blocks(self.input(x)))


def sequence_loss(pred, target):
    level = torch.nn.functional.smooth_l1_loss(pred, target)
    pd1 = pred[:, 1:] - pred[:, :-1]
    td1 = target[:, 1:] - target[:, :-1]
    slope = torch.nn.functional.smooth_l1_loss(pd1, td1)
    curve = torch.nn.functional.smooth_l1_loss(pd1[:, 1:] - pd1[:, :-1], td1[:, 1:] - td1[:, :-1])
    return level + 0.20 * slope + 0.05 * curve


def data_loader(x, y, shuffle):
    ds = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=DEVICE.type == "cuda")


@torch.no_grad()
def neural_predict(model, x):
    model.eval()
    outputs = []
    dummy = np.zeros((len(x), N_GRID), np.float32)
    for xb, _ in data_loader(np.asarray(x, np.float32), dummy, False):
        outputs.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.vstack(outputs)


def train_geometry(x_train, y_train, x_valid, y_valid, tag):
    torch.manual_seed(SEED)
    model = SurfaceMLP(x_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MLP_EPOCHS)
    best_score, best_state, best_epoch, stale = np.inf, None, 0, 0
    for epoch in range(1, MLP_EPOCHS + 1):
        model.train(); losses = []
        for xb, yb in data_loader(x_train, y_train, True):
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            if not SMOKE_MODE:
                xb = xb + 0.003 * torch.randn_like(xb)
            optimizer.zero_grad(set_to_none=True)
            loss = sequence_loss(model(xb), yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        scheduler.step()
        pred = neural_predict(model, x_valid)
        score = float(np.sqrt(np.mean((pred - y_valid) ** 2)))
        log.info("%s epoch=%02d train=%.5f valid=%.5f", tag, epoch, np.mean(losses), score)
        if score < best_score - 1e-5:
            best_score = score
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            best_epoch = epoch; stale = 0
        else:
            stale += 1
            if stale >= MLP_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, best_score


@torch.no_grad()
def predict_next(model, past):
    features, linear_future, scale = encode_window(past)
    residual = neural_predict(model, features[None, :])[0]
    return linear_future + scale * residual


def recursive_forecast(model, past, deltas):
    deltas = np.asarray(deltas, float)
    state = np.asarray(past, float)
    all_delta, all_value, start = [], [], 0.0
    while start < float(np.max(deltas)):
        future = predict_next(model, state)
        all_delta.extend(start + FUTURE_GRID); all_value.extend(future)
        state = future; start += FUTURE_FT
    return np.interp(deltas, np.asarray(all_delta), np.asarray(all_value))


def geometry_delta_for_well(model, well):
    data = WELLS[well]
    visible = np.isfinite(data["tvt_input"])
    last = np.flatnonzero(visible)[-1]
    anchor_md = data["md"][last]
    u_input = data["tvt_input"] + data["z"]
    past = np.interp(anchor_md + PAST_GRID, data["md"][: last + 1], u_input[: last + 1])
    hidden = np.arange(last + 1, len(data["md"]))
    deltas = data["md"][hidden] - anchor_md
    pred_u = recursive_forecast(model, past, deltas)
    pred_tvt = pred_u - data["z"][hidden]
    return np.clip(pred_tvt - data["tvt_input"][last], -80.0, 80.0)


LGB_PARAMS = dict(
    objective="regression",
    n_estimators=LGB_ESTIMATORS,
    learning_rate=0.02,
    num_leaves=127,
    min_child_samples=100,
    subsample=0.80,
    subsample_freq=1,
    colsample_bytree=0.70,
    reg_lambda=5.0,
    reg_alpha=1.0,
    verbose=-1,
    n_jobs=-1,
    random_state=SEED,
    device_type=LGB_DEVICE,
)


stack_oof = np.full(len(train), np.nan, float)
geometry_oof = np.full(len(train), np.nan, float)
fold_rows = []
outer = GroupKFold(n_splits=N_SPLITS)
for fold, (well_fit_idx, well_valid_idx) in enumerate(
    outer.split(selected_wells, groups=selected_wells), 1
):
    t0 = time.time()
    fit_wells = selected_wells[well_fit_idx]
    valid_wells = selected_wells[well_valid_idx]
    fit_rows = np.flatnonzero(np.isin(ROW_GROUPS, fit_wells))
    valid_rows = np.flatnonzero(np.isin(ROW_GROUPS, valid_wells))

    params = dict(LGB_PARAMS); params["random_state"] = SEED + fold
    try:
        stack_model = lgb.LGBMRegressor(**params).fit(X_STACK[fit_rows], Y_STACK[fit_rows])
    except Exception as exc:
        log.warning("fold%d LightGBM GPU failed (%s); retrying CPU", fold, exc)
        params["device_type"] = "cpu"
        stack_model = lgb.LGBMRegressor(**params).fit(X_STACK[fit_rows], Y_STACK[fit_rows])
    stack_oof[valid_rows] = np.clip(
        stack_model.predict(X_STACK[valid_rows]) * STACK_SHRINK, -STACK_CLIP, STACK_CLIP
    )

    geom_mask = np.isin(G_GEOM, fit_wells)
    xg, yg, gg = X_GEOM[geom_mask], Y_GEOM[geom_mask], G_GEOM[geom_mask]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=SEED + fold)
    inner_fit, inner_valid = next(inner.split(xg, yg, gg))
    geom_model, best_epoch, inner_score = train_geometry(
        xg[inner_fit], yg[inner_fit], xg[inner_valid], yg[inner_valid], f"fold{fold}_geometry"
    )
    for well in valid_wells:
        idx = train.index[train["well"].eq(well)].to_numpy()
        pred = geometry_delta_for_well(geom_model, str(well))
        if len(pred) != len(idx):
            raise RuntimeError(f"geometry row mismatch {well}: {len(pred)} != {len(idx)}")
        geometry_oof[idx] = pred
    fold_rows.append({
        "fold": fold,
        "fit_wells": len(fit_wells),
        "valid_wells": len(valid_wells),
        "geometry_best_epoch": best_epoch,
        "geometry_inner_rmse": inner_score,
        "seconds": time.time() - t0,
    })
    log.info("fold=%d complete elapsed=%.1fs", fold, time.time() - t0)
    del stack_model, geom_model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

assert np.isfinite(stack_oof).all() and np.isfinite(geometry_oof).all()
distance = train["d_md"].to_numpy(float)
decay = np.exp(-np.maximum(distance, 0.0) / PRIMARY_TAU)
selector = 0.85 * train["pf_d"].to_numpy(float) + 0.15 * train["beam_d"].to_numpy(float)
CANDIDATES = {
    "stack_control": stack_oof,
    "geometry_neural": geometry_oof,
    "selector_control": selector,
    "blend_fixed_w005": stack_oof + 0.05 * (geometry_oof - stack_oof),
    "blend_fixed_w010": stack_oof + 0.10 * (geometry_oof - stack_oof),
    "blend_decay_w005": stack_oof + 0.05 * decay * (geometry_oof - stack_oof),
    "blend_decay_w010_primary": stack_oof + PRIMARY_WEIGHT * decay * (geometry_oof - stack_oof),
    "blend_decay_w020": stack_oof + 0.20 * decay * (geometry_oof - stack_oof),
}


def pooled_rmse(actual, pred):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2)))


summary_rows, well_rows = [], []
horizon_defs = [(str(h), distance <= h) for h in HORIZONS] + [("all", np.ones(len(train), bool))]
for name, pred in CANDIDATES.items():
    for horizon, mask in horizon_defs:
        frame = pd.DataFrame({
            "well": ROW_GROUPS[mask],
            "fold": train.loc[mask, "well"].map({
                w: f for f, (_, vi) in enumerate(outer.split(selected_wells, groups=selected_wells), 1)
                for w in selected_wells[vi]
            }).to_numpy(),
            "err2": (Y_STACK[mask] - pred[mask]) ** 2,
        })
        grouped = frame.groupby(["fold", "well"], as_index=False)["err2"].agg(["sum", "count"]).reset_index()
        grouped["rmse"] = np.sqrt(grouped["sum"] / grouped["count"])
        for row in grouped.itertuples(index=False):
            well_rows.append({
                "candidate": name, "horizon": horizon, "fold": int(row.fold),
                "well": row.well, "rows": int(row.count), "sse": float(row.sum),
                "rmse": float(row.rmse),
            })
        summary_rows.append({
            "candidate": name,
            "horizon": horizon,
            "rows": int(mask.sum()),
            "wells": int(grouped["well"].nunique()),
            "pooled_rmse": pooled_rmse(Y_STACK[mask], pred[mask]),
            "mean_well_rmse": float(grouped["rmse"].mean()),
            "median_well_rmse": float(grouped["rmse"].median()),
            "p90_well_rmse": float(grouped["rmse"].quantile(0.90)),
        })

summary = pd.DataFrame(summary_rows)
well_metrics = pd.DataFrame(well_rows)
pd.DataFrame(fold_rows).to_csv(WORK / "training_folds.csv", index=False)
summary.to_csv(WORK / "cv_summary.csv", index=False)
well_metrics.to_csv(WORK / "cv_well_metrics.csv", index=False)


def metric(candidate, horizon, column="pooled_rmse"):
    row = summary[(summary["candidate"] == candidate) & (summary["horizon"] == str(horizon))]
    return float(row.iloc[0][column])


PRIMARY = "blend_decay_w010_primary"
all_primary = metric(PRIMARY, "all")
all_stack = metric("stack_control", "all")
wm = well_metrics[well_metrics["horizon"].eq("all")].pivot(
    index=["fold", "well"], columns="candidate", values="rmse"
)
well_win_rate = float((wm[PRIMARY] < wm["stack_control"]).mean())
fold_compare = []
for fold in sorted(train["well"].map({
    w: f for f, (_, vi) in enumerate(GroupKFold(n_splits=N_SPLITS).split(selected_wells, groups=selected_wells), 1)
    for w in selected_wells[vi]
}).dropna().unique()):
    fold = int(fold)
    mask = train["well"].map({
        w: f for f, (_, vi) in enumerate(GroupKFold(n_splits=N_SPLITS).split(selected_wells, groups=selected_wells), 1)
        for w in selected_wells[vi]
    }).to_numpy() == fold
    fold_compare.append({
        "fold": fold,
        "stack_rmse": pooled_rmse(Y_STACK[mask], CANDIDATES["stack_control"][mask]),
        "primary_rmse": pooled_rmse(Y_STACK[mask], CANDIDATES[PRIMARY][mask]),
    })
fold_compare = pd.DataFrame(fold_compare)
fold_compare["primary_wins"] = fold_compare["primary_rmse"] < fold_compare["stack_rmse"]
fold_compare.to_csv(WORK / "fold_comparison.csv", index=False)
fold_wins = int(fold_compare["primary_wins"].sum())

checks = {
    "pooled_gain_at_least_half_percent": all_primary <= 0.995 * all_stack,
    "absolute_gain_at_least_point10": all_stack - all_primary >= 0.10,
    "wins_majority_of_wells": well_win_rate >= 0.52,
    "wins_at_least_four_folds": fold_wins >= 4,
    "safe_at_600ft": metric(PRIMARY, 600) <= 1.002 * metric("stack_control", 600),
    "safe_at_1200ft": metric(PRIMARY, 1200) <= 1.002 * metric("stack_control", 1200),
    "worst_decile_safe": metric(PRIMARY, "all", "p90_well_rmse") <= metric("stack_control", "all", "p90_well_rmse") + 0.05,
}
gate = {
    "passed": bool(all(checks.values())),
    "primary_candidate": PRIMARY,
    "checks": {k: bool(v) for k, v in checks.items()},
    "metrics": {
        "stack_all_rmse": all_stack,
        "primary_all_rmse": all_primary,
        "absolute_gain": all_stack - all_primary,
        "relative_gain": 1.0 - all_primary / all_stack,
        "well_win_rate": well_win_rate,
        "fold_wins": fold_wins,
        "stack_600": metric("stack_control", 600),
        "primary_600": metric(PRIMARY, 600),
        "stack_1200": metric("stack_control", 1200),
        "primary_1200": metric(PRIMARY, 1200),
        "stack_p90": metric("stack_control", "all", "p90_well_rmse"),
        "primary_p90": metric(PRIMARY, "all", "p90_well_rmse"),
    },
    "next_step": (
        "Build full-training inference integration; still require submission audit."
        if all(checks.values()) else
        "Reject neural integration; keep current contact/PF submission unchanged."
    ),
}
(WORK / "quality_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
log.info("QUALITY GATE\n%s", json.dumps(gate, indent=2))

fig, ax = plt.subplots(figsize=(10, 6))
for name in ["stack_control", "geometry_neural", PRIMARY, "blend_decay_w020"]:
    part = summary[(summary["candidate"] == name) & (summary["horizon"] != "all")].copy()
    part["h"] = part["horizon"].astype(int)
    part = part.sort_values("h")
    ax.plot(part["h"], part["pooled_rmse"], marker="o", label=name)
ax.set_xlabel("Horizon after visible cut (ft)"); ax.set_ylabel("Grouped OOF RMSE (ft)")
ax.set_title("R3 current-only neural correction vs existing stack")
ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
fig.savefig(WORK / "cv_neural_stack_horizons.png", dpi=160)
plt.close(fig)

# CV-only invariant: this experiment is incapable of consuming a submission slot.
assert not (WORK / "submission.csv").exists()
print(summary.to_string(index=False))
print("R3 complete. Gate passed:", gate["passed"])
'''


NOTEBOOK = {
    "cells": [
        base.markdown(INTRO),
        base.code(CONTROL, hidden=False),
        base.code(base.PF),
        base.code(base.BEAM),
        base.code(base.SIGNALS),
        base.code(DRIVER),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


METADATA = {
    "id": "boltuzamaki/rogii-current-neural-stack-integration-r3",
    "title": "ROGII Current Neural Stack Integration R3",
    "code_file": "rogii-current-neural-stack-integration-r3.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "dataset_sources": [],
    "competition_sources": ["rogii-wellbore-geology-prediction"],
    "kernel_sources": [],
}


(OUT / METADATA["code_file"]).write_text(json.dumps(NOTEBOOK, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps(METADATA, indent=2), encoding="utf-8")
print(OUT / METADATA["code_file"])
print(OUT / "kernel-metadata.json")
