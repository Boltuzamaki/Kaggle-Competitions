from __future__ import annotations

import json
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "kernels" / "external_pretrain_gpu_r2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.splitlines(True),
    }


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
        """# ROGII External Geometry Transfer Learning — GPU R2

R1 showed that naïvely concatenating old and current geology windows produces
only a 0.10% short-horizon gain. R2 tests a stricter hypothesis: **pretrain a
non-linear surface representation on the older Geology Forecast Challenge,
then fine-tune only on current-competition wells**.

## Leakage and promotion policy

- External data comes from the original Kaggle competition mount.
- Only `train_raw/*.csv` is used; duplicated processed realization targets are
  excluded.
- Outer CV is grouped by current well.
- External pretraining validation is grouped by original external curve.
- Fine-tuning early stopping uses only an inner split of outer-training wells.
- Transfer is compared with the identical network trained from scratch and a
  current-only Ridge control.
- No `submission.csv` is created.
- A model package is exported only if every predeclared transfer and safety gate
  passes. A pass authorizes stack-integration CV, not a leaderboard submission.
"""
    ),
    code(
        """# Deterministic configuration
import copy
import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 20260721
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SMOKE_MODE = os.environ.get("ROGII_SMOKE", "0") == "1"
FORCE_CPU = os.environ.get("ROGII_FORCE_CPU", "0") == "1"

# Kaggle can still assign a Tesla P100 (sm_60), while recent PyTorch images may
# ship kernels only for sm_70 and newer. Detect that mismatch before the first
# tensor operation and use CPU instead of crashing halfway through setup.
CUDA_VISIBLE = torch.cuda.is_available()
CUDA_CAPABILITY = None
TORCH_CUDA_ARCHES = []
CUDA_COMPATIBLE = False
GPU_STATUS = "CUDA is not available"
if CUDA_VISIBLE:
    try:
        CUDA_CAPABILITY = torch.cuda.get_device_capability(0)
        TORCH_CUDA_ARCHES = torch.cuda.get_arch_list()
        supported_sms = [
            int(arch.split("_")[1])
            for arch in TORCH_CUDA_ARCHES
            if arch.startswith("sm_") and arch.split("_")[1].isdigit()
        ]
        device_sm = 10 * CUDA_CAPABILITY[0] + CUDA_CAPABILITY[1]
        minimum_sm = min(supported_sms) if supported_sms else 999
        CUDA_COMPATIBLE = device_sm >= minimum_sm
        GPU_STATUS = (
            f"{torch.cuda.get_device_name(0)} sm_{device_sm}; "
            f"PyTorch minimum sm_{minimum_sm}; compatible={CUDA_COMPATIBLE}"
        )
    except Exception as exc:
        GPU_STATUS = f"CUDA compatibility probe failed: {type(exc).__name__}: {exc}"

DEVICE = torch.device("cuda" if CUDA_COMPATIBLE and not FORCE_CPU else "cpu")
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.set_num_threads(min(8, os.cpu_count() or 4))
N_SPLITS = 2 if SMOKE_MODE else 5
MAX_CURRENT_WELLS = 60 if SMOKE_MODE else None
CURRENT_STRIDE = 600 if SMOKE_MODE else 200
EXTERNAL_STRIDE = 600 if SMOKE_MODE else 50
PRETRAIN_EPOCHS = 2 if SMOKE_MODE else 30
FINETUNE_EPOCHS = 2 if SMOKE_MODE else 25
PRETRAIN_PATIENCE = 2 if SMOKE_MODE else 6
FINETUNE_PATIENCE = 2 if SMOKE_MODE else 5
BATCH_SIZE = 512 if SMOKE_MODE else 1024
PAST_FT = 300
FUTURE_FT = 300
N_GRID = 60
MAX_LOCAL_SLOPE = 0.60
HORIZONS = [300, 600, 1200, 2400, 4800]
DECAY_TAU = 2400.0

# Transfer must be material, consistent, and safe.
GATE_TRANSFER_RELATIVE = 0.995
GATE_RIDGE_TOLERANCE = 1.01
GATE_MIN_TRANSFER_WIN_RATE = 0.52
GATE_LONG_RANGE_TO_LINEAR = 1.005

default_work = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
WORK = Path(os.environ.get("ROGII_OUTPUT_DIR", str(default_work)))
WORK.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rogii-transfer-r2")
log.info("accelerator probe: %s; force_cpu=%s", GPU_STATUS, FORCE_CPU)
log.info(
    "device=%s smoke=%s folds=%d pretrain_epochs=%d finetune_epochs=%d",
    DEVICE, SMOKE_MODE, N_SPLITS, PRETRAIN_EPOCHS, FINETUNE_EPOCHS,
)
"""
    ),
    code(
        """# Locate original competition mounts
def find_current_data():
    local = [base / "data" for base in [Path.cwd(), *Path.cwd().parents]]
    candidates = [
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    ] + local
    for candidate in candidates:
        if (candidate / "train").exists():
            return candidate
    matches = list(Path("/kaggle/input").glob("**/train/*__horizontal_well.csv")) if Path("/kaggle/input").exists() else []
    if matches:
        return matches[0].parent.parent
    raise FileNotFoundError("Current competition data not found")


def find_external_data():
    local = [
        base / "external_data/geology_forecast_challenge/extracted/data"
        for base in [Path.cwd(), *Path.cwd().parents]
    ]
    candidates = [
        Path("/kaggle/input/competitions/geology-forecast-challenge-open/data"),
        Path("/kaggle/input/geology-forecast-challenge-open/data"),
    ] + local
    for candidate in candidates:
        if (candidate / "train_raw").exists():
            return candidate
    matches = list(Path("/kaggle/input").glob("**/train_raw/*.csv")) if Path("/kaggle/input").exists() else []
    if matches:
        return matches[0].parent.parent
    raise FileNotFoundError("External Geology Forecast data not found")


CURRENT_DATA = find_current_data()
EXTERNAL_DATA = find_external_data()
current_files = sorted((CURRENT_DATA / "train").glob("*__horizontal_well.csv"))
external_files = sorted((EXTERNAL_DATA / "train_raw").glob("*.csv"))
if MAX_CURRENT_WELLS:
    current_files = current_files[:MAX_CURRENT_WELLS]
log.info("current=%s (%d wells)", CURRENT_DATA, len(current_files))
log.info("external=%s (%d curves)", EXTERNAL_DATA, len(external_files))
assert len(current_files) >= N_SPLITS * 5
assert len(external_files) >= 50
"""
    ),
    code(
        """# Scale-invariant residual windows
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


def encode_window(past, future=None):
    past = np.asarray(past, float)
    coef = np.polyfit(PAST_GRID, past, 1)
    linear_past = np.polyval(coef, PAST_GRID)
    detrended = past - linear_past
    scale = max(float(np.std(detrended)), 0.50)
    level = detrended / scale
    d1 = np.gradient(level)
    d2 = np.gradient(d1)
    extras = np.array([
        np.clip(coef[0] / 0.10, -5.0, 5.0),
        np.log1p(scale),
        np.mean(np.abs(d1)),
        np.mean(np.abs(d2)),
    ])
    features = np.concatenate([level, d1, d2, extras]).astype(np.float32)
    linear_future = np.polyval(coef, FUTURE_GRID)
    if future is None:
        return features, linear_future, scale
    target = ((np.asarray(future, float) - linear_future) / scale).astype(np.float32)
    return features, target


def valid_window(past, future):
    seq = np.r_[past, future]
    return bool(
        np.all(np.isfinite(seq))
        and np.quantile(np.abs(np.diff(seq)), 0.99) <= MAX_LOCAL_SLOPE
    )


def curve_windows(x, y, stride):
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
            xx, yy = encode_window(past, future)
            xs.append(xx); ys.append(yy)
    return xs, ys


def load_external_windows():
    xs, ys, groups = [], [], []
    for path in external_files:
        df = pd.read_csv(path, usecols=["VS_APPROX_adjusted", "HORIZON_Z_adjusted"]).dropna()
        xx, yy = curve_windows(
            df["VS_APPROX_adjusted"].to_numpy(float),
            df["HORIZON_Z_adjusted"].to_numpy(float),
            EXTERNAL_STRIDE,
        )
        xs.extend(xx); ys.extend(yy); groups.extend([path.stem] * len(xx))
    return np.asarray(xs), np.asarray(ys), np.asarray(groups)


def load_current_windows():
    wells = {}
    xs, ys, groups = [], [], []
    for number, path in enumerate(current_files, 1):
        well = path.name.split("__")[0]
        df = pd.read_csv(path, usecols=["MD", "Z", "TVT", "TVT_input"])
        md = df["MD"].to_numpy(float)
        z = df["Z"].to_numpy(float)
        tvt = df["TVT"].to_numpy(float)
        tvt_input = df["TVT_input"].to_numpy(float)
        md_clean, u_clean = clean_curve(md, tvt + z)
        xx, yy = curve_windows(md_clean, u_clean, CURRENT_STRIDE)
        xs.extend(xx); ys.extend(yy); groups.extend([well] * len(xx))
        wells[well] = {"md": md, "z": z, "tvt": tvt, "tvt_input": tvt_input}
        if number % 100 == 0:
            log.info("loaded %d/%d current wells", number, len(current_files))
    return wells, np.asarray(xs), np.asarray(ys), np.asarray(groups)


t0 = time.time()
X_EXT, Y_EXT, G_EXT = load_external_windows()
WELLS, X_CUR, Y_CUR, G_CUR = load_current_windows()
log.info(
    "external windows=%s curves=%d | current windows=%s wells=%d | %.1fs",
    X_EXT.shape, len(np.unique(G_EXT)), X_CUR.shape, len(np.unique(G_CUR)), time.time() - t0,
)
assert len(X_EXT) > 500 and len(X_CUR) > 100
"""
    ),
    code(
        """# Neural forecaster, training loop, and Ridge control
class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class SurfaceMLP(nn.Module):
    def __init__(self, input_dim, output_dim=N_GRID, width=256, blocks=4, dropout=0.08):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(blocks)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, output_dim))

    def forward(self, x):
        return self.output(self.blocks(self.input(x)))


def sequence_loss(pred, target):
    level = torch.nn.functional.smooth_l1_loss(pred, target)
    slope = torch.nn.functional.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
    pred_d1 = pred[:, 1:] - pred[:, :-1]
    true_d1 = target[:, 1:] - target[:, :-1]
    curvature = torch.nn.functional.smooth_l1_loss(
        pred_d1[:, 1:] - pred_d1[:, :-1], true_d1[:, 1:] - true_d1[:, :-1]
    )
    return level + 0.20 * slope + 0.05 * curvature


def loader(x, y, shuffle):
    ds = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=DEVICE.type == "cuda")


@torch.no_grad()
def neural_rmse(model, x, y):
    model.eval()
    preds = []
    for xb, _ in loader(x, y, False):
        preds.append(model(xb.to(DEVICE)).cpu().numpy())
    pred = np.vstack(preds)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def train_neural(x_train, y_train, x_valid, y_valid, max_epochs, patience, lr, tag, initial_state=None):
    torch.manual_seed(SEED)
    model = SurfaceMLP(x_train.shape[1]).to(DEVICE)
    if initial_state is not None:
        model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    best_score = np.inf
    best_state = None
    best_epoch = 0
    stale = 0
    train_loader = loader(x_train, y_train, True)
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            if not SMOKE_MODE:
                xb = xb + 0.003 * torch.randn_like(xb)
            optimizer.zero_grad(set_to_none=True)
            loss = sequence_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        score = neural_rmse(model, x_valid, y_valid)
        log.info("%s epoch=%02d train=%.5f valid_rmse=%.5f", tag, epoch, np.mean(losses), score)
        if score < best_score - 1e-5:
            best_score = score
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log.info("%s early_stop epoch=%d best_epoch=%d", tag, epoch, best_epoch)
                break
    model.load_state_dict(best_state)
    return model, best_state, best_epoch, best_score


class RidgeControl:
    def __init__(self, alpha=100.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, x, y):
        self.model.fit(x, y)
        return self

    def predict(self, x):
        return self.model.predict(x)


@torch.no_grad()
def neural_predict(model, x):
    model.eval()
    tensor = torch.from_numpy(np.asarray(x, np.float32)).to(DEVICE)
    return model(tensor).cpu().numpy()
"""
    ),
    code(
        """# Recursive official-cut evaluation
def predict_next(predictor, past, is_neural):
    features, linear_future, scale = encode_window(past)
    if is_neural:
        residual = neural_predict(predictor, features[None, :])[0]
    else:
        residual = predictor.predict(features[None, :])[0]
    return linear_future + scale * residual


def recursive_forecast(predictor, past, deltas, is_neural):
    deltas = np.asarray(deltas, float)
    state = np.asarray(past, float)
    all_delta, all_value = [], []
    start = 0.0
    while start < float(np.max(deltas)):
        future = predict_next(predictor, state, is_neural)
        all_delta.extend(start + FUTURE_GRID)
        all_value.extend(future)
        state = future
        start += FUTURE_FT
    return np.interp(deltas, np.asarray(all_delta), np.asarray(all_value))


def official_cut_context(data):
    md = data["md"]
    visible = np.isfinite(data["tvt_input"])
    if not visible.any() or visible.all():
        return None
    last = np.flatnonzero(visible)[-1]
    anchor = md[last]
    if anchor - md[0] < PAST_FT - 1:
        return None
    hidden = np.arange(last + 1, len(md))
    if not len(hidden):
        return None
    u_input = data["tvt_input"] + data["z"]
    u_true = data["tvt"] + data["z"]
    past = np.interp(anchor + PAST_GRID, md[: last + 1], u_input[: last + 1])
    delta = md[hidden] - anchor
    truth = u_true[hidden]
    linear_coef = np.polyfit(PAST_GRID, past, 1)
    linear = np.polyval(linear_coef, delta)
    return past, delta, truth, linear


def evaluate(predictor, wells, fold, candidate, is_neural, decay_tau=None):
    rows = []
    for well in wells:
        context = official_cut_context(WELLS[well])
        if context is None:
            continue
        past, delta, truth, linear = context
        raw = recursive_forecast(predictor, past, delta, is_neural)
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
            lin2 = (linear[mask] - truth[mask]) ** 2
            rows.append({
                "fold": fold,
                "well": well,
                "candidate": candidate,
                "horizon": horizon,
                "n_rows": int(mask.sum()),
                "sse": float(err2.sum()),
                "rmse": float(np.sqrt(err2.mean())),
                "linear_sse": float(lin2.sum()),
                "linear_rmse": float(np.sqrt(lin2.mean())),
            })
    return rows
"""
    ),
    code(
        """# Group-safe external pretraining and outer CV
ext_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
ext_train_idx, ext_valid_idx = next(ext_split.split(X_EXT, Y_EXT, G_EXT))
pretrain_model, PRETRAIN_STATE, PRETRAIN_BEST_EPOCH, PRETRAIN_SCORE = train_neural(
    X_EXT[ext_train_idx], Y_EXT[ext_train_idx],
    X_EXT[ext_valid_idx], Y_EXT[ext_valid_idx],
    PRETRAIN_EPOCHS, PRETRAIN_PATIENCE, 1e-3, "external_pretrain",
)
log.info(
    "external pretraining complete best_epoch=%d grouped_valid_rmse=%.5f",
    PRETRAIN_BEST_EPOCH, PRETRAIN_SCORE,
)
del pretrain_model
gc.collect()

all_wells = np.asarray(sorted(WELLS))
outer = GroupKFold(n_splits=N_SPLITS)
metric_rows = []
training_rows = []

for fold, (outer_train_idx, outer_valid_idx) in enumerate(outer.split(all_wells, groups=all_wells), 1):
    fold_start = time.time()
    outer_train_wells = all_wells[outer_train_idx]
    outer_valid_wells = all_wells[outer_valid_idx]
    outer_mask = np.isin(G_CUR, outer_train_wells)
    x_outer, y_outer, g_outer = X_CUR[outer_mask], Y_CUR[outer_mask], G_CUR[outer_mask]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=SEED + fold)
    inner_train, inner_valid = next(inner.split(x_outer, y_outer, g_outer))

    scratch_model, _, scratch_epoch, scratch_score = train_neural(
        x_outer[inner_train], y_outer[inner_train],
        x_outer[inner_valid], y_outer[inner_valid],
        FINETUNE_EPOCHS, FINETUNE_PATIENCE, 1e-3, f"fold{fold}_scratch",
    )
    transfer_model, _, transfer_epoch, transfer_score = train_neural(
        x_outer[inner_train], y_outer[inner_train],
        x_outer[inner_valid], y_outer[inner_valid],
        FINETUNE_EPOCHS, FINETUNE_PATIENCE, 3e-4, f"fold{fold}_transfer",
        initial_state=PRETRAIN_STATE,
    )
    ridge = RidgeControl(100.0).fit(x_outer[inner_train], y_outer[inner_train])

    training_rows.append({
        "fold": fold,
        "outer_train_wells": len(outer_train_wells),
        "outer_valid_wells": len(outer_valid_wells),
        "inner_train_windows": len(inner_train),
        "inner_valid_windows": len(inner_valid),
        "scratch_best_epoch": scratch_epoch,
        "scratch_inner_rmse": scratch_score,
        "transfer_best_epoch": transfer_epoch,
        "transfer_inner_rmse": transfer_score,
    })

    for model, name, neural in [
        (scratch_model, "scratch_raw", True),
        (transfer_model, "transfer_raw", True),
        (ridge, "ridge_raw", False),
    ]:
        metric_rows.extend(evaluate(model, outer_valid_wells, fold, name, neural))
        metric_rows.extend(evaluate(model, outer_valid_wells, fold, name.replace("_raw", "_decay2400"), neural, DECAY_TAU))

    log.info(
        "fold=%d complete train_wells=%d valid_wells=%d elapsed=%.1fs",
        fold, len(outer_train_wells), len(outer_valid_wells), time.time() - fold_start,
    )
    del scratch_model, transfer_model, ridge
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

well_metrics = pd.DataFrame(metric_rows)
training_log = pd.DataFrame(training_rows)
well_metrics.to_csv(WORK / "cv_well_metrics.csv", index=False)
training_log.to_csv(WORK / "training_folds.csv", index=False)
"""
    ),
    code(
        """# OOF aggregation and hard quality gate
def aggregate(frame):
    rows = []
    for (candidate, horizon), group in frame.groupby(["candidate", "horizon"]):
        rows.append({
            "candidate": candidate,
            "horizon": int(horizon),
            "wells": int(group.well.nunique()),
            "rows": int(group.n_rows.sum()),
            "row_weighted_rmse": float(np.sqrt(group.sse.sum() / group.n_rows.sum())),
            "mean_well_rmse": float(group.rmse.mean()),
            "median_well_rmse": float(group.rmse.median()),
            "linear_row_weighted_rmse": float(np.sqrt(group.linear_sse.sum() / group.n_rows.sum())),
        })
    return pd.DataFrame(rows).sort_values(["horizon", "row_weighted_rmse"])


summary = aggregate(well_metrics)
summary.to_csv(WORK / "cv_summary.csv", index=False)
display(summary)

paired = well_metrics.pivot_table(
    index=["fold", "well", "horizon"], columns="candidate", values="rmse", aggfunc="first"
).reset_index()
paired["transfer_beats_scratch_raw"] = paired["transfer_raw"] < paired["scratch_raw"]
paired["transfer_beats_scratch_decay"] = paired["transfer_decay2400"] < paired["scratch_decay2400"]
paired.to_csv(WORK / "paired_well_comparison.csv", index=False)


def m(candidate, horizon, column="row_weighted_rmse"):
    row = summary[(summary.candidate == candidate) & (summary.horizon == horizon)]
    return float(row.iloc[0][column]) if len(row) else np.inf


def win_rate(column, horizon):
    part = paired[paired.horizon == horizon]
    return float(part[column].mean())


checks = {
    "transfer_beats_scratch_300": m("transfer_raw", 300) <= GATE_TRANSFER_RELATIVE * m("scratch_raw", 300),
    "transfer_beats_scratch_600": m("transfer_raw", 600) <= GATE_TRANSFER_RELATIVE * m("scratch_raw", 600),
    "transfer_decay_beats_scratch_1200": m("transfer_decay2400", 1200) <= GATE_TRANSFER_RELATIVE * m("scratch_decay2400", 1200),
    "transfer_decay_beats_scratch_2400": m("transfer_decay2400", 2400) <= GATE_TRANSFER_RELATIVE * m("scratch_decay2400", 2400),
    "transfer_competitive_with_ridge_300": m("transfer_raw", 300) <= GATE_RIDGE_TOLERANCE * m("ridge_raw", 300),
    "transfer_competitive_with_ridge_600": m("transfer_raw", 600) <= GATE_RIDGE_TOLERANCE * m("ridge_raw", 600),
    "transfer_well_win_rate_600": win_rate("transfer_beats_scratch_raw", 600) >= GATE_MIN_TRANSFER_WIN_RATE,
    "long_range_safe": m("transfer_decay2400", 4800) <= GATE_LONG_RANGE_TO_LINEAR * m("transfer_decay2400", 4800, "linear_row_weighted_rmse"),
}
gate = {
    "passed": bool(all(checks.values())),
    "checks": checks,
    "pretraining": {
        "best_epoch": PRETRAIN_BEST_EPOCH,
        "external_group_valid_rmse": PRETRAIN_SCORE,
    },
    "metrics": {
        "scratch_300": m("scratch_raw", 300),
        "transfer_300": m("transfer_raw", 300),
        "ridge_300": m("ridge_raw", 300),
        "scratch_600": m("scratch_raw", 600),
        "transfer_600": m("transfer_raw", 600),
        "ridge_600": m("ridge_raw", 600),
        "scratch_decay_1200": m("scratch_decay2400", 1200),
        "transfer_decay_1200": m("transfer_decay2400", 1200),
        "scratch_decay_2400": m("scratch_decay2400", 2400),
        "transfer_decay_2400": m("transfer_decay2400", 2400),
        "transfer_decay_4800": m("transfer_decay2400", 4800),
        "linear_4800": m("transfer_decay2400", 4800, "linear_row_weighted_rmse"),
        "transfer_win_rate_600": win_rate("transfer_beats_scratch_raw", 600),
    },
    "next_step": (
        "Build stack-OOF integration CV; no leaderboard submission yet."
        if all(checks.values())
        else "Reject transfer model; do not integrate or submit."
    ),
}
(WORK / "quality_gate.json").write_text(json.dumps(gate, indent=2))
log.info("QUALITY GATE\\n%s", json.dumps(gate, indent=2))

fig, ax = plt.subplots(figsize=(10, 6))
for name in ["ridge_raw", "scratch_raw", "transfer_raw", "scratch_decay2400", "transfer_decay2400"]:
    part = summary[summary.candidate == name].sort_values("horizon")
    ax.plot(part.horizon, part.row_weighted_rmse, marker="o", label=name)
linear = summary[summary.candidate == "transfer_decay2400"].sort_values("horizon")
ax.plot(linear.horizon, linear.linear_row_weighted_rmse, marker="o", linestyle="--", label="linear")
ax.set(xlabel="Forecast distance (ft)", ylabel="Row-weighted RMSE (ft)", title="External pretraining vs scratch: grouped OOF")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(WORK / "cv_transfer_horizons.png", dpi=160)
plt.show()
"""
    ),
    code(
        """# Export a final transfer model only after every gate passes
def fit_fixed_epochs(model, x, y, epochs, lr):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_loader = loader(x, y, True)
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = sequence_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        log.info("final_fit epoch=%d/%d", epoch, epochs)
    return model


if gate["passed"]:
    final_epochs = max(1, int(round(training_log.transfer_best_epoch.median())))
    final_model = SurfaceMLP(X_CUR.shape[1])
    final_model.load_state_dict(PRETRAIN_STATE)
    final_model = fit_fixed_epochs(final_model, X_CUR, Y_CUR, final_epochs, 3e-4)
    package = {
        "state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
        "input_dim": int(X_CUR.shape[1]),
        "grid_points": N_GRID,
        "past_ft": PAST_FT,
        "future_ft": FUTURE_FT,
        "decay_tau": DECAY_TAU,
        "feature_version": "detrended_scale_invariant_v1",
        "cv_gate": gate,
    }
    torch.save(package, WORK / "external_pretrained_surface_mlp.pt")
    model_card = {
        "model": "external-pretrained residual SurfaceMLP",
        "external_curves": int(len(np.unique(G_EXT))),
        "external_windows": int(len(X_EXT)),
        "current_wells": int(len(WELLS)),
        "current_windows": int(len(X_CUR)),
        "final_finetune_epochs": final_epochs,
        "cv_gate": gate,
        "submission_authorized": False,
        "allowed_next_step": "stack-OOF integration CV only",
    }
    (WORK / "model_card.json").write_text(json.dumps(model_card, indent=2))
    log.info("Transfer model exported. Leaderboard submission remains disabled.")
else:
    log.warning("Transfer gate failed. No model package exported.")

assert not (WORK / "submission.csv").exists(), "CV notebook must never create submission.csv"
print("R2 complete. Gate passed:", gate["passed"])
print("Next step:", gate["next_step"])
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

notebook_path = OUT_DIR / "rogii-external-pretrain-gpu-r2.ipynb"
notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

metadata = {
    "id": "boltuzamaki/rogii-external-geometry-transfer-gpu-r2",
    "title": "ROGII External Geometry Transfer GPU R2",
    "code_file": notebook_path.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
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
