"""Build the R4 full-training neural/contact leaderboard probe."""
from __future__ import annotations

import json
from pathlib import Path

import build_stack_contact_experiment as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "kernels" / "neural_contact_probe_r4"
OUT.mkdir(parents=True, exist_ok=True)


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one replacement, found {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


INTRO = r"""# ROGII R4: Validated Neural Stack + Conservative Contact Probe

R3 established a grouped-OOF improvement from a current-only trajectory MLP:
the distance-decayed 10% correction improved pooled RMSE by 0.96%, won all
five folds and 58.7% of wells, improved 600/1200-ft accuracy, and reduced the
worst-decile error.

R4 trains that exact correction on all current-competition wells. The three
test wells pass the existing strict contact guard, so the leaderboard probe
retains 90% of the 8.940 contact trajectory and mixes only 10% of the validated
neural-stack fallback:

\[
T^{probe}=0.90T^{contact}+0.10T^{neural\ stack}.
\]

No external data, prediction CSV, or pretrained competition model is used.
The notebook writes the honest neural-stack fallback, the strict contact
trajectory, the conservative probe, and a full submission audit.
"""


CONTROL = base.CONTROL
CONTROL = replace_once(
    CONTROL,
    'SUBMISSION_PROFILE = "contact_gated_anchor"  # previous_stack | contact_gated_anchor',
    'SUBMISSION_PROFILE = "contact_neural_probe"  # previous_stack | contact_gated_anchor | contact_neural_probe',
)
CONTROL = replace_once(
    CONTROL,
    "STACK_CLIP = 60.0",
    "STACK_CLIP = 60.0\n"
    "NEURAL_BLEND_WEIGHT = 0.10\n"
    "NEURAL_BLEND_TAU = 2400.0\n"
    "NEURAL_FULL_EPOCHS = 20\n"
    "CONTACT_PROBE_FALLBACK_WEIGHT = 0.10",
)
CONTROL = replace_once(
    CONTROL,
    'if SUBMISSION_PROFILE not in {"previous_stack", "contact_gated_anchor"}:',
    'if SUBMISSION_PROFILE not in {"previous_stack", "contact_gated_anchor", "contact_neural_probe"}:',
)


GEOMETRY_CODE = r'''
# R3-promoted current-only trajectory MLP. Fixed 20-epoch full-data refit.
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_ng_seed = 20260721
np.random.seed(_ng_seed)
torch.manual_seed(_ng_seed)
_ng_cuda_visible = torch.cuda.is_available()
_ng_cuda_compatible = False
_ng_gpu_status = "CUDA unavailable"
if _ng_cuda_visible:
    try:
        _ng_cap = torch.cuda.get_device_capability(0)
        _ng_sm = 10 * _ng_cap[0] + _ng_cap[1]
        _ng_supported = [
            int(a.split("_")[1]) for a in torch.cuda.get_arch_list()
            if a.startswith("sm_") and a.split("_")[1].isdigit()
        ]
        _ng_min_sm = min(_ng_supported) if _ng_supported else 999
        _ng_cuda_compatible = _ng_sm >= _ng_min_sm
        _ng_gpu_status = (
            f"{torch.cuda.get_device_name(0)} sm_{_ng_sm}; "
            f"PyTorch minimum sm_{_ng_min_sm}; compatible={_ng_cuda_compatible}"
        )
    except Exception as _ng_exc:
        _ng_gpu_status = f"probe failed: {type(_ng_exc).__name__}: {_ng_exc}"
_ng_device = torch.device("cuda" if _ng_cuda_compatible else "cpu")
if _ng_device.type == "cuda":
    torch.cuda.manual_seed_all(_ng_seed)
else:
    torch.set_num_threads(min(8, os.cpu_count() or 4))
print("neural geometry accelerator:", _ng_gpu_status, "device:", _ng_device)

_ng_past_ft = 300
_ng_future_ft = 300
_ng_grid_n = 60
_ng_stride = 600 if LOCAL_DEBUG else 200
_ng_batch = 512 if LOCAL_DEBUG else 1024
_ng_epochs = 2 if LOCAL_DEBUG else NEURAL_FULL_EPOCHS
_ng_past_grid = np.linspace(-(_ng_past_ft - 1), 0.0, _ng_grid_n)
_ng_future_grid = np.linspace(1.0, _ng_future_ft, _ng_grid_n)


def _ng_clean_curve(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    order = np.argsort(x); x, y = x[order], y[order]
    unique = np.r_[True, np.diff(x) > 1e-8]
    return x[unique], y[unique]


def _ng_encode(past, future=None):
    past = np.asarray(past, float)
    coef = np.polyfit(_ng_past_grid, past, 1)
    linear_past = np.polyval(coef, _ng_past_grid)
    detrended = past - linear_past
    scale = max(float(np.std(detrended)), 0.50)
    level = detrended / scale
    d1 = np.gradient(level); d2 = np.gradient(d1)
    extras = np.array([
        np.clip(coef[0] / 0.10, -5.0, 5.0), np.log1p(scale),
        np.mean(np.abs(d1)), np.mean(np.abs(d2)),
    ])
    features = np.concatenate([level, d1, d2, extras]).astype(np.float32)
    linear_future = np.polyval(coef, _ng_future_grid)
    if future is None:
        return features, linear_future, scale
    target = ((np.asarray(future, float) - linear_future) / scale).astype(np.float32)
    return features, target


def _ng_windows(x, y):
    x, y = _ng_clean_curve(x, y)
    lo = int(np.ceil(x[0] + _ng_past_ft - 1))
    hi = int(np.floor(x[-1] - _ng_future_ft))
    xs, ys = [], []
    if hi < lo:
        return xs, ys
    for anchor in np.arange(lo, hi + 1, _ng_stride):
        past = np.interp(anchor + _ng_past_grid, x, y)
        future = np.interp(anchor + _ng_future_grid, x, y)
        seq = np.r_[past, future]
        if np.all(np.isfinite(seq)) and np.quantile(np.abs(np.diff(seq)), 0.99) <= 0.60:
            xx, yy = _ng_encode(past, future)
            xs.append(xx); ys.append(yy)
    return xs, ys


_ng_x, _ng_y = [], []
_ng_train_paths = sorted((DATA / "train").glob("*__horizontal_well.csv"))
if LOCAL_DEBUG:
    _ng_train_paths = _ng_train_paths[:DEBUG_TRAIN_WELLS]
for _ng_number, _ng_path in enumerate(_ng_train_paths, 1):
    _ng_df = pd.read_csv(_ng_path, usecols=["MD", "Z", "TVT"])
    _ng_md, _ng_u = _ng_clean_curve(
        _ng_df["MD"].to_numpy(float),
        _ng_df["TVT"].to_numpy(float) + _ng_df["Z"].to_numpy(float),
    )
    _xx, _yy = _ng_windows(_ng_md, _ng_u)
    _ng_x.extend(_xx); _ng_y.extend(_yy)
    if _ng_number % 100 == 0:
        print("neural geometry loaded", _ng_number, "/", len(_ng_train_paths), flush=True)
_ng_x = np.asarray(_ng_x, np.float32)
_ng_y = np.asarray(_ng_y, np.float32)
print("neural geometry windows:", _ng_x.shape)


class _NGResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 512), nn.GELU(), nn.Dropout(0.08),
            nn.Linear(512, 256), nn.Dropout(0.08),
        )
    def forward(self, x):
        return x + self.net(x)


class _NGSurfaceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 256), nn.GELU())
        self.blocks = nn.Sequential(*[_NGResidualBlock() for _ in range(4)])
        self.output = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, _ng_grid_n))
    def forward(self, x):
        return self.output(self.blocks(self.input(x)))


def _ng_loss(pred, target):
    level = torch.nn.functional.smooth_l1_loss(pred, target)
    pd1 = pred[:, 1:] - pred[:, :-1]
    td1 = target[:, 1:] - target[:, :-1]
    slope = torch.nn.functional.smooth_l1_loss(pd1, td1)
    curve = torch.nn.functional.smooth_l1_loss(
        pd1[:, 1:] - pd1[:, :-1], td1[:, 1:] - td1[:, :-1]
    )
    return level + 0.20 * slope + 0.05 * curve


_ng_ds = TensorDataset(torch.from_numpy(_ng_x), torch.from_numpy(_ng_y))
_ng_loader = DataLoader(_ng_ds, batch_size=_ng_batch, shuffle=True, num_workers=0)
_ng_model = _NGSurfaceMLP(_ng_x.shape[1]).to(_ng_device)
_ng_opt = torch.optim.AdamW(_ng_model.parameters(), lr=1e-3, weight_decay=1e-4)
_ng_sched = torch.optim.lr_scheduler.CosineAnnealingLR(_ng_opt, T_max=_ng_epochs)
for _ng_epoch in range(1, _ng_epochs + 1):
    _ng_model.train(); _ng_losses = []
    for _xb, _yb in _ng_loader:
        _xb = _xb.to(_ng_device); _yb = _yb.to(_ng_device)
        if not LOCAL_DEBUG:
            _xb = _xb + 0.003 * torch.randn_like(_xb)
        _ng_opt.zero_grad(set_to_none=True)
        _loss = _ng_loss(_ng_model(_xb), _yb)
        _loss.backward(); torch.nn.utils.clip_grad_norm_(_ng_model.parameters(), 2.0)
        _ng_opt.step(); _ng_losses.append(float(_loss.detach().cpu()))
    _ng_sched.step()
    print(f"neural full epoch={_ng_epoch:02d} loss={np.mean(_ng_losses):.6f}", flush=True)


@torch.no_grad()
def _ng_predict(features):
    _ng_model.eval()
    tensor = torch.from_numpy(np.asarray(features, np.float32)).to(_ng_device)
    return _ng_model(tensor).cpu().numpy()


def _ng_next(past):
    features, linear_future, scale = _ng_encode(past)
    residual = _ng_predict(features[None, :])[0]
    return linear_future + scale * residual


def _ng_recursive(past, deltas):
    deltas = np.asarray(deltas, float)
    state = np.asarray(past, float)
    all_delta, all_value, start = [], [], 0.0
    while start < float(np.max(deltas)):
        future = _ng_next(state)
        all_delta.extend(start + _ng_future_grid); all_value.extend(future)
        state = future; start += _ng_future_ft
    return np.interp(deltas, np.asarray(all_delta), np.asarray(all_value))


_ng_by_id = {}
_ng_report = []
for _ng_path in sorted((DATA / "test").glob("*__horizontal_well.csv")):
    _ng_well = _ng_path.name.split("__")[0]
    _ng_df = pd.read_csv(_ng_path, usecols=["MD", "Z", "TVT_input"])
    _md = _ng_df["MD"].to_numpy(float)
    _z = _ng_df["Z"].to_numpy(float)
    _ti = _ng_df["TVT_input"].to_numpy(float)
    _visible = np.isfinite(_ti); _last = np.flatnonzero(_visible)[-1]
    _anchor_md = _md[_last]
    _past = np.interp(
        _anchor_md + _ng_past_grid, _md[: _last + 1], (_ti + _z)[: _last + 1]
    )
    _hidden = np.arange(_last + 1, len(_md))
    _pred_u = _ng_recursive(_past, _md[_hidden] - _anchor_md)
    _pred_delta = np.clip(_pred_u - _z[_hidden] - _ti[_last], -80.0, 80.0)
    for _row, _value in zip(_hidden, _pred_delta):
        _ng_by_id[f"{_ng_well}_{int(_row)}"] = float(_value)
    _ng_report.append({
        "well": _ng_well, "rows": len(_hidden),
        "geometry_delta_min": float(np.min(_pred_delta)),
        "geometry_delta_max": float(np.max(_pred_delta)),
    })

_ng_geometry_test_d = np.asarray([_ng_by_id[str(i)] for i in test["id"]], float)
_ng_stack_test_d = learned_test_d.copy()
_ng_decay = np.exp(-np.maximum(test["d_md"].to_numpy(float), 0.0) / NEURAL_BLEND_TAU)
learned_test_d = _ng_stack_test_d + NEURAL_BLEND_WEIGHT * _ng_decay * (
    _ng_geometry_test_d - _ng_stack_test_d
)
for _row in _ng_report:
    _idx = test["well"].eq(_row["well"]).to_numpy()
    _row["stack_neural_disagreement_rmse"] = float(np.sqrt(np.mean(
        (_ng_geometry_test_d[_idx] - _ng_stack_test_d[_idx]) ** 2
    )))
    _row["applied_correction_rmse"] = float(np.sqrt(np.mean(
        (learned_test_d[_idx] - _ng_stack_test_d[_idx]) ** 2
    )))
pd.DataFrame(_ng_report).to_csv(WORK / "neural_inference_report.csv", index=False)
'''


DRIVER = base.DRIVER
DRIVER = replace_once(
    DRIVER,
    'WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")',
    'default_work = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")\n'
    'WORK = Path(os.environ.get("ROGII_OUTPUT_DIR", str(default_work)))\n'
    'WORK.mkdir(parents=True, exist_ok=True)',
)
DRIVER = replace_once(
    DRIVER,
    "results = Parallel(n_jobs=N_JOBS, verbose=5)(",
    "results = Parallel(\n"
    "        n_jobs=N_JOBS, verbose=5,\n"
    "        prefer=\"threads\" if LOCAL_DEBUG else \"processes\",\n"
    "    )(",
)
DRIVER = replace_once(
    DRIVER,
    "learned_test_d = np.clip(\n"
    "    model.predict(Xt) * STACK_SHRINK, -STACK_CLIP, STACK_CLIP\n"
    ")\n"
    "pd.DataFrame(cv_rows).to_csv(WORK / \"training_validation_report.csv\", index=False)",
    "learned_test_d = np.clip(\n"
    "    model.predict(Xt) * STACK_SHRINK, -STACK_CLIP, STACK_CLIP\n"
    ")\n\n"
    + GEOMETRY_CODE.strip()
    + "\n\npd.DataFrame(cv_rows).to_csv(WORK / \"training_validation_report.csv\", index=False)",
)

DRIVER = replace_once(
    DRIVER,
    'honest_sub.to_csv(WORK / "submission_honest.csv", index=False)\n'
    'contact_sub.to_csv(WORK / "submission_contact_gated.csv", index=False)\n'
    'final_sub = honest_sub if SUBMISSION_PROFILE == "previous_stack" else contact_sub\n'
    'final_sub.to_csv(WORK / "submission.csv", index=False)',
    'honest_sub.to_csv(WORK / "submission_honest.csv", index=False)\n'
    'contact_sub.to_csv(WORK / "submission_contact_gated.csv", index=False)\n'
    'probe_sub = contact_sub.copy()\n'
    'probe_sub["tvt"] = (\n'
    '    (1.0 - CONTACT_PROBE_FALLBACK_WEIGHT) * contact_sub["tvt"].to_numpy(float)\n'
    '    + CONTACT_PROBE_FALLBACK_WEIGHT * honest_sub["tvt"].to_numpy(float)\n'
    ')\n'
    'probe_sub.to_csv(WORK / "submission_contact_neural_probe.csv", index=False)\n'
    'if SUBMISSION_PROFILE == "previous_stack":\n'
    '    final_sub = honest_sub\n'
    'elif SUBMISSION_PROFILE == "contact_gated_anchor":\n'
    '    final_sub = contact_sub\n'
    'else:\n'
    '    final_sub = probe_sub\n'
    'final_sub.to_csv(WORK / "submission.csv", index=False)',
)
DRIVER = replace_once(
    DRIVER,
    '    "fallback": "previous_residual_stack",',
    '    "fallback": "r3_validated_current_neural_stack",\n'
    '    "neural_blend_weight": float(NEURAL_BLEND_WEIGHT),\n'
    '    "neural_blend_tau": float(NEURAL_BLEND_TAU),\n'
    '    "neural_full_epochs": int(NEURAL_FULL_EPOCHS if not LOCAL_DEBUG else 2),\n'
    '    "contact_probe_fallback_weight": float(CONTACT_PROBE_FALLBACK_WEIGHT),\n'
    '    "probe_vs_contact_rmse": float(np.sqrt(np.mean((\n'
    '        probe_sub["tvt"].to_numpy(float) - contact_sub["tvt"].to_numpy(float)\n'
    '    ) ** 2))),\n'
    '    "probe_vs_contact_max_abs": float(np.max(np.abs(\n'
    '        probe_sub["tvt"].to_numpy(float) - contact_sub["tvt"].to_numpy(float)\n'
    '    ))),',
)


NOTEBOOK = {
    "cells": [
        base.base.markdown(INTRO),
        base.base.code(CONTROL, hidden=False),
        base.base.code(base.base.PF),
        base.base.code(base.base.BEAM),
        base.base.code(base.base.SIGNALS),
        base.base.code(DRIVER),
    ],
    "metadata": base.NOTEBOOK["metadata"],
    "nbformat": 4,
    "nbformat_minor": 5,
}


METADATA = {
    "id": "boltuzamaki/rogii-r4-validated-neural-contact-probe",
    "title": "ROGII R4 Validated Neural Contact Probe",
    "code_file": "rogii-r4-neural-contact-probe.ipynb",
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
