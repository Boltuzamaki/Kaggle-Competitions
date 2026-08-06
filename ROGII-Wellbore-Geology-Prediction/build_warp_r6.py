"""Build an anchored GR/typewell cross-attention WARP experiment."""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "kernels" / "neural_stack_integration_r3" / "rogii-current-neural-stack-integration-r3.ipynb"
OUT = ROOT / "kernels" / "warp_r6"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence, found {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


def splice(text: str, start: str, end: str, replacement: str) -> str:
    lo = text.find(start)
    hi = text.find(end, lo + len(start))
    if lo < 0 or hi < 0:
        raise RuntimeError(f"could not splice {start!r} -> {end!r}")
    return text[:lo] + replacement.rstrip() + "\n\n\n" + text[hi:]


WARP_BLOCK = r'''
# WARP representation: the high-frequency TVT term is the known -Z trajectory.
# The model predicts only the smooth structural surface while using future GR
# and the full local typewell profile as a weak cross-attention correction.
PAST_GRID = np.linspace(-(PAST_FT - 1), 0.0, N_GRID)
FUTURE_GRID = np.linspace(1.0, FUTURE_FT, N_GRID)
TYPE_GRID = np.linspace(-80.0, 80.0, N_GRID)
WARP_CHANNELS = 8


def clean_curve(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    order = np.argsort(x); x, y = x[order], y[order]
    unique = np.r_[True, np.diff(x) > 1e-8]
    return x[unique], y[unique]


def fit_gr_calibration(md, gr, tvt, tw_tvt, tw_gr, anchor):
    use = np.isfinite(md) & np.isfinite(gr) & np.isfinite(tvt) & (md <= anchor)
    use &= md >= anchor - max(PAST_FT * 2.0, 500.0)
    if use.sum() < 30:
        return 1.0, 0.0
    expected = np.interp(tvt[use], tw_tvt, tw_gr)
    observed = gr[use]
    matrix = np.c_[expected, np.ones(len(expected))]
    try:
        alpha, beta = np.linalg.lstsq(matrix, observed, rcond=None)[0]
    except Exception:
        return 1.0, 0.0
    alpha = float(np.clip(alpha, 0.25, 4.0))
    beta = float(np.clip(beta, -150.0, 150.0))
    return alpha, beta


def encode_warp(past_u, future_gr, future_z, type_gr, linear_future=None, future_u=None):
    past_u = np.asarray(past_u, float)
    coef = np.polyfit(PAST_GRID, past_u, 1)
    linear_past = np.polyval(coef, PAST_GRID)
    detrended = past_u - linear_past
    surface_scale = max(float(np.std(detrended)), 0.50)
    surface = detrended / surface_scale
    surface_d1 = np.gradient(surface)
    surface_d2 = np.gradient(surface_d1)

    type_gr = np.asarray(type_gr, float)
    center = float(np.nanmedian(type_gr))
    gr_scale = max(float(1.4826 * np.nanmedian(np.abs(type_gr - center))), 5.0)
    hgr = np.clip((np.asarray(future_gr, float) - center) / gr_scale, -6.0, 6.0)
    tgr = np.clip((type_gr - center) / gr_scale, -6.0, 6.0)
    hgr_d1 = np.gradient(hgr)
    tgr_d1 = np.gradient(tgr)
    z_rate = np.gradient(np.asarray(future_z, float), FUTURE_GRID)
    z_rate = np.clip(z_rate, -1.5, 1.5)
    type_position = TYPE_GRID / 80.0
    channels = np.stack([
        surface, surface_d1, surface_d2,
        hgr, hgr_d1, z_rate,
        tgr, tgr_d1,
    ]).astype(np.float32)
    extras = np.array([
        np.clip(coef[0] / 0.10, -5.0, 5.0),
        np.log1p(surface_scale),
        np.mean(np.abs(hgr_d1)),
        np.mean(np.abs(tgr_d1)),
        np.mean(z_rate),
    ], dtype=np.float32)
    features = np.concatenate([channels.reshape(-1), extras]).astype(np.float32)
    if linear_future is None:
        linear_future = np.polyval(coef, FUTURE_GRID)
    if future_u is None:
        return features, np.asarray(linear_future, float), surface_scale
    target = ((np.asarray(future_u, float) - linear_future) / surface_scale).astype(np.float32)
    return features, target


def make_warp_windows(data):
    md, z, tvt, gr = data["md"], data["z"], data["tvt"], data["gr"]
    tw_tvt, tw_gr = data["tw_tvt"], data["tw_gr"]
    lo = int(np.ceil(md[0] + PAST_FT - 1))
    hi = int(np.floor(md[-1] - FUTURE_FT))
    xs, ys = [], []
    if hi < lo:
        return xs, ys
    u = tvt + z
    for anchor in np.arange(lo, hi + 1, CURRENT_STRIDE):
        past_u = np.interp(anchor + PAST_GRID, md, u)
        future_md = anchor + FUTURE_GRID
        future_u = np.interp(future_md, md, u)
        future_z = np.interp(future_md, md, z)
        alpha, beta = fit_gr_calibration(md, gr, tvt, tw_tvt, tw_gr, anchor)
        future_gr = (np.interp(future_md, md, gr) - beta) / alpha
        anchor_tvt = float(np.interp(anchor, md, tvt))
        type_gr = np.interp(anchor_tvt + TYPE_GRID, tw_tvt, tw_gr)
        seq = np.r_[past_u, future_u, future_gr, future_z, type_gr]
        if not np.all(np.isfinite(seq)):
            continue
        if np.quantile(np.abs(np.diff(future_u)), 0.99) > MAX_LOCAL_SLOPE:
            continue
        xx, yy = encode_warp(past_u, future_gr, future_z, type_gr, future_u=future_u)
        xs.append(xx); ys.append(yy)
    return xs, ys


WELLS = {}
window_x, window_y, window_g = [], [], []
selected_set = set(selected_wells)
for number, path in enumerate(all_paths, 1):
    well = path.name.split("__")[0]
    if well not in selected_set:
        continue
    df = pd.read_csv(path, usecols=["MD", "Z", "TVT", "TVT_input", "GR"])
    tw = pd.read_csv(path.parent / f"{well}__typewell.csv", usecols=["TVT", "GR"]).sort_values("TVT")
    data = {
        "md": df["MD"].to_numpy(float), "z": df["Z"].to_numpy(float),
        "tvt": df["TVT"].to_numpy(float), "tvt_input": df["TVT_input"].to_numpy(float),
        "gr": df["GR"].interpolate(limit_direction="both").to_numpy(float),
        "tw_tvt": tw["TVT"].to_numpy(float),
        "tw_gr": tw["GR"].interpolate(limit_direction="both").to_numpy(float),
    }
    WELLS[well] = data
    xx, yy = make_warp_windows(data)
    window_x.extend(xx); window_y.extend(yy); window_g.extend([well] * len(xx))
    if number % 100 == 0:
        log.info("warp windows loaded %d/%d", number, len(all_paths))
X_GEOM = np.asarray(window_x, np.float32)
Y_GEOM = np.asarray(window_y, np.float32)
G_GEOM = np.asarray(window_g)
log.info("warp windows=%s wells=%d", X_GEOM.shape, len(np.unique(G_GEOM)))
assert len(X_GEOM) > 100


class ConvBlock(nn.Module):
    def __init__(self, width=96, dropout=0.08):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, width), nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
            nn.Dropout(dropout), nn.Conv1d(width, width, 5, padding=2), nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.net(x)


class WARPNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        assert input_dim == WARP_CHANNELS * N_GRID + 5
        self.surface = nn.Conv1d(3, 96, 7, padding=3)
        self.horizontal = nn.Linear(3, 96)
        self.typewell = nn.Linear(2, 96)
        self.cross = nn.MultiheadAttention(96, 4, dropout=0.08, batch_first=True)
        self.extra = nn.Sequential(nn.Linear(5, 96), nn.GELU(), nn.Linear(96, 96))
        self.blocks = nn.Sequential(*[ConvBlock() for _ in range(4)])
        self.head = nn.Sequential(nn.GroupNorm(8, 96), nn.GELU(), nn.Conv1d(96, 1, 1))
    def forward(self, x):
        channels = x[:, : WARP_CHANNELS * N_GRID].reshape(-1, WARP_CHANNELS, N_GRID)
        surface = self.surface(channels[:, 0:3])
        horizontal_tokens = channels[:, 3:6].transpose(1, 2)
        type_tokens = channels[:, 6:8].transpose(1, 2)
        h = self.horizontal(horizontal_tokens)
        t = self.typewell(type_tokens)
        cross, _ = self.cross(h, t, t, need_weights=False)
        extra = self.extra(x[:, WARP_CHANNELS * N_GRID:]).unsqueeze(-1)
        fused = surface + (h + 0.25 * cross).transpose(1, 2) + extra
        return self.head(self.blocks(fused)).squeeze(1)


def sequence_loss(pred, target):
    level = torch.nn.functional.smooth_l1_loss(pred, target)
    pd1 = pred[:, 1:] - pred[:, :-1]
    td1 = target[:, 1:] - target[:, :-1]
    slope = torch.nn.functional.smooth_l1_loss(pd1, td1)
    curve = torch.nn.functional.smooth_l1_loss(
        pd1[:, 1:] - pd1[:, :-1], td1[:, 1:] - td1[:, :-1]
    )
    return level + 0.20 * slope + 0.05 * curve


def data_loader(x, y, shuffle):
    ds = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=DEVICE.type == "cuda")


@torch.no_grad()
def neural_predict(model, x):
    model.eval(); outputs = []
    dummy = np.zeros((len(x), N_GRID), np.float32)
    for xb, _ in data_loader(np.asarray(x, np.float32), dummy, False):
        outputs.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.vstack(outputs)


def train_geometry(x_train, y_train, x_valid, y_valid, tag):
    torch.manual_seed(SEED)
    model = WARPNet(x_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MLP_EPOCHS)
    best_score, best_state, best_epoch, stale = np.inf, None, 0, 0
    for epoch in range(1, MLP_EPOCHS + 1):
        model.train(); losses = []
        for xb, yb in data_loader(x_train, y_train, True):
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch; stale = 0
        else:
            stale += 1
            if stale >= MLP_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, best_score


@torch.no_grad()
def predict_next(model, past_u, future_gr, future_z, type_gr):
    features, linear_future, scale = encode_warp(past_u, future_gr, future_z, type_gr)
    residual = neural_predict(model, features[None, :])[0]
    return linear_future + scale * residual


def geometry_delta_for_well(model, well):
    data = WELLS[well]
    visible = np.isfinite(data["tvt_input"])
    last = np.flatnonzero(visible)[-1]
    anchor_md = float(data["md"][last])
    u_input = data["tvt_input"] + data["z"]
    past_u = np.interp(anchor_md + PAST_GRID, data["md"][: last + 1], u_input[: last + 1])
    alpha, beta = fit_gr_calibration(
        data["md"], data["gr"], data["tvt_input"], data["tw_tvt"], data["tw_gr"], anchor_md
    )
    all_md, all_u = [], []
    while anchor_md < float(data["md"][-1]):
        future_md = anchor_md + FUTURE_GRID
        future_z = np.interp(future_md, data["md"], data["z"])
        future_gr = (np.interp(future_md, data["md"], data["gr"]) - beta) / alpha
        anchor_tvt = float(past_u[-1] - np.interp(anchor_md, data["md"], data["z"]))
        type_gr = np.interp(anchor_tvt + TYPE_GRID, data["tw_tvt"], data["tw_gr"])
        future_u = predict_next(model, past_u, future_gr, future_z, type_gr)
        all_md.extend(future_md); all_u.extend(future_u)
        past_u = future_u; anchor_md += FUTURE_FT
    hidden = np.arange(last + 1, len(data["md"]))
    pred_u = np.interp(data["md"][hidden], np.asarray(all_md), np.asarray(all_u))
    pred_tvt = pred_u - data["z"][hidden]
    return np.clip(pred_tvt - data["tvt_input"][last], -80.0, 80.0)
'''


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
control = "".join(notebook["cells"][1]["source"])
control = replace_once(control, "CURRENT_STRIDE = 200", "CURRENT_STRIDE = 250")
control = replace_once(control, "MLP_EPOCHS = 25", "MLP_EPOCHS = 22")
control = replace_once(control, "MLP_PATIENCE = 5", "MLP_PATIENCE = 5")
control = replace_once(control, "BATCH_SIZE = 1024", "BATCH_SIZE = 256")
notebook["cells"][1]["source"] = control.splitlines(keepends=True)

driver = "".join(notebook["cells"][5]["source"])
driver = splice(driver, "# Geometry-window representation used by", "LGB_PARAMS = dict(", WARP_BLOCK)
driver = replace_once(
    driver,
    '    "blend_decay_w020": stack_oof + 0.20 * decay * (geometry_oof - stack_oof),',
    '    "blend_decay_w020": stack_oof + 0.20 * decay * (geometry_oof - stack_oof),\n'
    '    "blend_decay_w030": stack_oof + 0.30 * decay * (geometry_oof - stack_oof),',
)
driver = driver.replace("rogii-neural-stack-r3", "rogii-warp-crossattn-r6")
notebook["cells"][5]["source"] = driver.splitlines(keepends=True)
notebook["cells"][0]["source"] = (
    "# ROGII R6: WARP Cross-Attention\n\n"
    "Grouped-CV implementation of weak-GR/strong-continuity WARP. Future horizontal GR queries a local typewell profile, while the network forecasts only the smooth structural surface U=TVT+Z from the last visible anchor.\n"
).splitlines(keepends=True)

for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"r6-warp-{index}"
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]), filename=f"r6-warp:cell-{index}")

metadata = {
    "id": "boltuzamaki/rogii-r6-warp-cross-attention",
    "title": "ROGII R6 WARP Cross Attention",
    "code_file": "rogii-r6-warp-cross-attention.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "dataset_sources": [],
    "competition_sources": ["rogii-wellbore-geology-prediction"],
    "kernel_sources": [],
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / metadata["code_file"]).write_text(json.dumps(notebook, indent=1), encoding="utf-8")
(OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print("wrote", OUT)
