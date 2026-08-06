"""Auxiliary EGFDU surface-shape prediction from legal whole-well inputs.

Formation markers are supervision only. At inference the model consumes
trajectory, GR, typewell summary, and visible TVT prefix. Predicted marker
shape is converted through the exact identity
    TVT = EGFDU - Z + median_prefix(TVT_input + Z - EGFDU_pred).
"""
from pathlib import Path
import glob
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"exp/results/marker_surface_sequence_gpu"
OUT.mkdir(parents=True, exist_ok=True)
L = 512


def interp(v, q):
    x = np.linspace(0, 1, len(v))
    return np.interp(q, x, np.asarray(v, float))


def load_one(path):
    d = pd.read_csv(path)
    if "EGFDU" not in d or "TVT" not in d:
        return None
    known = np.flatnonzero(d.TVT_input.notna().to_numpy())
    if not len(known) or known[-1] >= len(d)-5:
        return None
    ps = int(known[-1])
    q = np.linspace(0, 1, L)
    md, x, y, z = [d[c].to_numpy(float) for c in ("MD", "X", "Y", "Z")]
    gr = d.GR.to_numpy(float)
    fill = np.nanmedian(gr[:ps+1]) if np.isfinite(gr[:ps+1]).any() else 0.
    gr = np.nan_to_num(gr, nan=fill)
    # Coordinate/trajectory inputs, relative to prediction start.
    chans = [
        (md-md[ps])/1000, (x-x[ps])/1000, (y-y[ps])/1000,
        (z-z[ps])/100, np.gradient(x)/np.maximum(np.gradient(md), 1e-4),
        np.gradient(y)/np.maximum(np.gradient(md), 1e-4),
        np.gradient(z)/np.maximum(np.gradient(md), 1e-4),
    ]
    gz = (gr-np.median(gr))/(1.4826*np.median(np.abs(gr-np.median(gr)))+5)
    chans += [gz, gaussian_filter1d(gz, 5), gaussian_filter1d(gz, 20)]
    # Legal visible structural coordinate. Unknown suffix is explicitly zero.
    vis = np.zeros(len(d), float)
    vis[:ps+1] = (d.TVT_input.to_numpy(float)[:ps+1]+z[:ps+1] -
                  (d.TVT_input.iloc[ps]+z[ps]))
    mask = np.arange(len(d)) <= ps
    chans += [vis/20, mask.astype(float)]
    X = np.stack([interp(v, q) for v in chans]).astype(np.float32)
    marker = d.EGFDU.to_numpy(float)
    target = interp(marker-marker[ps], q).astype(np.float32)/20
    return {"well": Path(path).name.split("__")[0], "X": X, "target": target,
            "frame": d, "ps": ps, "q": q}


class Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.c1 = nn.Conv1d(width, width, 5, padding=2*dilation,
                            dilation=dilation)
        self.c2 = nn.Conv1d(width, width, 5, padding=2*dilation,
                            dilation=dilation)
        self.n1 = nn.GroupNorm(8, width)
        self.n2 = nn.GroupNorm(8, width)
    def forward(self, x):
        h = torch.nn.functional.silu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return torch.nn.functional.silu(x+h)


class MarkerNet(nn.Module):
    """Bidirectional multi-scale TCN with a global transformer bottleneck."""
    def __init__(self, cin=12, width=64):
        super().__init__()
        self.stem = nn.Conv1d(cin, width, 1)
        self.blocks = nn.Sequential(*[Block(width, d) for d in
                                      (1, 2, 4, 8, 16, 32)])
        layer = nn.TransformerEncoderLayer(
            width, 4, dim_feedforward=128, dropout=.1, batch_first=True,
            norm_first=True)
        self.context = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.Conv1d(width, 32, 3, padding=1),
                                  nn.SiLU(), nn.Conv1d(32, 1, 1))
    def forward(self, x):
        h = self.blocks(self.stem(x))
        # Global context at 4x lower resolution, restored by interpolation.
        low = torch.nn.functional.avg_pool1d(h, 4).transpose(1, 2)
        low = self.context(low).transpose(1, 2)
        low = torch.nn.functional.interpolate(
            low, size=h.shape[-1], mode="linear", align_corners=False)
        return self.head(h+low).squeeze(1)


def score(seq, pred_rel):
    d, ps = seq["frame"], seq["ps"]
    marker_pred = np.interp(np.linspace(0, 1, len(d)), seq["q"],
                            pred_rel*20)
    z = d.Z.to_numpy(float)
    # Exact visible-prefix calibration; no marker value is used here.
    b = np.median(d.TVT_input.to_numpy(float)[:ps+1]+z[:ps+1] -
                  marker_pred[:ps+1])
    tvt = marker_pred-z+b
    m = np.arange(len(d)) > ps
    return d.TVT.to_numpy(float)[m], tvt[m]


def main(limit=150, epochs=45):
    np.random.seed(731); random.seed(731); torch.manual_seed(731)
    files = sorted(glob.glob(str(ROOT/"data/train/*__horizontal_well.csv")))
    rng = np.random.RandomState(731)
    if limit and limit < len(files):
        files = sorted(rng.choice(files, limit, replace=False))
    seqs = [s for s in map(load_one, files) if s is not None]
    groups = np.array([s["well"] for s in seqs])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds = [None]*len(seqs)
    fold_rows = []
    for fold, (tr, va) in enumerate(GroupKFold(5).split(seqs, groups=groups)):
        model = MarkerNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3,
                                weight_decay=2e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, best_state = 1e9, None
        for ep in range(epochs):
            model.train()
            order = rng.permutation(tr)
            losses = []
            for st in range(0, len(order), 8):
                ii = order[st:st+8]
                x = torch.tensor(np.stack([seqs[i]["X"] for i in ii]),
                                 device=device)
                y = torch.tensor(np.stack([seqs[i]["target"] for i in ii]),
                                 device=device)
                p = model(x)
                # Surface and slope losses; slope regularization protects long
                # suffix extrapolation and prevents noisy GR memorization.
                loss = ((p-y)**2).mean()+.35*(
                    (p[:, 1:]-p[:, :-1])-(y[:, 1:]-y[:, :-1])).square().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2)
                opt.step(); losses.append(float(loss.detach()))
            sch.step()
            if ep % 5 == 4 or ep == epochs-1:
                model.eval()
                with torch.inference_mode():
                    pv = model(torch.tensor(
                        np.stack([seqs[i]["X"] for i in va]),
                        device=device)).cpu().numpy()
                val = np.sqrt(np.mean(np.concatenate([
                    (score(seqs[i], pv[j])[0]-score(seqs[i], pv[j])[1])**2
                    for j, i in enumerate(va)])))
                if val < best:
                    best = val
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
        model.load_state_dict(best_state); model.eval()
        with torch.inference_mode():
            pv = model(torch.tensor(np.stack([seqs[i]["X"] for i in va]),
                                    device=device)).cpu().numpy()
        yy, pp = [], []
        for j, i in enumerate(va):
            preds[i] = pv[j]
            a, b = score(seqs[i], pv[j]); yy.append(a); pp.append(b)
        fr = {"fold": fold, "wells": len(va), "rmse": float(np.sqrt(
            np.mean((np.concatenate(yy)-np.concatenate(pp))**2)))}
        fold_rows.append(fr); print(fr, flush=True)
    yy, pp = [], []
    oracle = []
    flat = []
    for s, p in zip(seqs, preds):
        a, b = score(s, p); yy.append(a); pp.append(b)
        d, ps = s["frame"], s["ps"]
        z = d.Z.to_numpy(float); m = d.EGFDU.to_numpy(float)
        bb = np.median(d.TVT_input.to_numpy(float)[:ps+1]+z[:ps+1]-m[:ps+1])
        oracle.append(m[ps+1:]-z[ps+1:]+bb)
        flat.append(np.repeat(d.TVT_input.iloc[ps], len(d)-ps-1))
    y = np.concatenate(yy); p = np.concatenate(pp)
    result = {"wells": len(seqs), "rows": len(y), "device": device,
              "rmse": float(np.sqrt(np.mean((y-p)**2))),
              "flat": float(np.sqrt(np.mean((y-np.concatenate(flat))**2))),
              "marker_oracle": float(np.sqrt(
                  np.mean((y-np.concatenate(oracle))**2))),
              "folds": fold_rows}
    print(json.dumps(result, indent=2), flush=True)
    (OUT/("pilot_summary.json" if limit else "full_summary.json")).write_text(
        json.dumps(result, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 150,
         int(sys.argv[2]) if len(sys.argv)>2 else 45)
