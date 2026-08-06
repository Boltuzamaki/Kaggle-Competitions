"""Complete-well residual-curve basis prediction, grouped OOF pilot.

A sequence encoder sees the complete legal suffix evidence and predicts DCT
coefficients of the residual to a strong OOF path. Each example is one well,
not one station. No marker columns or test-specific information are used.
"""
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.fft import dct, idct
from sklearn.model_selection import GroupKFold
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"exp/results/complete_well_residual_basis"
OUT.mkdir(parents=True, exist_ok=True)
L, K = 384, 20
FEATURES = [
    "d_md", "d_z", "d_xy", "dz_dmd", "gr", "gr_m21", "gr_s21", "frac",
    "slp_all", "slp_50", "ktvt_rng", "ktvt_std", "pfx_gr_rmse",
    "pf_d", "pf3_d", "beam_d", "pf_vs_beam", "ncc8_d", "ncc8_s",
    "ncc15_d", "ncc15_s", "ncc25_d", "ncc25_s", "sp_mean_d",
    "sp_std", "sp_dmin",
]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def resample(v):
    return np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(v)),
                     np.asarray(v, float))


class Encoder(nn.Module):
    def __init__(self, cin):
        super().__init__()
        widths = (64, 96, 128, 160)
        layers = []
        c = cin
        for w in widths:
            layers += [nn.Conv1d(c, w, 7, stride=2, padding=3),
                       nn.GroupNorm(8, w), nn.SiLU(),
                       nn.Conv1d(w, w, 5, padding=2),
                       nn.GroupNorm(8, w), nn.SiLU()]
            c = w
        self.net = nn.Sequential(*layers)
        # Mean, std, toe and heel embeddings retain global shape/direction.
        self.head = nn.Sequential(nn.Linear(c*4, 256), nn.SiLU(),
                                  nn.Dropout(.15), nn.Linear(256, K))
    def forward(self, x):
        h = self.net(x)
        s = torch.cat([h.mean(2), h.std(2), h[:, :, 0], h[:, :, -1]], 1)
        return self.head(s)


def make_sequences(limit):
    d = pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
    oo = joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
    wells = np.array(sorted(d.well.unique()))
    rng = np.random.RandomState(801)
    if limit and limit < len(wells):
        wells = rng.choice(wells, limit, replace=False)
        d = d[d.well.isin(wells)]
    seq = []
    for w, g in d.groupby("well", sort=True):
        ix = g.index.to_numpy()
        base = (.55*np.asarray(oo["lgb123"])[ix]+
                .20*np.asarray(oo["lgb7"])[ix]+
                .15*np.asarray(oo["xgb"])[ix]+
                .10*np.asarray(oo["cat"])[ix])
        # Base itself and its derivatives make this a learned whole-curve
        # correction rather than rebuilding the strong path from scratch.
        channels = [resample(g[c]) for c in FEATURES]
        channels += [resample(base), resample(np.gradient(base)),
                     resample(np.gradient(np.gradient(base)))]
        residual = resample(g.target.to_numpy()-base)
        coef = dct(residual, norm="ortho")[:K]
        seq.append({"well": w, "X": np.stack(channels).astype(np.float32),
                    "coef": coef.astype(np.float32), "g": g,
                    "base": base.astype(np.float32)})
    return seq


def main(limit=240, epochs=50):
    torch.manual_seed(801); np.random.seed(801)
    seq = make_sequences(limit)
    allx = np.stack([s["X"] for s in seq])
    # Input normalization uses no target and is fold-insensitive.
    mean = np.nanmean(allx, axis=(0, 2), keepdims=True)
    std = np.nanstd(allx, axis=(0, 2), keepdims=True)+1e-4
    allx = np.nan_to_num((allx-mean)/std).astype(np.float32)
    coef = np.stack([s["coef"] for s in seq]).astype(np.float32)/10
    groups = np.array([s["well"] for s in seq])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predc = np.zeros_like(coef)
    folds = []
    rng = np.random.RandomState(801)
    for fold, (tr, va) in enumerate(GroupKFold(5).split(allx, groups=groups)):
        m = Encoder(allx.shape[1]).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3,
                                weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, state = 1e9, None
        for ep in range(epochs):
            m.train()
            order = rng.permutation(tr)
            for st in range(0, len(tr), 16):
                ii = order[st:st+16]
                x = torch.tensor(allx[ii], device=device)
                y = torch.tensor(coef[ii], device=device)
                p = m(x)
                # Low orders control datum/slope and receive more weight.
                weight = torch.linspace(2, .5, K, device=device)
                loss = ((p-y).square()*weight).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 2); opt.step()
            sch.step()
            if ep % 5 == 4 or ep == epochs-1:
                m.eval()
                with torch.inference_mode():
                    pv = m(torch.tensor(allx[va], device=device)).cpu().numpy()
                val = float(np.sqrt(np.mean((pv-coef[va])**2)))
                if val < best:
                    best, state = val, {k: v.detach().cpu().clone()
                                       for k, v in m.state_dict().items()}
        m.load_state_dict(state); m.eval()
        with torch.inference_mode():
            predc[va] = m(torch.tensor(allx[va], device=device)).cpu().numpy()
        folds.append({"fold": fold, "wells": len(va),
                      "coef_rmse": float(np.sqrt(
                          np.mean((predc[va]-coef[va])**2)))})
        print(folds[-1], flush=True)
    yy, bb, rr = [], [], []
    for s, c in zip(seq, predc):
        full = np.zeros(L); full[:K] = c*10
        r = idct(full, norm="ortho")
        r0 = np.interp(np.linspace(0, 1, len(s["g"])),
                       np.linspace(0, 1, L), r)
        yy.append(s["g"].target.to_numpy(float)); bb.append(s["base"]); rr.append(r0)
    y, base, residual = map(np.concatenate, (yy, bb, rr))
    grid = []
    for blend in (0, .05, .1, .15, .2, .3, .4, .55, .7, 1):
        for clip in (1, 2, 4, 6, 10, 20):
            grid.append({"blend": blend, "clip": clip,
                         "rmse": rmse(y, base+blend*np.clip(
                             residual, -clip, clip))})
    grid = pd.DataFrame(grid).sort_values("rmse")
    result = {"wells": len(seq), "rows": len(y), "device": device,
              "baseline": rmse(y, base),
              "raw": rmse(y, base+residual),
              "best": grid.iloc[0].to_dict(), "folds": folds}
    print(json.dumps(result, indent=2), flush=True)
    grid.to_csv(OUT/"pilot_grid.csv", index=False)
    (OUT/"pilot_summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 240,
         int(sys.argv[2]) if len(sys.argv)>2 else 50)
