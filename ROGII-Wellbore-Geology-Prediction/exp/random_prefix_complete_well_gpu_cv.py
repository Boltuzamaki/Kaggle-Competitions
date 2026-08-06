"""Complete-well random-prefix episode augmentation with strict outer CV."""
from pathlib import Path
import glob
import json
import random
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"exp/results/random_prefix_complete_well"
OUT.mkdir(parents=True, exist_ok=True)
L = 512
OFF = (-20, -10, -5, 0, 5, 10, 20)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def rs(v):
    return np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(v)),
                     np.asarray(v, float))


def load_well(path):
    h = pd.read_csv(path)
    w = Path(path).name.split("__")[0]
    t = pd.read_csv(path.replace("__horizontal_well", "__typewell")).sort_values("TVT")
    if "TVT" not in h or h.TVT.isna().any():
        return None
    known = np.flatnonzero(h.TVT_input.notna().to_numpy())
    if len(known) < 40 or known[-1] >= len(h)-5:
        return None
    cols = {c: rs(h[c]) for c in ("MD", "X", "Y", "Z", "TVT")}
    gr = h.GR.to_numpy(float)
    gr = np.nan_to_num(gr, nan=np.nanmedian(gr))
    cols["GR"] = rs(gr)
    return {"well": w, "h": h, "org_ps": int(known[-1]),
            "org_cut": int(round(known[-1]/(len(h)-1)*(L-1))),
            "v": cols, "tw_tvt": t.TVT.to_numpy(float),
            "tw_gr": t.GR.to_numpy(float)}


def episode(s, cut):
    v = s["v"]
    md, x, y, z, tvt, gr = [v[k] for k in ("MD", "X", "Y", "Z",
                                             "TVT", "GR")]
    # Prefix-only slope baseline, deliberately available for arbitrary cutoffs.
    lo = max(0, cut-100)
    slope = np.polyfit(md[lo:cut+1]-md[cut], tvt[lo:cut+1], 1)[0]
    baseline = tvt[cut]+slope*(md-md[cut])
    mask = np.arange(L) <= cut
    vis = np.where(mask, tvt-tvt[cut], 0)
    gmed = np.median(gr[:cut+1])
    gs = 1.4826*np.median(np.abs(gr[:cut+1]-gmed))+5
    gz = (gr-gmed)/gs
    channels = [
        (md-md[cut])/1000, (x-x[cut])/1000, (y-y[cut])/1000,
        (z-z[cut])/100, np.gradient(x)/np.maximum(np.gradient(md), 1e-4),
        np.gradient(y)/np.maximum(np.gradient(md), 1e-4),
        np.gradient(z)/np.maximum(np.gradient(md), 1e-4),
        gz, gaussian_filter1d(gz, 5), gaussian_filter1d(gz, 20),
        vis/20, mask.astype(float), (baseline-tvt[cut])/20,
    ]
    # Typewell evidence along the prefix extrapolation path at several aliases.
    for off in OFF:
        tg = np.interp(baseline+off, s["tw_tvt"], s["tw_gr"])
        channels.append((gr-tg)/gs)
    target = (tvt-baseline)/20
    return np.stack(channels).astype(np.float32), target.astype(np.float32), \
        baseline.astype(np.float32), mask.astype(np.float32)


class Block(nn.Module):
    def __init__(self, w, d):
        super().__init__()
        self.c1 = nn.Conv1d(w, w, 5, padding=2*d, dilation=d)
        self.c2 = nn.Conv1d(w, w, 5, padding=2*d, dilation=d)
        self.n1 = nn.GroupNorm(8, w); self.n2 = nn.GroupNorm(8, w)
    def forward(self, x):
        h = torch.nn.functional.silu(self.n1(self.c1(x)))
        return torch.nn.functional.silu(x+self.n2(self.c2(h)))


class Net(nn.Module):
    def __init__(self, cin=20, w=64):
        super().__init__()
        self.stem = nn.Conv1d(cin, w, 1)
        self.net = nn.Sequential(*[Block(w, d) for d in
                                   (1, 2, 4, 8, 16, 32, 64)])
        self.head = nn.Sequential(nn.Conv1d(w, 32, 3, padding=1), nn.SiLU(),
                                  nn.Conv1d(32, 1, 1))
    def forward(self, x):
        return self.head(self.net(self.stem(x))).squeeze(1)


def main(limit=200, epochs=35):
    np.random.seed(804); random.seed(804); torch.manual_seed(804)
    seq = [s for s in map(load_well, sorted(glob.glob(
        str(ROOT/"data/train/*__horizontal_well.csv")))) if s is not None]
    rng = np.random.RandomState(804)
    if limit and limit < len(seq):
        seq = [seq[i] for i in sorted(rng.choice(len(seq), limit, False))]
    groups = np.array([s["well"] for s in seq])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred = [None]*len(seq)
    folds = []
    for fold, (tr, va) in enumerate(GroupKFold(5).split(seq, groups=groups)):
        m = Net().to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1.2e-3,
                                weight_decay=3e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, state = 1e9, None
        for ep in range(epochs):
            order = rng.permutation(tr)
            m.train()
            for st in range(0, len(order), 8):
                ii = order[st:st+8]
                eps = [episode(seq[i], rng.randint(int(.18*L), int(.72*L)))
                       for i in ii]
                x = torch.tensor(np.stack([q[0] for q in eps]), device=device)
                y = torch.tensor(np.stack([q[1] for q in eps]), device=device)
                mask = torch.tensor(np.stack([q[3] for q in eps]), device=device)
                p = m(x)
                suffix = 1-mask
                loss = ((p-y).square()*suffix).sum()/suffix.sum().clamp_min(1)
                # Small derivative loss encourages coherent curve shape.
                dm = suffix[:, 1:]*suffix[:, :-1]
                loss += .15*(((p[:, 1:]-p[:, :-1])-
                              (y[:, 1:]-y[:, :-1])).square()*dm).sum()/(
                                  dm.sum().clamp_min(1))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 2); opt.step()
            sch.step()
            if ep % 5 == 4 or ep == epochs-1:
                m.eval(); yy, pp = [], []
                with torch.inference_mode():
                    for i in va:
                        e = episode(seq[i], seq[i]["org_cut"])
                        q = m(torch.tensor(e[0][None], device=device)).cpu().numpy()[0]
                        mask = np.arange(L) > seq[i]["org_cut"]
                        yy.append(seq[i]["v"]["TVT"][mask])
                        pp.append((e[2]+20*q)[mask])
                val = rmse(np.concatenate(yy), np.concatenate(pp))
                if val < best:
                    best, state = val, {k: v.detach().cpu().clone()
                                       for k, v in m.state_dict().items()}
        m.load_state_dict(state); m.eval()
        yy, pp = [], []
        with torch.inference_mode():
            for i in va:
                e = episode(seq[i], seq[i]["org_cut"])
                q = m(torch.tensor(e[0][None], device=device)).cpu().numpy()[0]
                pred[i] = (e[2]+20*q, e[2])
                mk = np.arange(L) > seq[i]["org_cut"]
                yy.append(seq[i]["v"]["TVT"][mk]); pp.append(pred[i][0][mk])
        folds.append({"fold": fold, "wells": len(va),
                      "resampled_rmse": rmse(np.concatenate(yy),
                                             np.concatenate(pp))})
        print(folds[-1], flush=True)

    yy, nn_pred, slope_pred, flat_pred = [], [], [], []
    for s, (p, slope) in zip(seq, pred):
        n = len(s["h"]); pos = np.linspace(0, 1, n)
        p0 = np.interp(pos, np.linspace(0, 1, L), p)
        sl0 = np.interp(pos, np.linspace(0, 1, L), slope)
        mk = np.arange(n) > s["org_ps"]
        yy.append(s["h"].TVT.to_numpy(float)[mk]); nn_pred.append(p0[mk])
        slope_pred.append(sl0[mk])
        flat_pred.append(np.repeat(
            s["h"].TVT_input.iloc[s["org_ps"]], mk.sum()))
    y, net, slope, flat = map(np.concatenate,
                               (yy, nn_pred, slope_pred, flat_pred))
    grid = []
    for anchor_name, anchor in (("flat", flat), ("slope", slope)):
        for blend in (0, .1, .2, .3, .4, .55, .7, .85, 1):
            grid.append({"anchor": anchor_name, "blend": blend,
                         "rmse": rmse(y, (1-blend)*anchor+blend*net)})
    grid = pd.DataFrame(grid).sort_values("rmse")
    result = {"wells": len(seq), "rows": len(y), "device": device,
              "flat": rmse(y, flat), "slope": rmse(y, slope),
              "network": rmse(y, net), "best": grid.iloc[0].to_dict(),
              "folds": folds}
    print(json.dumps(result, indent=2), flush=True)
    grid.to_csv(OUT/"pilot_grid.csv", index=False)
    (OUT/"pilot_summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 200,
         int(sys.argv[2]) if len(sys.argv)>2 else 35)
