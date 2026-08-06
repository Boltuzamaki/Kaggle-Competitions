"""Complete-well structural derivative TCN with exact anchored integration."""
from pathlib import Path
import glob
import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"exp/results/derivative_shape"
L = 512


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def rs(v):
    return np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(v)),
                     np.asarray(v, float))


def load_one(path):
    d = pd.read_csv(path); w = Path(path).name.split("__")[0]
    known = np.flatnonzero(d.TVT_input.notna().to_numpy())
    if len(known) < 50 or known[-1] >= len(d)-5:
        return None
    ps = int(known[-1]); cut = int(round(ps/(len(d)-1)*(L-1)))
    tw = pd.read_csv(path.replace("__horizontal_well", "__typewell")).sort_values("TVT")
    md, x, y, z, tvt = [rs(d[c]) for c in ("MD", "X", "Y", "Z", "TVT")]
    gr0 = d.GR.to_numpy(float); gr0 = np.nan_to_num(gr0, nan=np.nanmedian(gr0))
    gr = rs(gr0)
    S = tvt+z
    lo = max(0, cut-100)
    slope = float(np.polyfit(md[lo:cut+1]-md[cut], S[lo:cut+1]-S[cut], 1)[0])
    slope = float(np.clip(slope, -.05, .05))
    mask = np.arange(L) <= cut
    vis = np.where(mask, S-S[cut], 0)
    gmed = np.median(gr[:cut+1]); gs = 1.4826*np.median(
        np.abs(gr[:cut+1]-gmed))+5
    gz = (gr-gmed)/gs
    baseline = S[cut]+slope*(md-md[cut])
    ch = [(md-md[cut])/1000, (x-x[cut])/1000, (y-y[cut])/1000,
          (z-z[cut])/100,
          np.gradient(x)/np.maximum(np.gradient(md), 1e-4),
          np.gradient(y)/np.maximum(np.gradient(md), 1e-4),
          np.gradient(z)/np.maximum(np.gradient(md), 1e-4),
          gz, gaussian_filter1d(gz, 5), gaussian_filter1d(gz, 20),
          vis/20, mask.astype(float), np.full(L, slope/.03)]
    for off in (-15, -5, 0, 5, 15):
        tg = np.interp(baseline+off, tw.TVT, tw.GR)
        ch.append((gr-tg)/gs)
    return {"well": w, "d": d, "ps": ps, "cut": cut,
            "X": np.stack(ch).astype(np.float32), "md": md.astype(np.float32),
            "S": S.astype(np.float32), "slope": slope}


class Block(nn.Module):
    def __init__(self, width, dil):
        super().__init__()
        self.c1 = nn.Conv1d(width, width, 5, padding=2*dil, dilation=dil)
        self.c2 = nn.Conv1d(width, width, 5, padding=2*dil, dilation=dil)
        self.n1 = nn.GroupNorm(8, width); self.n2 = nn.GroupNorm(8, width)
    def forward(self, x):
        h = torch.nn.functional.silu(self.n1(self.c1(x)))
        return torch.nn.functional.silu(x+self.n2(self.c2(h)))


class DerivativeNet(nn.Module):
    def __init__(self, cin=18, width=64):
        super().__init__()
        self.stem = nn.Conv1d(cin, width, 1)
        self.net = nn.Sequential(*[Block(width, d) for d in
                                   (1, 2, 4, 8, 16, 32, 64)])
        self.local = nn.Conv1d(width, 1, 1)
        self.mean = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(width, 32), nn.SiLU(),
                                  nn.Linear(32, 1))
    def forward(self, x, suffix):
        h = self.net(self.stem(x))
        local = .03*torch.tanh(self.local(h).squeeze(1))
        # Separate mean dip from zero-mean local stratigraphic wiggle.
        den = suffix.sum(1, keepdim=True).clamp_min(1)
        local = local-(local*suffix).sum(1, keepdim=True)/den
        mean_corr = .03*torch.tanh(self.mean(h))
        base = .03*x[:, 12, :]  # stored normalized prefix slope channel
        return base+mean_corr+local


def integrate(ds, md, cut, anchor):
    step = md[:, 1:]-md[:, :-1]
    inc = .5*(ds[:, 1:]+ds[:, :-1])*step
    cum = torch.cat([torch.zeros_like(anchor[:, None]), torch.cumsum(inc, 1)], 1)
    batch_idx = torch.arange(cum.shape[0], device=cum.device)
    return anchor[:, None] + cum - cum[batch_idx, cut][:, None]


def main(limit=200, epochs=35):
    np.random.seed(806); torch.manual_seed(806)
    seq = [s for s in map(load_one, sorted(glob.glob(
        str(ROOT/"data/train/*__horizontal_well.csv")))) if s is not None]
    rng = np.random.RandomState(806)
    if limit and limit < len(seq):
        seq = [seq[i] for i in sorted(rng.choice(len(seq), limit, False))]
    groups = np.array([s["well"] for s in seq])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred = [None]*len(seq); folds = []
    for fold, (tr, va) in enumerate(GroupKFold(5).split(seq, groups=groups)):
        m = DerivativeNet().to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1.2e-3, weight_decay=3e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, state = 1e9, None
        for ep in range(epochs):
            order = rng.permutation(tr); m.train()
            for st in range(0, len(order), 8):
                ii = order[st:st+8]
                x = torch.tensor(np.stack([seq[i]["X"] for i in ii]), device=device)
                md = torch.tensor(np.stack([seq[i]["md"] for i in ii]), device=device)
                target = torch.tensor(np.stack([seq[i]["S"] for i in ii]), device=device)
                cuts = torch.tensor([seq[i]["cut"] for i in ii], device=device)
                suffix = torch.arange(L, device=device)[None] > cuts[:, None]
                ds = m(x, suffix.float())
                anchor = target[torch.arange(len(ii), device=device), cuts]
                sp = integrate(ds, md, cuts, anchor)
                loss = (((sp-target)/20).square()*suffix).sum()/suffix.sum()
                true_ds = torch.gradient(target, dim=1)[0]/torch.gradient(md, dim=1)[0].clamp_min(1e-4)
                loss += .08*((ds-true_ds).square()*suffix).sum()/suffix.sum()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 2); opt.step()
            sch.step()
            if ep % 5 == 4 or ep == epochs-1:
                m.eval(); yy, pp = [], []
                with torch.inference_mode():
                    for i in va:
                        s = seq[i]
                        x = torch.tensor(s["X"][None], device=device)
                        md = torch.tensor(s["md"][None], device=device)
                        sf = (torch.arange(L, device=device)[None] > s["cut"]).float()
                        ds = m(x, sf)
                        sp = integrate(ds, md, torch.tensor([s["cut"]], device=device),
                                       torch.tensor([s["S"][s["cut"]]], device=device))
                        mk = np.arange(L)>s["cut"]
                        yy.append(s["S"][mk]); pp.append(sp.cpu().numpy()[0, mk])
                val = rmse(np.concatenate(yy), np.concatenate(pp))
                if val < best:
                    best, state = val, {k: v.detach().cpu().clone()
                                       for k, v in m.state_dict().items()}
        m.load_state_dict(state); m.eval()
        with torch.inference_mode():
            for i in va:
                s=seq[i]; x=torch.tensor(s["X"][None],device=device)
                md=torch.tensor(s["md"][None],device=device)
                sf=(torch.arange(L,device=device)[None]>s["cut"]).float()
                ds=m(x,sf); sp=integrate(ds,md,torch.tensor([s["cut"]],device=device),
                    torch.tensor([s["S"][s["cut"]]],device=device))
                pred[i]=sp.cpu().numpy()[0]
        folds.append({"fold":fold,"wells":len(va),"resampled_S_rmse":best})
        print(folds[-1],flush=True)
    yy, pp, prefix = [], [], []
    for s,p in zip(seq,pred):
        n=len(s["d"]); pos=np.linspace(0,1,n)
        ps=np.interp(pos,np.linspace(0,1,L),p)
        mk=np.arange(n)>s["ps"]
        yy.append((s["d"].TVT+s["d"].Z).to_numpy(float)[mk])
        pp.append(ps[mk])
        md=s["d"].MD.to_numpy(float); S=(s["d"].TVT+s["d"].Z).to_numpy(float)
        prefix.append(S[s["ps"]]+s["slope"]*(md[mk]-md[s["ps"]]))
    y,p,b=map(np.concatenate,(yy,pp,prefix))
    grid=[]
    for blend in (0,.05,.1,.2,.3,.4,.55,.7,1):
        grid.append({"blend":blend,"rmse":rmse(y,(1-blend)*b+blend*p)})
    grid=pd.DataFrame(grid).sort_values("rmse")
    result={"wells":len(seq),"rows":len(y),"prefix_slope":rmse(y,b),
            "network":rmse(y,p),"best":grid.iloc[0].to_dict(),"folds":folds}
    print(json.dumps(result,indent=2),flush=True)
    grid.to_csv(OUT/"tcn_grid.csv",index=False)
    (OUT/"tcn_summary.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 200,
         int(sys.argv[2]) if len(sys.argv)>2 else 35)
