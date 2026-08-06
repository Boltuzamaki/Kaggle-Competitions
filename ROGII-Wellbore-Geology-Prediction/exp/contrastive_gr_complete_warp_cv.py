"""Outer-fold contrastive GR matcher aggregated over complete warp paths."""
from pathlib import Path
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
OUT = ROOT/"exp/results/contrastive_gr_complete_warp"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT/"exp"))
from whole_well_gr_warp_selector_cv import candidate_paths, rmse  # noqa

W = 64
NEG = np.array([-30, -20, -12, -6, -3, 3, 6, 12, 20, 30], float)


def channels(v):
    v = np.asarray(v, np.float32)
    med = np.median(v)
    scale = 1.4826*np.median(np.abs(v-med))+5
    z = (v-med)/scale
    return np.stack([z, gaussian_filter1d(z, 2),
                     gaussian_filter1d(z, 7)]).astype(np.float32)


def window_starts(n, max_windows=24):
    if n <= W:
        return np.array([0])
    starts = np.arange(0, n-W+1, max(W//2, 1))
    if len(starts) > max_windows:
        starts = starts[np.linspace(0, len(starts)-1, max_windows).astype(int)]
    return starts


def segment(v, st):
    x = np.asarray(v[st:min(st+W, len(v))], float)
    if len(x) < W:
        x = np.pad(x, (0, W-len(x)), mode="edge")
    return channels(x)


class PairNet(nn.Module):
    def __init__(self):
        super().__init__()
        def enc():
            return nn.Sequential(
                nn.Conv1d(3, 32, 7, padding=3), nn.SiLU(),
                nn.Conv1d(32, 48, 5, stride=2, padding=2), nn.SiLU(),
                nn.Conv1d(48, 64, 5, stride=2, padding=2), nn.SiLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Linear(64, 32))
        self.henc, self.tenc = enc(), enc()
        self.head = nn.Sequential(nn.Linear(96, 64), nn.SiLU(),
                                  nn.Dropout(.1), nn.Linear(64, 1))
    def forward(self, h, t):
        a, b = self.henc(h), self.tenc(t)
        a = nn.functional.normalize(a, dim=1)
        b = nn.functional.normalize(b, dim=1)
        return self.head(torch.cat([a, b, torch.abs(a-b)], 1)).squeeze(1)


def raw_well(w, g):
    hw = pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv")
    tw = pd.read_csv(ROOT/f"data/train/{w}__typewell.csv").sort_values("TVT")
    idx = g.id.astype(str).str.rsplit("_", n=1).str[1].astype(int).to_numpy()
    h = hw.GR.to_numpy(float)[idx]
    h = np.nan_to_num(h, nan=np.nanmedian(h))
    return h, tw.TVT.to_numpy(float), tw.GR.to_numpy(float)


def training_pairs(items, rng, max_per_well=20):
    hs, ts, labels = [], [], []
    for w, g, h, tt, tg in items:
        true = float(g.last_known_tvt.iloc[0])+g.target.to_numpy(float)
        starts = window_starts(len(g), max_per_well)
        if len(starts) > max_per_well:
            starts = rng.choice(starts, max_per_well, False)
        for st in starts:
            hv = segment(h, st)
            en = min(st+W, len(g))
            # Positive plus two randomly sampled hard aliases.
            offs = [rng.uniform(-.5, .5)] + list(rng.choice(NEG, 2, False))
            for oi, off in enumerate(offs):
                tv = np.interp(true[st:en]+off, tt, tg)
                if len(tv) < W:
                    tv = np.pad(tv, (0, W-len(tv)), mode="edge")
                hs.append(hv); ts.append(channels(tv))
                labels.append(1. if oi == 0 else 0.)
    return (np.stack(hs), np.stack(ts),
            np.asarray(labels, np.float32))


def candidate_scores(model, device, h, tt, tg, absolute_paths):
    starts = window_starts(len(h), 20)
    hemb, temb, owner = [], [], []
    for ci, path in enumerate(absolute_paths):
        for st in starts:
            en = min(st+W, len(h))
            tv = np.interp(path[st:en], tt, tg)
            if len(tv) < W:
                tv = np.pad(tv, (0, W-len(tv)), mode="edge")
            hemb.append(segment(h, st)); temb.append(channels(tv)); owner.append(ci)
    logits = []
    model.eval()
    with torch.inference_mode():
        for st in range(0, len(hemb), 1024):
            a = torch.tensor(np.stack(hemb[st:st+1024]), device=device)
            b = torch.tensor(np.stack(temb[st:st+1024]), device=device)
            logits.extend(model(a, b).cpu().numpy())
    z = pd.DataFrame({"candidate": owner, "logit": logits})
    return z.groupby("candidate").logit.mean().to_numpy()


def main(limit=150, epochs=10):
    torch.manual_seed(803); np.random.seed(803); random.seed(803)
    d = pd.read_pickle(ROOT/"r_v4b/train_feats.pkl")
    oof = joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]
    # Same quartile-stratified pilot as preceding exact-family tests.
    stats = []
    for w, g in d.groupby("well"):
        ix = g.index.to_numpy()
        b = (.55*np.asarray(oof["lgb123"])[ix]+.20*np.asarray(oof["lgb7"])[ix]+
             .15*np.asarray(oof["xgb"])[ix]+.10*np.asarray(oof["cat"])[ix])
        stats.append((w, rmse(g.target, b)))
    st = pd.DataFrame(stats, columns=["well", "base_rmse"])
    st["bin"] = pd.qcut(st.base_rmse, 4, labels=False)
    rng = np.random.RandomState(803)
    selected = []
    for _, q in st.groupby("bin"):
        selected.extend(rng.choice(q.well, min(len(q), limit//4), False))
    d = d[d.well.isin(selected)]
    items = []
    for w, g in d.groupby("well", sort=True):
        h, tt, tg = raw_well(w, g)
        items.append((w, g, h, tt, tg))
    groups = np.array([x[0] for x in items])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path_score = {}
    fold_rows = []
    for fold, (tr, va) in enumerate(GroupKFold(5).split(items, groups=groups)):
        H, T, Y = training_pairs([items[i] for i in tr], rng)
        model = PairNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=2e-4)
        pos_weight = torch.tensor([(Y == 0).sum()/max(1, (Y == 1).sum())],
                                  device=device)
        lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        for ep in range(epochs):
            order = rng.permutation(len(Y))
            model.train()
            for st0 in range(0, len(Y), 256):
                ii = order[st0:st0+256]
                a = torch.tensor(H[ii], device=device)
                b = torch.tensor(T[ii], device=device)
                y = torch.tensor(Y[ii], device=device)
                loss = lossfn(model(a, b), y)
                opt.zero_grad(); loss.backward(); opt.step()
        yy, bb, pp = [], [], []
        for i in va:
            w, g, h, tt, tg = items[i]
            base, paths, _ = candidate_paths(g, oof)
            absolute = float(g.last_known_tvt.iloc[0])+paths
            score = candidate_scores(model, device, h, tt, tg, absolute)
            path_score[w] = score
            j = int(np.argmax(score))
            yy.append(g.target.to_numpy(float)); bb.append(base); pp.append(paths[j])
        fold_rows.append({"fold": fold, "wells": len(va),
                          "base": rmse(np.concatenate(yy), np.concatenate(bb)),
                          "hard": rmse(np.concatenate(yy), np.concatenate(pp))})
        print(fold_rows[-1], flush=True)
    truth, bases, hard, soft, oracle = [], [], [], [], []
    for w, g, _, _, _ in items:
        base, paths, _ = candidate_paths(g, oof)
        score = path_score[w]
        temp = np.std(score)+1e-5
        wt = np.exp(np.clip((score-score.max())/(.5*temp), -30, 0)); wt /= wt.sum()
        y = g.target.to_numpy(float)
        truth.append(y); bases.append(base); hard.append(paths[np.argmax(score)])
        soft.append(wt@paths)
        oracle.append(paths[np.argmin(np.mean((paths-y[None])**2, 1))])
    y, base, hard, soft, oracle = map(np.concatenate,
                                       (truth, bases, hard, soft, oracle))
    grid = []
    for name, p in (("hard", hard), ("soft", soft)):
        for blend in (0, .1, .2, .3, .4, .55, .7, .85, 1):
            grid.append({"method": name, "blend": blend,
                         "rmse": rmse(y, (1-blend)*base+blend*p)})
    grid = pd.DataFrame(grid).sort_values("rmse")
    result = {"wells": len(items), "rows": len(y), "device": device,
              "baseline": rmse(y, base), "hard": rmse(y, hard),
              "soft": rmse(y, soft), "oracle": rmse(y, oracle),
              "best": grid.iloc[0].to_dict(), "folds": fold_rows}
    print(json.dumps(result, indent=2), flush=True)
    grid.to_csv(OUT/"pilot_grid.csv", index=False)
    (OUT/"pilot_summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 150,
         int(sys.argv[2]) if len(sys.argv)>2 else 10)
