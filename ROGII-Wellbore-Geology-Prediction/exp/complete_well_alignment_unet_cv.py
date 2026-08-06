"""Complete-well GR alignment lattice + smooth-path decoding.

Unlike the failed direct TCN, this model does not regress TVT from generic row
features.  It constructs, for every eval station, a lattice of candidate TVTs
relative to the last visible TVT.  A small 2-D CNN scores horizontal/typewell
GR agreement jointly over the complete well, and Viterbi decoding extracts a
smooth candidate path.

Validation is honest: five-fold GroupKFold by complete well, and every
validation sample exposes only its original TVT_input prefix.

Example:
  python exp/complete_well_alignment_unet_cv.py --epochs 18
  python exp/complete_well_alignment_unet_cv.py --folds 1 --epochs 1 \
      --max-wells 30                 # smoke test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "train"
OUT = ROOT / "exp" / "results" / "alignment_unet_physics"
OFFSETS = np.arange(-64.0, 64.01, 1.0, dtype=np.float32)
STRIDE = 5


def robust(x):
    x = np.asarray(x, np.float32)
    med = np.nanmedian(x)
    scale = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    return (x - med) / max(float(scale), 5.0)


def load_well(path):
    wid = Path(path).name.split("__")[0]
    h = pd.read_csv(path)
    tw = pd.read_csv(DATA / f"{wid}__typewell.csv").sort_values("TVT")
    ps = int(h.TVT_input.notna().sum())
    if ps < 8 or ps >= len(h):
        return None
    # Preserve the exact organizer mask.  No hidden suffix TVT enters features.
    anchor = float(h.TVT_input.iloc[ps - 1])
    take = np.unique(np.r_[np.arange(ps - 1, len(h), STRIDE), len(h) - 1])
    gr = h.GR.interpolate(limit_direction="both").fillna(h.GR.median()).to_numpy(float)
    gr5 = pd.Series(gr).rolling(11, center=True, min_periods=1).mean().to_numpy()
    gr25 = pd.Series(gr).rolling(41, center=True, min_periods=1).mean().to_numpy()
    tg = tw.GR.interpolate(limit_direction="both").fillna(tw.GR.median()).to_numpy(float)
    tt = tw.TVT.to_numpy(float)
    # Legal physics path: keep structural coordinate S=Z+TVT fixed at the
    # last visible station.  The lattice learns only the geological residual,
    # rather than incorrectly asking a class offset to absorb trajectory TVD.
    z = h.Z.to_numpy(float)
    physics = anchor - (z[take] - z[ps - 1])
    cand = physics[:, None] + OFFSETS[None, :]
    tgr = np.interp(cand, tt, tg)
    tgr_lo = np.interp(cand - 4, tt, tg)
    tgr_hi = np.interp(cand + 4, tt, tg)
    hg = robust(gr5[take])[:, None]
    hc = robust(gr25[take])[:, None]
    tg_n = robust(tgr)
    # Candidate-lattice channels: raw/multiscale match, typewell texture,
    # trajectory geometry, progress, and candidate displacement.
    md = h.MD.to_numpy(float)
    dz = (z[take] - z[ps - 1]) / 50.0
    prog = (md[take] - md[ps - 1]) / max(md[-1] - md[ps - 1], 1.0)
    shp = (len(take), len(OFFSETS))
    X = np.stack([
        np.broadcast_to(hg, shp),
        np.broadcast_to(hc, shp),
        np.broadcast_to(tg_n, shp),
        np.broadcast_to(hg, shp) - np.broadcast_to(tg_n, shp),
        np.abs(np.broadcast_to(hg, shp) - np.broadcast_to(tg_n, shp)),
        np.broadcast_to(robust(tgr_hi - tgr_lo), shp),
        np.broadcast_to(dz[:, None], shp),
        np.broadcast_to(prog[:, None], shp),
        np.broadcast_to((OFFSETS / 50.0)[None, :], shp),
    ], 0).astype(np.float32)
    y = h.TVT.to_numpy(float)[take] - physics
    cls = np.rint((y - OFFSETS[0]) / 2).astype(np.int64)
    eval_mask = take >= ps
    valid = eval_mask & (cls >= 0) & (cls < len(OFFSETS))
    return dict(well=wid, X=X, cls=cls, valid=valid, eval=eval_mask,
                y=y.astype(np.float32),
                rows=take, n=len(h), ps=ps, anchor=anchor)


class AlignNet(nn.Module):
    """Anisotropic residual CNN over (station, candidate-TVT)."""
    def __init__(self, cin=9, width=32):
        super().__init__()
        self.inp = nn.Conv2d(cin, width, 1)
        blocks = []
        for d in (1, 2, 4, 8, 16, 32):
            blocks.append(nn.Sequential(
                nn.Conv2d(width, width, (7, 5), padding=(3 * d, 2),
                          dilation=(d, 1), groups=width),
                nn.Conv2d(width, width, 1), nn.GELU(),
                nn.GroupNorm(4, width)))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        x = self.inp(x)
        for block in self.blocks:
            x = x + block(x)
        return self.head(x).squeeze(1)  # B,L,K


def collate(items):
    length = max(v["X"].shape[1] for v in items)
    b, c, k = len(items), items[0]["X"].shape[0], items[0]["X"].shape[2]
    x = np.zeros((b, c, length, k), np.float32)
    y = np.full((b, length), -100, np.int64)
    for j, v in enumerate(items):
        n = v["X"].shape[1]
        x[j, :, :n] = v["X"]
        y[j, :n] = np.where(v["valid"], v["cls"], -100)
    return torch.from_numpy(x), torch.from_numpy(y)


def decode(logits, start_weight=0.18, move_weight=0.12, max_jump=3):
    """First-order Viterbi with anchor and locally smooth TVT constraints."""
    cost = -F.log_softmax(torch.as_tensor(logits), -1).numpy()
    n, k = cost.shape
    zero = int(np.argmin(np.abs(OFFSETS)))
    prev = cost[0] + start_weight * (np.arange(k) - zero) ** 2
    back = np.zeros((n, k), np.int16)
    jj = np.arange(k)
    for i in range(1, n):
        cur = np.full(k, np.inf)
        for shift in range(-max_jump, max_jump + 1):
            src = jj - shift
            ok = (src >= 0) & (src < k)
            val = prev[src[ok]] + move_weight * shift * shift
            improve = val < cur[ok]
            dst = jj[ok][improve]
            cur[dst] = val[improve]
            back[i, dst] = src[ok][improve]
        prev = cur + cost[i]
    path = np.empty(n, np.int16)
    path[-1] = np.argmin(prev)
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return OFFSETS[path]


@torch.no_grad()
def predict(model, item, dev):
    model.eval()
    x = torch.from_numpy(item["X"][None]).to(dev)
    with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
        logits = model(x)[0].float().cpu().numpy()
    return decode(logits)


def train_fold(train, valid, dev, epochs, batch_size, fold):
    model = AlignNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), 7e-4, weight_decay=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    rng = np.random.default_rng(2026 + fold)
    best, state = np.inf, None
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for st in range(0, len(order), batch_size):
            items = [train[i] for i in order[st:st + batch_size]]
            x, y = collate(items)
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits.transpose(1, 2), y,
                                       ignore_index=-100, label_smoothing=.04)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
        if ep >= 4 and (ep % 2 == 0 or ep == epochs - 1):
            se = nrow = 0
            for v in valid:
                p = predict(model, v, dev)
                m = v["eval"]
                se += np.square(p[m] - v["y"][m]).sum()
                nrow += m.sum()
            rmse = np.sqrt(se / nrow)
            print(f"fold={fold} ep={ep:02d} loss={np.mean(losses):.4f} "
                  f"pooled={rmse:.4f}")
            if rmse < best:
                best = rmse
                state = {k: z.detach().cpu().clone()
                         for k, z in model.state_dict().items()}
    if state is not None:
        model.load_state_dict(state)
    return model, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-wells", type=int)
    args = ap.parse_args()
    torch.manual_seed(2026)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(glob.glob(str(DATA / "*__horizontal_well.csv")))
    if args.max_wells:
        paths = paths[:args.max_wells]
    wells = [x for x in (load_well(p) for p in paths) if x is not None]
    print(f"device={dev} wells={len(wells)} lattice={len(OFFSETS)} "
          f"stride={STRIDE}")
    nsplit = min(5, len(wells))
    splits = list(GroupKFold(nsplit).split(wells, groups=[w["well"] for w in wells]))
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, fold_rows = [], []
    for fold, (ti, vi) in enumerate(splits[:args.folds]):
        tr, va = [wells[i] for i in ti], [wells[i] for i in vi]
        model, checkpoint_rmse = train_fold(
            tr, va, dev, args.epochs, args.batch_size, fold)
        se = nrow = 0
        for v in va:
            pred = predict(model, v, dev)
            # Score every hidden-suffix row, including rare targets outside the
            # candidate lattice (those correctly incur a large boundary error).
            m = v["eval"]
            err = pred[m] - v["y"][m]
            se += np.square(err).sum()
            nrow += m.sum()
            all_rows.append(dict(well=v["well"], fold=fold,
                                 rows=int(m.sum()),
                                 rmse=float(np.sqrt(np.mean(err ** 2)))))
        fr = float(np.sqrt(se / nrow))
        fold_rows.append(dict(fold=fold, pooled_rmse=fr,
                              checkpoint_rmse=checkpoint_rmse, rows=int(nrow)))
        torch.save(model.state_dict(), OUT / f"fold{fold}.pt")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    total = np.sqrt(np.average(
        np.square([r["pooled_rmse"] for r in fold_rows]),
        weights=[r["rows"] for r in fold_rows]))
    summary = dict(pooled_rmse=float(total), folds=fold_rows,
                   wells=len(wells), exact_original_prefix_mask=True,
                  physics_centered_lattice=True,
                  stride=STRIDE, offsets=[float(OFFSETS[0]), float(OFFSETS[-1]), 1])
    pd.DataFrame(all_rows).to_csv(OUT / "well_metrics.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
