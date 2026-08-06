"""Train sequence models with 5-fold GroupKFold-by-well; produce OOF dTVT and score.
Random-crop training (fixed window), full-well windowed inference."""
import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import load_train, per_well_rmse, summarize, best_shrink, N_FOLDS
from seq_prep import CHANNELS
from models_seq import build

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = 20.0          # target dTVT scaled by 1/SCALE for training
WIN = 1024            # training crop length (strided; *2 ft)
BATCH = 16
EPOCHS = 22
torch.manual_seed(0)


def make_crops(seqs, win, per_well=2, rng=None):
    rng = rng or np.random.RandomState(0)
    crops = []
    for si, s in enumerate(seqs):
        L = len(s["y"])
        for _ in range(per_well):
            if L <= win:
                st = 0
            else:
                # bias starts so the crop tends to include post-PS region
                st = rng.randint(0, L - win + 1)
            crops.append((si, st))
    rng.shuffle(crops)
    return crops


def batch_crops(seqs, crops, win):
    for i in range(0, len(crops), BATCH):
        chunk = crops[i:i + BATCH]
        B = len(chunk)
        X = np.zeros((B, win, len(CHANNELS)), np.float32)
        Y = np.zeros((B, win), np.float32)
        M = np.zeros((B, win), bool)      # valid (non-pad)
        W = np.zeros((B, win), np.float32)  # loss weight (post-PS only)
        for j, (si, st) in enumerate(chunk):
            s = seqs[si]; L = len(s["y"]); e = min(st + win, L); n = e - st
            X[j, :n] = s["X"][st:e]; Y[j, :n] = s["y"][st:e]
            M[j, :n] = True; W[j, :n] = s["post"][st:e]
        yield (torch.from_numpy(X).to(DEV), torch.from_numpy(Y).to(DEV),
               torch.from_numpy(M).to(DEV), torch.from_numpy(W).to(DEV))


@torch.no_grad()
def infer_well(model, s, win=1536, overlap=256):
    model.eval()
    X = torch.from_numpy(s["X"][None]).to(DEV)     # (1,L,C)
    L = X.shape[1]
    if L <= win:
        m = torch.ones(1, L, dtype=torch.bool, device=DEV)
        return (model(X, m)[0].cpu().numpy()) * SCALE
    out = np.zeros(L); cnt = np.zeros(L)
    step = win - overlap
    for st in range(0, L, step):
        e = min(st + win, L)
        seg = X[:, st:e]
        m = torch.ones(1, e - st, dtype=torch.bool, device=DEV)
        p = model(seg, m)[0].cpu().numpy()
        out[st:e] += p; cnt[st:e] += 1
        if e == L:
            break
    return (out / np.maximum(cnt, 1)) * SCALE


def masked_loss(pred, Y, W):
    w = W + 0.15 * (1 - W)         # post weight 1.0, pre weight 0.15
    d = (pred - Y / SCALE)
    return (w * d * d).sum() / (w.sum() + 1e-6)


def train_fold(name, train_wells, val_wells, epochs=EPOCHS, log=print):
    model = build(name, len(CHANNELS)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    rng = np.random.RandomState(0)
    best_rmse, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        crops = make_crops(train_wells, WIN, per_well=2, rng=rng)
        tot = 0.0; nb = 0
        for X, Y, M, W in batch_crops(train_wells, crops, WIN):
            opt.zero_grad()
            pred = model(X, M)
            loss = masked_loss(pred, Y, W)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); tot += loss.item(); nb += 1
        sched.step()
        if ep >= 6 and (ep % 3 == 0 or ep == epochs - 1):
            pw = {s["well"]: infer_well(model, s) for s in val_wells}
            r = per_well_rmse(val_wells, pw).mean()
            if r < best_rmse:
                best_rmse = r; best_state = {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()}
            log(f"    ep{ep:02d} loss {tot/nb:.4f} val_rmse {r:.3f} best {best_rmse:.3f}")
    if best_state:
        model.load_state_dict(best_state)
    return model


def run_model(name, seqs, log=print):
    t0 = time.time()
    pred_by_well = {}
    for k in range(N_FOLDS):
        tr = [s for s in seqs if s["fold"] != k]
        va = [s for s in seqs if s["fold"] == k]
        log(f"  fold {k}: train {len(tr)} val {len(va)}")
        model = train_fold(name, tr, va, log=log)
        for s in va:
            pred_by_well[s["well"]] = infer_well(model, s)
        del model; torch.cuda.empty_cache()
    r, sh, cl = best_shrink(seqs, pred_by_well)
    pp = {w: np.clip(p * sh, -cl, cl) for w, p in pred_by_well.items()}
    summarize(seqs, pp, name, extra=dict(shrink=sh, clip=cl,
              seconds=round(time.time() - t0, 1), device=DEV))
    # persist OOF for potential ensembling
    np.savez(os.path.join(os.path.dirname(__file__), "results", f"oof_{name}.npz"),
             **{w: p for w, p in pred_by_well.items()})


if __name__ == "__main__":
    names = sys.argv[1:] or ["convgru", "tcn", "transformer", "tcntransformer"]
    seqs = load_train()
    print(f"device={DEV} wells={len(seqs)} channels={len(CHANNELS)}")
    for nm in names:
        print(f"=== {nm} ===")
        run_model(nm, seqs)
