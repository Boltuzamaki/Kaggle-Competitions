"""Train a listwise option ranker with card-identity embeddings.

Architecture rationale, from the public post-mortem in discussion 713608: nine
prior methods (1-ply search, 2-ply expectimax, ISMCTS, MLP self-play, DAgger, ...)
all plateaued at the same ceiling, and the diagnosis was representational --
"none of these methods encoded which card an option was, only its type and
damage. The network literally could not see card synergies." The lever that broke
through was a per-card identity embedding. That is the centrepiece here.

The head is listwise: every legal option is scored independently, a softmax is
taken over the variable-length option list, and the loss is cross-entropy against
the option the (strong, winning) teacher actually chose. This matches the agent's
real interface -- rank the offered menu, return an index -- rather than inventing
an action space.
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_CARDS = 1400      # card ids run to ~1267; pad the table
N_TYPES = 20        # OptionType has 17 members
N_CTXTYPE = 64      # SelectContext has 49
CTX_DIM = 21
NUM_DIM = 8


class Ranker(nn.Module):
    def __init__(self, d_card=64, d_hidden=256):
        super().__init__()
        self.card = nn.Embedding(N_CARDS, d_card)
        self.otype = nn.Embedding(N_TYPES, 16)
        self.sctx = nn.Embedding(N_CTXTYPE, 16)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(CTX_DIM, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        )
        opt_in = d_card + 16 + NUM_DIM
        self.opt_mlp = nn.Sequential(
            nn.Linear(opt_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        )
        self.score = nn.Sequential(
            nn.Linear(d_hidden * 2 + 16, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, ctx, ctxid, cids, types, nums, mask):
        """ctx:(B,CTX) ctxid:(B,) cids/types:(B,O) nums:(B,O,NUM) mask:(B,O)"""
        B, O = cids.shape
        c = self.ctx_mlp(ctx)                       # (B,H)
        sc = self.sctx(ctxid)                       # (B,16)
        opt = torch.cat([self.card(cids), self.otype(types), nums], dim=-1)
        opt = self.opt_mlp(opt)                     # (B,O,H)
        joint = torch.cat([opt,
                           c.unsqueeze(1).expand(-1, O, -1),
                           sc.unsqueeze(1).expand(-1, O, -1)], dim=-1)
        s = self.score(joint).squeeze(-1)           # (B,O)
        return s.masked_fill(~mask, -1e9)


def collate(batch, max_opts):
    B = len(batch)
    O = min(max(len(r["cids"]) for r in batch), max_opts)
    ctx = torch.zeros(B, CTX_DIM)
    ctxid = torch.zeros(B, dtype=torch.long)
    cids = torch.zeros(B, O, dtype=torch.long)
    types = torch.zeros(B, O, dtype=torch.long)
    nums = torch.zeros(B, O, NUM_DIM)
    mask = torch.zeros(B, O, dtype=torch.bool)
    y = torch.zeros(B, dtype=torch.long)
    for i, r in enumerate(batch):
        n = min(len(r["cids"]), O)
        ctx[i] = torch.tensor(r["ctx"][:CTX_DIM], dtype=torch.float)
        ctxid[i] = min(r.get("ctxid", 0), N_CTXTYPE - 1)
        for j in range(n):
            cids[i, j] = min(max(r["cids"][j], 0), N_CARDS - 1)
            types[i, j] = min(max(r["types"][j], 0), N_TYPES - 1)
            nums[i, j] = torch.tensor(r["nums"][j][:NUM_DIM], dtype=torch.float)
        mask[i, :n] = True
        y[i] = r["y"] if r["y"] < n else 0
    return ctx, ctxid, cids, types, nums, mask, y


def normalise(rows):
    """Standardise the context + numeric blocks; return the stats for inference."""
    import statistics as st
    ctxs = [r["ctx"] for r in rows[:200000]]
    mu = [st.mean(c[i] for c in ctxs) for i in range(CTX_DIM)]
    sd = [max(st.pstdev(c[i] for c in ctxs), 1e-3) for i in range(CTX_DIM)]
    nums = [n for r in rows[:50000] for n in r["nums"]]
    nmu = [st.mean(n[i] for n in nums) for i in range(NUM_DIM)]
    nsd = [max(st.pstdev(n[i] for n in nums), 1e-3) for i in range(NUM_DIM)]
    for r in rows:
        r["ctx"] = [(v - mu[i]) / sd[i] for i, v in enumerate(r["ctx"][:CTX_DIM])]
        r["nums"] = [[(v - nmu[i]) / nsd[i] for i, v in enumerate(n[:NUM_DIM])]
                     for n in r["nums"]]
    return {"ctx_mu": mu, "ctx_sd": sd, "num_mu": nmu, "num_sd": nsd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "rank_dataset.pkl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "agent", "ranker.pt"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-opts", type=int, default=40)
    ap.add_argument("--d-card", type=int, default=64)
    ap.add_argument("--group-split", action="store_true",
                    help="hold out whole replay episodes instead of decisions")
    a = ap.parse_args()

    rows = pickle.load(open(a.data, "rb"))
    print(f"{len(rows)} decisions loaded")
    stats = normalise(rows)
    if a.group_split:
        groups = sorted(set(r.get("episode", str(i)) for i, r in enumerate(rows)))
        random.Random(7).shuffle(groups)
        val_groups = set(groups[:max(1, int(0.2 * len(groups)))])
        train = [r for r in rows if r.get("episode") not in val_groups]
        val = [r for r in rows if r.get("episode") in val_groups]
    else:
        random.Random(7).shuffle(rows)
        cut = int(0.95 * len(rows))
        train, val = rows[:cut], rows[cut:]
    print(f"train {len(train)}  val {len(val)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Ranker(d_card=a.d_card).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    def run_eval():
        model.eval()
        cor = tot = 0
        top3 = 0
        loss_sum = 0.0
        with torch.no_grad():
            for i in range(0, len(val), a.batch):
                b = val[i:i + a.batch]
                t = [x.to(dev) for x in collate(b, a.max_opts)]
                s = model(*t[:6])
                y = t[6]
                loss_sum += F.cross_entropy(s, y, reduction="sum").item()
                pred = s.argmax(-1)
                cor += (pred == y).sum().item()
                k = min(3, s.shape[1])
                top3 += (s.topk(k, dim=-1).indices == y.unsqueeze(1)).any(-1).sum().item()
                tot += len(b)
        return cor / tot, top3 / tot, loss_sum / tot

    # A ranker that always picks the engine's first option is the baseline to beat:
    # discussion 713608 reports the engine enumerates options best->worst.
    base = sum(1 for r in val if r["y"] == 0) / len(val)
    print(f"baseline (always option 0) accuracy on val: {base:.4f}")

    best = 0.0
    for ep in range(a.epochs):
        model.train()
        random.shuffle(train)
        tl = n = 0
        for i in range(0, len(train), a.batch):
            b = train[i:i + a.batch]
            t = [x.to(dev) for x in collate(b, a.max_opts)]
            s = model(*t[:6])
            loss = F.cross_entropy(s, t[6])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(b); n += len(b)
        sched.step()
        acc, t3, vl = run_eval()
        flag = ""
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "stats": stats,
                        "d_card": a.d_card, "acc": acc}, a.out)
            flag = "  *saved"
        print(f"epoch {ep+1}/{a.epochs} train_loss {tl/n:.4f}  "
              f"val_loss {vl:.4f}  top1 {acc:.4f}  top3 {t3:.4f}{flag}", flush=True)

    print(f"\nbest top1 {best:.4f} vs baseline {base:.4f} "
          f"(lift {best - base:+.4f})")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
