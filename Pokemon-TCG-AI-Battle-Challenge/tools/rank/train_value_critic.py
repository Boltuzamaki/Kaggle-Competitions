"""Train a win-probability critic on replay board states, for use as a search leaf.

Reports AUC against the reference point from discussion 713608: a 10-feature
linear probe on public board state reached AUC 0.818 (0.69 on turns 1-4), while
that team's neural value head managed 0.61. If we clear ~0.80 the critic is worth
wiring into hybrid's leaf evaluation; if it lands near 0.6 the feature set is the
problem, not the model.

Trains a logistic-regression baseline and a small MLP, keeps whichever validates
better, and reports AUC by game phase because a leaf evaluator is most valuable
EARLY (late boards are already decided and the search sees terminal states anyway).
"""
from __future__ import annotations

import argparse
import os
import pickle
import random

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    ranks = {}
    i = 0
    r = 1
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (r + (r + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        r += (j - i + 1)
        i = j + 1
    pos = sum(1 for _, l in pairs if l == 1)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return 0.5
    s = sum(ranks[k] for k, (_, l) in enumerate(pairs) if l == 1)
    return (s - pos * (pos + 1) / 2.0) / (pos * neg)


class MLP(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "value_dataset.pkl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "agent", "value_critic.pt"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    a = ap.parse_args()

    d = pickle.load(open(a.data, "rb"))
    X, Y, feats = d["X"], d["Y"], d["feats"]
    print(f"{len(X)} boards, {len(feats)} features, positive rate {sum(Y)/len(Y):.3f}")

    idx = list(range(len(X)))
    random.Random(11).shuffle(idx)
    cut = int(0.95 * len(idx))
    tr, va = idx[:cut], idx[cut:]

    import statistics as st
    mu = [st.mean(X[i][k] for i in tr[:200000]) for k in range(len(feats))]
    sd = [max(st.pstdev(X[i][k] for i in tr[:200000]), 1e-3) for k in range(len(feats))]

    def norm(i):
        return [(X[i][k] - mu[k]) / sd[k] for k in range(len(feats))]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.tensor([norm(i) for i in tr], dtype=torch.float, device=dev)
    Ytr = torch.tensor([float(Y[i]) for i in tr], dtype=torch.float, device=dev)
    Xva = torch.tensor([norm(i) for i in va], dtype=torch.float, device=dev)
    Yva = [Y[i] for i in va]
    turn_va = [X[i][0] for i in va]

    results = {}
    for name, model in (("logistic", nn.Linear(len(feats), 1)),
                        ("mlp", MLP(len(feats)))):
        m = (model if name == "mlp" else nn.Sequential(model, nn.Flatten(0))).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-5)
        lossf = nn.BCEWithLogitsLoss()
        for ep in range(a.epochs):
            m.train()
            perm = torch.randperm(len(Xtr), device=dev)
            for i in range(0, len(Xtr), a.batch):
                b = perm[i:i + a.batch]
                opt.zero_grad()
                loss = lossf(m(Xtr[b]), Ytr[b])
                loss.backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            s = m(Xva).cpu().tolist()
        A = auc(s, Yva)
        early = [(sc, y) for sc, y, t in zip(s, Yva, turn_va) if t <= 8]
        Ae = auc([x[0] for x in early], [x[1] for x in early]) if early else 0.5
        results[name] = (A, Ae, m)
        print(f"  {name:9s} AUC {A:.4f}   early-game (turn<=8) AUC {Ae:.4f}")

    best = max(results.items(), key=lambda kv: kv[1][0])
    name, (A, Ae, m) = best
    torch.save({"kind": name, "model": m.state_dict(), "mu": mu, "sd": sd,
                "feats": feats, "auc": A, "auc_early": Ae}, a.out)
    print(f"\nbest: {name} AUC {A:.4f} (early {Ae:.4f}) -> {a.out}")
    print("reference: rival team's linear probe AUC 0.818, their NN value head 0.61")


if __name__ == "__main__":
    main()
