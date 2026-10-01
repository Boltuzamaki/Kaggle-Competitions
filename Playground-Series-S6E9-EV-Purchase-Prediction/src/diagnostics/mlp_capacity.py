"""Can the NN be made strong enough to earn blend weight? Fold 0."""
import os, sys
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)
os.chdir(_SRC)   # so the ../data and ../artifacts paths below resolve
import numpy as np, time, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
import common as C, features as F

DEV = "cuda"
X, Xt, y, tid = F.base_frame()
itr, iva = C.get_folds(y)[0]
A, (B,) = F.fold_transform(X.iloc[itr], y[itr], [X.iloc[iva]])

EMB = [c for c in A.columns if A[c].dtype.kind in "iu"
       and int(A[c].max()) < 32 and int(A[c].min()) >= 0]
NUM = [c for c in A.columns if c not in EMB]
cards = [int(max(A[c].max(), B[c].max())) + 1 for c in EMB]
print(f"num={len(NUM)} emb={len(EMB)}", flush=True)

qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                         subsample=300000, random_state=0).fit(A[NUM])
Ntr = torch.tensor(qt.transform(A[NUM]), dtype=torch.float32, device=DEV)
Nva = torch.tensor(qt.transform(B[NUM]), dtype=torch.float32, device=DEV)
Ctr = torch.tensor(A[EMB].values.astype(np.int64), device=DEV)
Cva = torch.tensor(B[EMB].values.astype(np.int64), device=DEV)
Ytr = torch.tensor(y[itr], dtype=torch.float32, device=DEV)
yva = y[iva]


class Net(nn.Module):
    def __init__(self, hidden, emb=16, drop=0.1, ln=False):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, min(emb, c)) for c in cards])
        prev = len(NUM) + sum(min(emb, c) for c in cards)
        L = []
        for h in hidden:
            L += [nn.Linear(prev, h),
                  nn.LayerNorm(h) if ln else nn.BatchNorm1d(h),
                  nn.SiLU(), nn.Dropout(drop)]
            prev = h
        L.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*L)

    def forward(self, n, c):
        return self.net(torch.cat([n] + [e(c[:, i]) for i, e in enumerate(self.embs)], 1)).squeeze(1)


def train(tag, hidden, epochs, bs, lr, drop, wd=1e-5, ln=False):
    t0 = time.time(); torch.manual_seed(0)
    m = Net(hidden, drop=drop, ln=ln).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    nb = (len(Ytr) + bs - 1) // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * nb)
    lossf = nn.BCEWithLogitsLoss()
    best = 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Ytr), device=DEV)
        for i in range(nb):
            idx = perm[i * bs:(i + 1) * bs]
            opt.zero_grad(set_to_none=True)
            lossf(m(Ntr[idx], Ctr[idx]), Ytr[idx]).backward()
            opt.step(); sch.step()
        if ep >= epochs - 8 or ep % 5 == 4:
            m.eval()
            with torch.no_grad():
                a = roc_auc_score(yva, torch.sigmoid(m(Nva, Cva)).cpu().numpy())
            best = max(best, a)
    print(f"{tag:46s} best={best:.6f}  {time.time()-t0:.0f}s", flush=True)


train("baseline 512-256-128 e24 bs8192 lr3e-3", (512, 256, 128), 24, 8192, 3e-3, 0.10)
train("wider 1024-512-256 e40 bs4096 lr2e-3", (1024, 512, 256), 40, 4096, 2e-3, 0.10)
train("deep 512x4 e40 bs4096 lr2e-3 drop.05", (512, 512, 512, 512), 40, 4096, 2e-3, 0.05)
train("wide+long 1024-512-256 e80 bs4096 lr1e-3", (1024, 512, 256), 80, 4096, 1e-3, 0.10)
train("no-drop 1024-512-256 e60 bs2048 lr1e-3", (1024, 512, 256), 60, 2048, 1e-3, 0.0)
