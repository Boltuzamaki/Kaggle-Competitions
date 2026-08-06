"""Whole-well TabICL regression of regularized simplex-oracle expert weights."""
from pathlib import Path
import json

import numpy as np
from sklearn.model_selection import GroupKFold
from tabicl import TabICLRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/tabicl_oracle_weight_gate"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = ROOT / "exp/public_artifacts/tabicl/tabicl-regressor-v2-20260212.ckpt"

c = np.load(ROOT / "exp/results/continuous_well_moe/cpu_search_cache.npz", allow_pickle=True)
t = np.load(ROOT / "exp/results/continuous_well_moe/oof_weights.npz", allow_pickle=True)
X, wn, nrow, G = c["X"].astype(np.float32), c["well"], c["nrow"], c["G"]
target = t["target"].astype(np.float32)
if not np.array_equal(wn.astype(str), t["wells"].astype(str)):
    raise RuntimeError("well identity mismatch")

def score(W, ids):
    z = np.einsum("ni,nij,nj->n", W, G[ids], W)
    return float(np.sqrt(np.sum(nrow[ids] * z) / np.sum(nrow[ids])))

pred = np.zeros_like(target)
fid = np.full(len(wn), -1)
splits = list(GroupKFold(5).split(X, groups=wn))
for fold, (tr, va) in enumerate(splits):
    for j in range(target.shape[1]):
        m = TabICLRegressor(
            n_estimators=4, batch_size=4, kv_cache=False, model_path=MODEL,
            allow_auto_download=False, device="cuda", use_amp=True,
            random_state=7300 + fold * 20 + j, verbose=False,
        )
        m.fit(X[tr], target[tr, j])
        pred[va, j] = m.predict(X[va])
    fid[va] = fold
    print({"fold": fold, "done": True}, flush=True)

# Projection to the simplex is intentionally elementary and inference-safe.
W = np.maximum(pred, 0)
W /= np.maximum(W.sum(1, keepdims=True), 1e-8)
empty = W.sum(1) == 0
W[empty, 0] = 1
base = np.zeros_like(W); base[:, 0] = 1
grid = []
for blend in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
    Q = (1 - blend) * base + blend * W
    gains = []
    for fold, (_, va) in enumerate(splits):
        gains.append(score(base[va], va) - score(Q[va], va))
    grid.append({"blend": blend, "rmse": score(Q, np.arange(len(wn))),
                 "fold_gains": gains, "fold_wins": int(sum(x > 0 for x in gains))})
summary = {
    "protocol": "strict outer whole-well GroupKFold; TabICL predicts lambda=0.5 regularized per-well simplex oracle weights",
    "accepted_base": score(base, np.arange(len(wn))),
    "oracle_target": score(target, np.arange(len(wn))),
    "best": min(grid, key=lambda x: x["rmse"]),
    "grid": grid,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
np.savez_compressed(OUT / "oof.npz", predicted_weights=pred, fold=fid)
print(json.dumps(summary, indent=2))
