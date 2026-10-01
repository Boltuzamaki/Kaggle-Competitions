"""Learned edge scoring for frame-to-frame linking.

Why this exists
---------------
`scripts/error_attribution.py` shows 13.1% of ground-truth edges are lost to
detection misses and 4.0% to linking mistakes. The obvious fix for the first --
propose more candidates - *lowers* the score (see
:func:`biohub_ct.detect_dog_union`), because pure-distance assignment cannot
cope with the extra candidates. A learned scorer attacks both: it fixes some of
the 4%, and it is what makes higher-recall detection usable at all.

Design
------
Every candidate pair (node at t, node at t+1) within a generous gate becomes a
training example, labelled from the ground truth. Features are cheap and
CPU-only - no GPU, so this trains in minutes on all 199 videos:

* geometry: displacement in µm, and its z / in-plane split (Z is 4x coarser,
  so vertical and lateral motion are not equivalent)
* motion: agreement with the source track's previous velocity
* competition: the pair's distance rank from each end, plus the margin to the
  runner-up - an unambiguous nearest neighbour is very different from a
  coin-flip between two equidistant cells
* mutuality: whether each is the other's nearest neighbour
* context: local candidate density, which is what makes dense regions hard
* appearance: normalised cross-correlation of small image patches

The model is a gradient-boosted tree on ~12 features. Deliberately not a deep
model: with ~50 annotated nodes per video the labelled set is small, and trees
are far more sample-efficient here than a transformer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub_ct import SCALE, TrackGraph

FEATURE_NAMES = [
    "dist_um",
    "dz_um_abs",
    "dxy_um",
    "motion_residual_um",
    "has_velocity",
    "rank_fwd",
    "rank_bwd",
    "margin_fwd_um",
    "margin_bwd_um",
    "mutual_nn",
    "density_src",
    "density_tgt",
    "patch_ncc",
]

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _patch(vol: np.ndarray, c: np.ndarray, half=(1, 4, 4)) -> np.ndarray:
    z, y, x = (int(round(v)) for v in c)
    hz, hy, hx = half
    Z, Y, X = vol.shape
    z0, z1 = max(0, z - hz), min(Z, z + hz + 1)
    y0, y1 = max(0, y - hy), min(Y, y + hy + 1)
    x0, x1 = max(0, x - hx), min(X, x + hx + 1)
    p = vol[z0:z1, y0:y1, x0:x1].astype(np.float32)
    full = np.zeros((2 * hz + 1, 2 * hy + 1, 2 * hx + 1), dtype=np.float32)
    full[:p.shape[0], :p.shape[1], :p.shape[2]] = p
    return full

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.clip(a @ b / (na * nb), -1.0, 1.0))

def pair_features(src_vox: np.ndarray, tgt_vox: np.ndarray,
                  velocity: dict[int, np.ndarray] | None = None,
                  src_ids: np.ndarray | None = None,
                  gate_um: float = 12.0,
                  src_vol: np.ndarray | None = None,
                  tgt_vol: np.ndarray | None = None,
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build features for every (source, target) pair inside ``gate_um``.

    Returns ``(pairs, features, dist)`` where ``pairs`` is (P, 2) of indices
    into ``src_vox`` / ``tgt_vox``.
    """
    if len(src_vox) == 0 or len(tgt_vox) == 0:
        return np.zeros((0, 2), int), np.zeros((0, len(FEATURE_NAMES))), np.zeros(0)

    sp = np.asarray(src_vox, float) * SCALE
    tp = np.asarray(tgt_vox, float) * SCALE
    diff = tp[None, :, :] - sp[:, None, :]          # (S, T, 3)
    dist = np.sqrt((diff ** 2).sum(2))

    si, ti = np.where(dist <= gate_um)
    if len(si) == 0:
        return np.zeros((0, 2), int), np.zeros((0, len(FEATURE_NAMES))), np.zeros(0)

    # Ranks and runner-up margins, computed once per row/column.
    order_f = np.argsort(dist, axis=1)
    rank_f = np.empty_like(order_f)
    np.put_along_axis(rank_f, order_f, np.arange(dist.shape[1])[None, :], axis=1)
    sorted_f = np.take_along_axis(dist, order_f, axis=1)
    second_f = sorted_f[:, 1] if dist.shape[1] > 1 else sorted_f[:, 0] + gate_um

    order_b = np.argsort(dist, axis=0)
    rank_b = np.empty_like(order_b)
    np.put_along_axis(rank_b, order_b, np.arange(dist.shape[0])[:, None], axis=0)
    sorted_b = np.take_along_axis(dist, order_b, axis=0)
    second_b = sorted_b[1, :] if dist.shape[0] > 1 else sorted_b[0, :] + gate_um

    nn_f = dist.argmin(axis=1)
    nn_b = dist.argmin(axis=0)

    density_src = (dist <= gate_um).sum(axis=1)
    density_tgt = (dist <= gate_um).sum(axis=0)

    patches_s = patches_t = None
    if src_vol is not None and tgt_vol is not None:
        patches_s = [_patch(src_vol, c) for c in src_vox]
        patches_t = [_patch(tgt_vol, c) for c in tgt_vox]

    feats = np.zeros((len(si), len(FEATURE_NAMES)), dtype=np.float32)
    for k, (i, j) in enumerate(zip(si, ti)):
        d = diff[i, j]
        resid, has_v = 0.0, 0.0
        if velocity is not None and src_ids is not None:
            v = velocity.get(int(src_ids[i]))
            if v is not None:
                resid = float(np.linalg.norm(d - v))
                has_v = 1.0
        feats[k] = (
            dist[i, j],
            abs(d[0]),
            float(np.hypot(d[1], d[2])),
            resid,
            has_v,
            rank_f[i, j],
            rank_b[i, j],
            float(second_f[i] - dist[i, j]),
            float(second_b[j] - dist[i, j]),
            1.0 if (nn_f[i] == j and nn_b[j] == i) else 0.0,
            density_src[i],
            density_tgt[j],
            _ncc(patches_s[i], patches_t[j]) if patches_s is not None else 0.0,
        )
    return np.stack([si, ti], axis=1), feats, dist[si, ti]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class EdgeScorer:
    model: object
    feature_names: list[str]
    gate_um: float = 12.0
    use_appearance: bool = True

    def predict(self, feats: np.ndarray) -> np.ndarray:
        if len(feats) == 0:
            return np.zeros(0)
        return self.model.predict_proba(feats)[:, 1]

    def save(self, path: Path | str) -> None:
        import pickle
        path = Path(path)
        with path.open("wb") as fh:
            pickle.dump({"model": self.model, "feature_names": self.feature_names,
                         "gate_um": self.gate_um, "use_appearance": self.use_appearance}, fh)

    @staticmethod
    def load(path: Path | str) -> "EdgeScorer":
        import pickle
        with Path(path).open("rb") as fh:
            d = pickle.load(fh)
        return EdgeScorer(d["model"], d["feature_names"], d["gate_um"],
                          d.get("use_appearance", True))

def train_edge_scorer(X: np.ndarray, y: np.ndarray, *, seed: int = 0) -> EdgeScorer:
    """Fit a gradient-boosted classifier on candidate pairs."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=seed,
    )
    model.fit(X, y)
    return EdgeScorer(model=model, feature_names=list(FEATURE_NAMES))

# ---------------------------------------------------------------------------
# Linking with a learned cost
# ---------------------------------------------------------------------------

def link_learned(frames: list[np.ndarray], scorer: EdgeScorer,
                 volumes: list[np.ndarray] | None = None,
                 min_prob: float = 0.5,
                 max_link_um: float = 7.0,
                 motion_weight: float = 0.6) -> TrackGraph:
    """Frame-to-frame linking whose assignment cost is ``-log p(edge)``.

    Falls back to the gate on physical distance as a hard constraint, so the
    learned score can only *re-rank* candidates the geometry already allows --
    it can never invent a physically implausible link.
    """
    node_t: list[int] = []
    node_zyx: list[np.ndarray] = []
    frame_ids: list[list[int]] = []
    nid = 1
    for t, coords in enumerate(frames):
        ids_t = []
        for c in coords:
            node_t.append(t)
            node_zyx.append(np.asarray(c, float))
            ids_t.append(nid)
            nid += 1
        frame_ids.append(ids_t)

    edges: list[tuple[int, int]] = []
    velocity: dict[int, np.ndarray] = {}

    for t in range(len(frames) - 1):
        a, b = frames[t], frames[t + 1]
        if len(a) == 0 or len(b) == 0:
            continue
        vol_a = volumes[t] if volumes is not None else None
        vol_b = volumes[t + 1] if volumes is not None else None

        pairs, feats, dist = pair_features(
            a, b, velocity=velocity, src_ids=np.array(frame_ids[t]),
            gate_um=max(scorer.gate_um, max_link_um),
            src_vol=vol_a, tgt_vol=vol_b)
        if len(pairs) == 0:
            continue

        prob = scorer.predict(feats)
        cost = np.full((len(a), len(b)), 1e6)
        allowed = dist <= max_link_um
        cost[pairs[allowed, 0], pairs[allowed, 1]] = -np.log(
            np.clip(prob[allowed], 1e-6, 1 - 1e-6))

        ri, ci = linear_sum_assignment(cost)
        pm = {(int(i), int(j)): p for (i, j), p in zip(pairs, prob)}
        ap, bp = np.asarray(a, float) * SCALE, np.asarray(b, float) * SCALE
        for r, c in zip(ri, ci):
            if cost[r, c] >= 1e6:
                continue
            if pm.get((int(r), int(c)), 0.0) < min_prob:
                continue
            src, tgt = frame_ids[t][r], frame_ids[t + 1][c]
            edges.append((src, tgt))
            velocity[tgt] = bp[c] - ap[r]

    zyx = np.stack(node_zyx) if node_zyx else np.zeros((0, 3))
    return TrackGraph(
        t=np.array(node_t, dtype=np.int64),
        z=zyx[:, 0], y=zyx[:, 1], x=zyx[:, 2],
        ids=np.arange(1, len(node_t) + 1, dtype=np.int64),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
    )
