"""Dependency-free reimplementation of the competition metric.

The organizers' scorer needs `tracksdata` + `geff` + `polars`, none of which are
installed on Kaggle. That confined honest evaluation to the two placeholder
videos downloadable locally - an n=2 sample where a single edge moves the
Jaccard by ~0.02.

This module reproduces `tracking_cellmot.metrics` with numpy + scipy only, so
the pipeline can be scored against *all 199 training videos* inside a Kaggle
notebook, where the data is already mounted.

**It is verified against the official implementation** by
`tests/test_metric_parity.py`, which requires exact agreement on TP/FP/FN and
agreement to 1e-9 on the adjusted Jaccard. Treat any divergence as a bug here,
not there.

Faithfully reproduced quirks, each of which changes the number materially:

* Edges are kept only when ``t_target == t_source + 1``.
* A predicted edge counts as a false positive **only if** one endpoint matches a
  GT node that itself has a GT edge (``pred_valid = out_valid | in_valid``).
  Edges among unannotated cells are invisible to the metric.
* Several predicted edges collapsing onto the same GT edge are counted once.
* Out-degree is capped at 2 (lowest edge ids kept).
* ``adj = max(0, J * (1 - 0.1 * (N_pred - N_true) / N_true))`` - note this
  exceeds 1 when fewer nodes than expected are predicted.
* Run level: adjusted Jaccard is averaged **weighted by ``TP+FP+FN``**, while
  the plain Jaccard is micro-averaged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub_ct import SCALE, TrackGraph

ADJUSTMENT_ALPHA = 0.1
SCORE_DIVISION_WEIGHT = 0.1
MAX_DISTANCE_UM = 7.0

@dataclass
class EvalResult:
    edge_tp: int
    edge_fp: int
    edge_fn: int
    num_pred_nodes: int
    node_recall: float
    total_node_ratio: float
    edge_jaccard: float
    adj_edge_jaccard: float

def match_nodes(pred: TrackGraph, gt: TrackGraph,
                max_distance: float = MAX_DISTANCE_UM,
                scale: np.ndarray = SCALE) -> dict[int, int]:
    """Optimal per-timepoint assignment; returns predicted id -> GT id."""
    pred_by_t: dict[int, list[int]] = {}
    for i, t in enumerate(pred.t):
        pred_by_t.setdefault(int(t), []).append(i)
    gt_by_t: dict[int, list[int]] = {}
    for i, t in enumerate(gt.t):
        gt_by_t.setdefault(int(t), []).append(i)

    p = pred.coords() * scale
    g = gt.coords() * scale
    out: dict[int, int] = {}
    for t, gi in gt_by_t.items():
        pi = pred_by_t.get(t)
        if not pi:
            continue
        d = np.sqrt(((g[gi][:, None, :] - p[pi][None, :, :]) ** 2).sum(2))
        ri, ci = linear_sum_assignment(np.where(d <= max_distance, d, 1e9))
        for r, c in zip(ri, ci):
            if d[r, c] <= max_distance:
                out[int(pred.ids[pi[c]])] = int(gt.ids[gi[r]])
    return out

def evaluate(pred: TrackGraph, gt: TrackGraph, n_total: float | None = None,
             max_distance: float = MAX_DISTANCE_UM) -> EvalResult:
    n_pred = pred.n_nodes
    ratio = ((n_pred - n_total) / n_total) if n_total else float("nan")

    if pred.n_nodes == 0 or pred.n_edges == 0:
        j = 0.0
        return EvalResult(0, 0, gt.n_edges, n_pred, 0.0, ratio, j,
                          max(0.0, j * (1 - ADJUSTMENT_ALPHA * ratio)) if n_total else float("nan"))

    matched = match_nodes(pred, gt, max_distance)
    node_recall = len(set(matched.values())) / gt.n_nodes if gt.n_nodes else 0.0

    t_of = {int(i): int(t) for i, t in zip(pred.t * 0 + pred.ids, pred.t)}
    gt_out, gt_in = {}, {}
    for s, d in gt.edges:
        gt_out[int(s)] = gt_out.get(int(s), 0) + 1
        gt_in[int(d)] = gt_in.get(int(d), 0) + 1
    gt_edge_set = {(int(s), int(d)) for s, d in gt.edges}

    # Keep consecutive-frame edges only, capped at out-degree 2 by edge order.
    kept: list[tuple[int, int, int]] = []          # (edge_id, source, target)
    out_count: dict[int, int] = {}
    for eid, (s, d) in enumerate(pred.edges):
        s, d = int(s), int(d)
        if t_of.get(d, -10**9) - t_of.get(s, 10**9) != 1:
            continue
        if out_count.get(s, 0) >= 2:
            continue
        out_count[s] = out_count.get(s, 0) + 1
        kept.append((eid, s, d))

    # Collapse merges: several predicted edges mapping onto one GT edge count once.
    seen_gt_pair: set[tuple[int, int]] = set()
    tp = 0
    valid = 0
    for _eid, s, d in kept:
        ms, md = matched.get(s), matched.get(d)
        is_valid = (ms is not None and gt_out.get(ms, 0) > 0) or \
                   (md is not None and gt_in.get(md, 0) > 0)
        if ms is not None and md is not None:
            if (ms, md) in seen_gt_pair:
                continue                       # duplicate mapping onto the same GT edge
            if (ms, md) in gt_edge_set:
                seen_gt_pair.add((ms, md))
                tp += 1
                valid += 1
                continue
        if is_valid:
            valid += 1

    fp = valid - tp
    fn = gt.n_edges - tp
    denom = tp + fp + fn
    j = tp / denom if denom else float("nan")
    adj = max(0.0, j * (1 - ADJUSTMENT_ALPHA * ratio)) if (n_total and j == j) else float("nan")

    return EvalResult(tp, fp, fn, n_pred, node_recall, ratio, j, adj)

def summarise(rows: list[EvalResult]) -> dict:
    """Run-level aggregation, matching `tracking_cellmot.metrics.summarise`."""
    valid = [r for r in rows if r.edge_tp == r.edge_tp]
    if not valid:
        return {"n": 0, "edge_jaccard": float("nan"),
                "adj_edge_jaccard": float("nan"), "score": float("nan"),
                "node_recall": float("nan")}

    tp = sum(r.edge_tp for r in valid)
    fp = sum(r.edge_fp for r in valid)
    fn = sum(r.edge_fn for r in valid)

    adj_rows = [r for r in valid if r.adj_edge_jaccard == r.adj_edge_jaccard]
    w = [r.edge_tp + r.edge_fp + r.edge_fn for r in adj_rows]
    total_w = sum(w)
    adj = (sum(wi * r.adj_edge_jaccard for wi, r in zip(w, adj_rows)) / total_w
           if total_w else float("nan"))

    denom = tp + fp + fn
    return {
        "n": len(valid),
        "edge_jaccard": tp / denom if denom else float("nan"),
        "adj_edge_jaccard": adj,
        "node_recall": sum(r.node_recall for r in valid) / len(valid),
        "score": adj,   # divisions are not predicted; see experiment 8
        "edge_tp": tp, "edge_fp": fp, "edge_fn": fn,
    }

def format_summary(s: dict) -> str:
    return (f"n={s['n']}  SCORE={s['score']:.4f}  edge_J={s['edge_jaccard']:.4f}  "
            f"adj_edge_J={s['adj_edge_jaccard']:.4f}  node_recall={s['node_recall']:.4f}  "
            f"TP/FP/FN={s.get('edge_tp')}/{s.get('edge_fp')}/{s.get('edge_fn')}")
