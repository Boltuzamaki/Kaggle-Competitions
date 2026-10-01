"""Local cross-validation using the *official* competition metric.

The scoring code lives in royerlab/kaggle-cell-tracking-competition; rather than
reimplement it (and risk optimising against a subtly different target) this
module imports ``tracking_cellmot.metrics`` directly and only supplies the glue
that turns our :class:`biohub_ct.TrackGraph` into a ``tracksdata`` graph.

Requires ``tracksdata`` + ``geff`` locally.  Not needed on Kaggle.

Setup
-----
    git clone https://github.com/royerlab/kaggle-cell-tracking-competition
    pip install tracksdata geff polars
    export TRACKING_CELLMOT_SRC=<repo>/src
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from biohub_ct import SCALE, TrackGraph, read_geff

def _ensure_metrics_importable() -> None:
    if "tracking_cellmot" in sys.modules:
        return
    candidates = [
        os.environ.get("TRACKING_CELLMOT_SRC"),
        "third_party/kaggle-cell-tracking-competition/src",
        "../third_party/kaggle-cell-tracking-competition/src",
    ]
    for c in candidates:
        if c and Path(c).is_dir():
            sys.path.insert(0, str(Path(c).resolve()))
            return
    raise ImportError(
        "tracking_cellmot not found. Clone royerlab/kaggle-cell-tracking-competition "
        "and set TRACKING_CELLMOT_SRC to its src/ directory."
    )

def to_tracksdata(g: TrackGraph):
    """Convert a :class:`TrackGraph` into a ``tracksdata`` in-memory graph.

    Node ids are reassigned by tracksdata, so our ids are remapped for the edge
    insertion - exactly what the organizers' ``csv_to_geffs.py`` does, which
    keeps local scoring faithful to the Kaggle path.
    """
    import polars as pl
    import tracksdata as td

    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    assigned = graph.bulk_add_nodes([
        {"t": int(tt), "z": float(zz), "y": float(yy), "x": float(xx)}
        for tt, zz, yy, xx in zip(g.t, g.z, g.y, g.x)
    ])
    id_map = {int(nid): a for nid, a in zip(g.ids, assigned)}

    if g.n_edges:
        graph.bulk_add_edges([
            {"source_id": id_map[int(s)], "target_id": id_map[int(d)]}
            for s, d in g.edges
            if int(s) in id_map and int(d) in id_map
        ])
    return graph

def score_dataset(pred: TrackGraph, gt_geff_path: Path | str,
                  max_distance: float = 7.0) -> dict:
    """Score one predicted graph against a ground-truth ``.geff``.

    Returns the per-sample metric row used by
    ``tracking_cellmot.metrics.summarise``.
    """
    _ensure_metrics_importable()
    import tracksdata as td
    from tracking_cellmot.metrics import (
        evaluate, nan_metrics_row, node_recall, per_sample_metrics,
    )

    gt_geff_path = Path(gt_geff_path)
    gt_raw = read_geff(gt_geff_path)
    n_total = float(gt_raw.meta.get("estimated_number_of_nodes", float("nan")))

    if pred.n_nodes == 0 or pred.n_edges == 0:
        row = nan_metrics_row()
        row.update(edge_tp=0, edge_fp=0, edge_fn=gt_raw.n_edges,
                   division_tp=0, division_fp=0, division_fn=0,
                   num_pred_nodes=pred.n_nodes, node_recall=0.0)
        return row

    loaded = td.graph.IndexedRXGraph.from_geff(gt_geff_path)
    gt_graph = loaded[0] if isinstance(loaded, tuple) else loaded

    pred_graph = to_tracksdata(pred)
    er = evaluate(pred_graph, gt_graph, scale=tuple(SCALE), max_distance=max_distance)
    recall = node_recall(pred_graph, gt_graph)
    return per_sample_metrics(er, n_total, recall)

def summarise_rows(rows: list[dict]) -> dict:
    """Run-level summary - identical to ``scripts/evaluate.py``."""
    _ensure_metrics_importable()
    from tracking_cellmot.metrics import summarise
    return summarise(rows)

def format_summary(s: dict) -> str:
    return (
        f"n={s['n']}  SCORE={s['score']:.4f}  "
        f"edge_J={s['edge_jaccard']:.4f}  adj_edge_J={s['adj_edge_jaccard']:.4f}  "
        f"div_J={s['division_jaccard']:.4f} "
        f"(TP={s['division_tp']} FP={s['division_fp']} FN={s['division_fn']})  "
        f"node_recall={s['node_recall']:.4f}"
    )

def oracle_upper_bound(gt_geff_path: Path | str) -> dict:
    """Score the ground truth against itself.

    Useful as a reality check: because the adjustment factor rewards a node
    count below ``estimated_number_of_nodes``, a *perfect* prediction that
    emits only the annotated lineage scores well above 1.0.  This quantifies
    the headroom that node-count discipline buys.
    """
    gt = read_geff(gt_geff_path)
    return score_dataset(gt, gt_geff_path)
