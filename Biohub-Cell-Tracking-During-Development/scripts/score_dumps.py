#!/usr/bin/env python
"""Score public prediction dumps with the *patched* division metric.

``zhincez/biohub-diagnostic-dumps`` publishes node/edge CSVs produced by a fork
of the public stack, on training films whose ground truth we hold locally. That
makes it possible to answer, without a single GPU minute, the question that
decides this competition's endgame:

    how much of the 0.1 division term is the public stack actually collecting,
    and how much does its own validator *think* it is collecting?

Three scorers are run on the same graphs:

``official``    ``tracking_cellmot.division_metrics`` - the truth, needs tracksdata
``patched``     ``src/biohub_div_metric`` - our port, must equal ``official``
``prepatch``    the weakly-connected-component rule the public stack still ships
                in its own held-out validator

Usage
-----
    PYTHONPATH=src python scripts/score_dumps.py <dumps_dir> [--gt data/comp/train]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from biohub_ct import SCALE, TrackGraph, read_geff  # noqa: E402
import biohub_div_metric as D  # noqa: E402
import biohub_metric as M  # noqa: E402

def load_dump(nodes_csv: Path, edges_csv: Path) -> TrackGraph:
    nodes = np.genfromtxt(nodes_csv, delimiter=",", names=True)
    edges = np.genfromtxt(edges_csv, delimiter=",", names=True, dtype=np.int64)
    e = np.stack([np.atleast_1d(edges["source_id"]), np.atleast_1d(edges["target_id"])], axis=1)
    return TrackGraph(
        t=np.atleast_1d(nodes["t"]).astype(np.int64),
        z=np.atleast_1d(nodes["z"]).astype(np.float64),
        y=np.atleast_1d(nodes["y"]).astype(np.float64),
        x=np.atleast_1d(nodes["x"]).astype(np.float64),
        ids=np.atleast_1d(nodes["node_id"]).astype(np.int64),
        edges=e.astype(np.int64),
        meta={},
    )

def _weak_components(node_ids: list[int], edges: list[tuple[int, int]]) -> dict[int, int]:
    comp: dict[int, int] = {}
    adj: dict[int, list[int]] = {n: [] for n in node_ids}
    for s, t in edges:
        if s in adj and t in adj:
            adj[s].append(t)
            adj[t].append(s)
    for seed in node_ids:
        if seed in comp:
            continue
        comp[seed] = seed
        stack = [seed]
        while stack:
            cur = stack.pop()
            for nbr in adj.get(cur, ()):
                if nbr not in comp:
                    comp[nbr] = seed
                    stack.append(nbr)
    return comp

def prepatch_division_confusion(pred: TrackGraph, gt: TrackGraph,
                                max_distance: float = 7.0) -> tuple[int, int, int]:
    """The pre-patch rule, transcribed from the public stack's own validator.

    A GT division counts as recovered when the matched parent and both matched
    daughter lineages land anywhere in one weakly connected component of the
    prediction, and that component contains a fork *somewhere*. This is what
    commit ``aa65e90`` removed on 2026-07-17, and what the 622-team block is
    still selecting its safe-division thresholds against.
    """
    pred_to_gt = M.match_nodes(pred, gt, max_distance)
    gt_to_pred = {g: p for p, g in pred_to_gt.items()}

    gt_out: dict[int, set[int]] = {}
    gt_in: dict[int, int] = {}
    for s, t in gt.edges:
        gt_out.setdefault(int(s), set()).add(int(t))
        gt_in[int(t)] = int(s)
    pred_out: dict[int, set[int]] = {}
    for s, t in pred.edges:
        pred_out.setdefault(int(s), set()).add(int(t))

    node_ids = [int(i) for i in pred.ids]
    components = _weak_components(node_ids, [(int(s), int(t)) for s, t in pred.edges])
    fork_components = {components[n] for n, outs in pred_out.items()
                       if len(outs) >= 2 and n in components}
    gt_division_sources = [s for s, outs in gt_out.items() if len(outs) >= 2]

    def lineage(root: int) -> set[int]:
        seen, stack = {root}, [root]
        while stack:
            cur = stack.pop()
            for nxt in gt_out.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    tp = fn = 0
    for gsrc in gt_division_sources:
        children = sorted(gt_out[gsrc])[:2]
        anchors = [gsrc] + ([gt_in[gsrc]] if gsrc in gt_in else [])
        anchor_pred = [gt_to_pred[a] for a in anchors if a in gt_to_pred]
        hits, ok = [], True
        for child in children:
            comps = {components[p] for g in lineage(child)
                     if (p := gt_to_pred.get(g)) is not None and p in components}
            if not comps:
                ok = False
                break
            hits.append(comps)
        if not ok or not anchor_pred:
            fn += 1
            continue
        anchor_comps = {components[p] for p in anchor_pred if p in components}
        if any(c in hits[0] and c in hits[1] and c in fork_components for c in anchor_comps):
            tp += 1
        else:
            fn += 1

    # the public validator's FP rule: forks matched onto an annotated GT node
    fp = sum(1 for n, outs in pred_out.items()
             if len(outs) >= 2 and n in pred_to_gt and gt_out.get(pred_to_gt[n]))
    return tp, fn, fp

def official_divisions(pred: TrackGraph, gt_geff: Path) -> tuple[int, int, int] | None:
    try:
        from local_eval import _ensure_metrics_importable, to_tracksdata
        _ensure_metrics_importable()
        import tracksdata as td
        from tracking_cellmot.division_metrics import evaluate_divisions
    except Exception:
        return None
    loaded = td.graph.IndexedRXGraph.from_geff(gt_geff)
    gt_graph = loaded[0] if isinstance(loaded, tuple) else loaded
    c = evaluate_divisions(to_tracksdata(pred), gt_graph, scale=tuple(SCALE), max_distance=7.0)
    return c.tp, c.fn, c.fp

def jaccard(tp: int, fn: int, fp: int) -> float:
    d = tp + fp + fn
    return tp / d if d else 0.0

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dumps", type=Path)
    ap.add_argument("--gt", type=Path, default=REPO / "data" / "comp" / "train")
    ap.add_argument("--no-official", action="store_true",
                    help="skip the tracksdata reference (slow on large graphs)")
    args = ap.parse_args()

    for cfg_dir in sorted(p for p in args.dumps.iterdir() if p.is_dir()):
        rows = []
        for nodes_csv in sorted(cfg_dir.glob("*_nodes.csv")):
            stem = nodes_csv.name[: -len("_nodes.csv")]
            edges_csv = cfg_dir / f"{stem}_edges.csv"
            gt_geff = args.gt / f"{stem}.geff"
            if not edges_csv.is_file() or not gt_geff.exists():
                continue
            pred = load_dump(nodes_csv, edges_csv)
            try:
                gt = read_geff(gt_geff)
            except FileNotFoundError:
                # session 1's rate-limited per-file download left some .geff
                # directories partially fetched; skip rather than half-score
                print(f"  skip {stem}: incomplete local ground truth")
                continue

            patched = D.evaluate_divisions(pred, gt)
            prepatch = prepatch_division_confusion(pred, gt)
            official = None if args.no_official else official_divisions(pred, gt_geff)

            n_total = float(gt.meta.get("estimated_number_of_nodes", float("nan")))
            edge = M.evaluate(pred, gt, n_total=n_total)
            rows.append((stem, edge, patched, prepatch, official))

        if not rows:
            continue
        print(f"\n=== {cfg_dir.name} ===")
        print(f"{'film':<18} {'adj_edge':>8} {'gt_div':>6} "
              f"{'patched tp/fn/fp':>18} {'prepatch tp/fn/fp':>18} {'official tp/fn/fp':>18}")
        tot = {"p": [0, 0, 0], "q": [0, 0, 0], "o": [0, 0, 0], "w": 0.0, "adj": 0.0}
        for stem, edge, patched, prepatch, official in rows:
            gt_divs = patched.tp + patched.fn
            o = "-" if official is None else f"{official[0]}/{official[1]}/{official[2]}"
            print(f"{stem:<18} {edge.adj_edge_jaccard:>8.4f} {gt_divs:>6} "
                  f"{patched.tp}/{patched.fn}/{patched.fp:<14} "
                  f"{prepatch[0]}/{prepatch[1]}/{prepatch[2]:<14} {o:>18}")
            for k, v in (("p", (patched.tp, patched.fn, patched.fp)), ("q", prepatch)):
                for i in range(3):
                    tot[k][i] += v[i]
            if official:
                for i in range(3):
                    tot["o"][i] += official[i]
            w = edge.edge_tp + edge.edge_fp + edge.edge_fn
            tot["w"] += w
            tot["adj"] += edge.adj_edge_jaccard * w

        adj = tot["adj"] / tot["w"] if tot["w"] else 0.0
        jp = jaccard(*[tot["p"][i] for i in (0, 1, 2)])
        jq = jaccard(*[tot["q"][i] for i in (0, 1, 2)])
        print(f"{'TOTAL':<18} {adj:>8.4f} {tot['p'][0] + tot['p'][1]:>6} "
              f"{tot['p'][0]}/{tot['p'][1]}/{tot['p'][2]:<14} "
              f"{tot['q'][0]}/{tot['q'][1]}/{tot['q'][2]:<14}")
        print(f"  division Jaccard   patched {jp:.4f}   pre-patch {jq:.4f}"
              + (f"   (ratio {jq / jp:.2f}x)" if jp else ""))
        print(f"  score              patched {adj + 0.1 * jp:.4f}   "
              f"as the public validator reports it {adj + 0.1 * jq:.4f}")

if __name__ == "__main__":
    main()
