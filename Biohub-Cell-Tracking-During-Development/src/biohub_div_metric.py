"""Dependency-free port of the **patched** official division metric.

``tracking_cellmot.division_metrics`` needs ``tracksdata`` + ``polars``, neither
of which exists in the Kaggle image, so the division half of the score has never
been measurable inside a notebook. This module reproduces it with plain
dicts and sets (plus ``biohub_metric.match_nodes`` for the distance matching),
and is checked against the real implementation by ``tests/test_div_metric_parity.py``.

Why this matters more than a convenience
----------------------------------------
The organizers patched the division metric on **2026-07-17** (``aa65e90``,
"updating metric to patch weakly connected component exploit"). Before the
patch a GT division counted as recovered whenever its parent and both daughters
landed anywhere in one weakly connected component of the prediction that held a
fork *somewhere*. After it the rule is local:

* the predicted fork must be the matched parent **or its immediate predecessor**;
* the two GT daughter lineages must land on **two distinct direct-child
  branches** of that fork.

The widely-forked public 0.947 stack still scores divisions with the *pre-patch*
rule in its own held-out validator, so it reports roughly **double** the official
division Jaccard - 0.2500 where the official scorer says 0.1250 on the same
prediction. Every safe-division threshold tuned against that validator was tuned
against a compass reading double. That is the reason this file exists.

Faithfully reproduced details, each of which changes the count:

* Matching is done **per GT division window**, against a six-node subgraph
  (grandparent, divider, two children, their children) - *not* against the full
  GT graph. A fresh matching per division is what the official code does, and
  a pred node may take a different partner in each one.
* Candidate forks are restricted to the matched parent side and its immediate
  successors; everything else is out of scope for that division.
* A fork whose two direct-child branches carry nearest matched evidence in two
  *different* GT weakly connected components is rejected outright, as is one
  whose local branches are merged (a child with more than one parent).
* Direct-child evidence takes precedence over grandchild evidence; grandchildren
  are a fallback only when the child itself is unmatched.
* A maximum-cardinality bipartite matching pairs each pred fork with at most one
  GT division, so one lucky fork cannot claim several divisions.
* **False positives are unioned, not summed**: ``fp = |considered u evaluable u
  invalid| - |tp|``. ``evaluable`` is every predicted fork matched to an
  annotated GT node that has at least one child - so a spurious fork on an
  *unannotated* cell is invisible to the metric, exactly as for edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biohub_ct import SCALE, TrackGraph
from biohub_metric import MAX_DISTANCE_UM, match_nodes

@dataclass
class DivisionCounts:
    tp: int
    fn: int
    fp: int

    @property
    def jaccard(self) -> float:
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom else 0.0

# --------------------------------------------------------------------------
# adjacency helpers - the official code calls methods on a graph object, so
# these keep the port readable next to it
# --------------------------------------------------------------------------

def _adjacency(graph: TrackGraph) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """``(successors, predecessors)`` keyed by node id, every node present."""
    succ: dict[int, list[int]] = {int(i): [] for i in graph.ids}
    pred: dict[int, list[int]] = {int(i): [] for i in graph.ids}
    for s, d in graph.edges:
        s, d = int(s), int(d)
        if s in succ and d in pred:
            succ[s].append(d)
            pred[d].append(s)
    return succ, pred

def _forks(succ: dict[int, list[int]]) -> set[int]:
    return {n for n, outs in succ.items() if len(outs) >= 2}

def _subgraph_nodes(graph_succ: dict[int, list[int]], graph_pred: dict[int, list[int]],
                    div_node: int) -> set[int]:
    """The official ``extract_divisions`` window around one dividing node."""
    children = graph_succ.get(div_node, [])
    grandchildren = [gc for c in children for gc in graph_succ.get(c, [])]
    return {div_node, *graph_pred.get(div_node, []), *children, *grandchildren}

def _bipartite_max_matching(left: list, edges: dict) -> dict:
    """Maximum-cardinality bipartite matching via DFS augmenting paths.

    A direct transcription of the official helper, including the fact that it
    returns only the ``left -> right`` side of the pairing.
    """
    match_r: dict = {}
    match_l: dict = {}

    def augment(u, seen: set) -> bool:
        for v in edges.get(u, ()):
            if v in seen:
                continue
            seen.add(v)
            if v not in match_r or augment(match_r[v], seen):
                match_l[u] = v
                match_r[v] = u
                return True
        return False

    for u in left:
        augment(u, set())
    return match_l

def _weak_components(graph: TrackGraph) -> dict[int, int]:
    """Map each node id to a representative id of its weakly connected component."""
    succ, pred = _adjacency(graph)
    comp: dict[int, int] = {}
    for seed in (int(i) for i in graph.ids):
        if seed in comp:
            continue
        comp[seed] = seed
        stack = [seed]
        while stack:
            cur = stack.pop()
            for nbr in succ.get(cur, ()) + pred.get(cur, ()):
                if nbr not in comp:
                    comp[nbr] = seed
                    stack.append(nbr)
    return comp

def _sub_track_graph(graph: TrackGraph, keep: set[int]) -> TrackGraph:
    """The induced subgraph on ``keep``, as a TrackGraph (for matching against)."""
    mask = np.isin(graph.ids, np.fromiter(keep, dtype=np.int64, count=len(keep)))
    idx = np.flatnonzero(mask)
    if len(graph.edges):
        e = graph.edges
        emask = np.isin(e[:, 0], graph.ids[idx]) & np.isin(e[:, 1], graph.ids[idx])
        edges = e[emask]
    else:
        edges = graph.edges
    return TrackGraph(t=graph.t[idx], z=graph.z[idx], y=graph.y[idx], x=graph.x[idx],
                      ids=graph.ids[idx], edges=edges, meta=dict(graph.meta))

# --------------------------------------------------------------------------
# the metric
# --------------------------------------------------------------------------

def _matched_division_nodes(pred_to_gt: dict[int, int], gt_succ: dict[int, list[int]],
                            gt_pred: dict[int, list[int]], divider: int):
    """Group matched pred nodes into the parent side and each daughter lineage.

    Returns ``None`` when the window cannot be evaluated - fewer than two GT
    children, no matched parent, or fewer than two daughter lineages with any
    matched evidence. The official code treats all three the same way.
    """
    if not pred_to_gt:
        return None
    gt_children = gt_succ.get(divider, [])
    if len(gt_children) < 2:
        return None

    gt_parent_ids = {divider, *gt_pred.get(divider, [])}
    parent_ids = {p for p, g in pred_to_gt.items() if g in gt_parent_ids}
    daughter_ids = [
        {p for p, g in pred_to_gt.items() if g in {child, *gt_succ.get(child, [])}}
        for child in gt_children
    ]
    if not parent_ids or sum(bool(ids) for ids in daughter_ids) < 2:
        return None
    return parent_ids, daughter_ids

def _is_strongly_connected_division(pred_succ: dict[int, list[int]],
                                    pred_pred: dict[int, list[int]],
                                    fork: int, parent_ids: set[int],
                                    daughter_ids: list[set[int]]) -> bool:
    """The patched local-topology test.

    The fork must *be* the matched parent or sit immediately after it, and the
    GT daughter lineages must reach two **distinct** direct-child branches of
    the fork. Sharing a weakly connected component is explicitly not enough --
    that was the exploit.
    """
    if {fork, *pred_pred.get(fork, [])}.isdisjoint(parent_ids):
        return False

    pred_lineages = [{c, *pred_succ.get(c, [])} for c in pred_succ.get(fork, [])]
    lineage_edges = {
        gt_lineage: {i for i, ids in enumerate(pred_lineages) if not matched.isdisjoint(ids)}
        for gt_lineage, matched in enumerate(daughter_ids)
    }
    return len(_bipartite_max_matching(list(lineage_edges), lineage_edges)) >= 2

def _branch_component_evidence(pred_succ: dict[int, list[int]],
                               pred_pred: dict[int, list[int]],
                               fork: int, child: int, pred_to_gt: dict[int, int],
                               gt_component: dict[int, int]) -> tuple[int | None, bool]:
    """One GT component id for a predicted child branch, plus a malformed flag.

    A matched child speaks for its own branch. Only an *unmatched* child falls
    back to its grandchildren, and only when they agree. A child (or grandchild)
    with more than one parent cannot be attributed to this fork at all, which is
    what the official code calls malformed.
    """
    if set(pred_pred.get(child, [])) != {fork}:
        return None, True
    if child in pred_to_gt:
        return gt_component[pred_to_gt[child]], False

    grandchildren = pred_succ.get(child, [])
    if any(set(pred_pred.get(gc, [])) != {child} for gc in grandchildren):
        return None, True

    components = {gt_component[pred_to_gt[gc]] for gc in grandchildren if gc in pred_to_gt}
    return (next(iter(components)), False) if len(components) == 1 else (None, False)

def _pred_division_fork_sets(pred: TrackGraph, gt: TrackGraph,
                             max_distance: float) -> tuple[set[int], set[int], set[int]]:
    """``(evaluable, cross_component, malformed)`` predicted forks."""
    pred_to_gt = match_nodes(pred, gt, max_distance)
    pred_succ, pred_pred = _adjacency(pred)
    gt_succ, _ = _adjacency(gt)

    forks = _forks(pred_succ)
    evaluable = {f for f in forks
                 if f in pred_to_gt and len(gt_succ.get(pred_to_gt[f], [])) >= 1}

    gt_component = _weak_components(gt)
    cross_component: set[int] = set()
    malformed: set[int] = set()
    for fork in forks:
        evidence: list[int] = []
        for child in pred_succ.get(fork, []):
            component, is_malformed = _branch_component_evidence(
                pred_succ, pred_pred, fork, child, pred_to_gt, gt_component)
            if is_malformed:
                malformed.add(fork)
                break
            if component is not None:
                evidence.append(component)
        else:
            # no malformed branch: two branches in different GT components is
            # a fork that stitched two unrelated lineages together
            if len(set(evidence)) >= 2:
                cross_component.add(fork)

    return evaluable, cross_component, malformed

def evaluate_divisions(pred: TrackGraph, gt: TrackGraph,
                       max_distance: float = MAX_DISTANCE_UM,
                       scale: np.ndarray = SCALE) -> DivisionCounts:
    """TP / FN / FP for division events under the post-patch rule."""
    pred_succ, pred_pred = _adjacency(pred)
    gt_succ, gt_pred = _adjacency(gt)

    gt_divisions = {d: _subgraph_nodes(gt_succ, gt_pred, d) for d in _forks(gt_succ)}
    if not gt_divisions:
        # No GT divisions: every evaluable predicted fork is a false positive.
        evaluable, cross, malformed = _pred_division_fork_sets(pred, gt, max_distance)
        return DivisionCounts(tp=0, fn=0, fp=len(evaluable | cross | malformed))

    pred_forks = _forks(pred_succ)
    evaluable, cross, malformed = _pred_division_fork_sets(pred, gt, max_distance)
    invalid = cross | malformed

    candidates: dict[int, set[int]] = {}
    considered: set[int] = set()
    for divider, keep in gt_divisions.items():
        # A fresh matching against this division's window alone - the official
        # code copies the pred graph per division precisely so that a node is
        # free to partner differently in each one.
        pred_to_gt = match_nodes(pred, _sub_track_graph(gt, keep), max_distance, scale)
        matched_nodes = _matched_division_nodes(pred_to_gt, gt_succ, gt_pred, divider)
        if matched_nodes is None:
            candidates[divider] = set()
            continue

        parent_ids, daughter_ids = matched_nodes
        local_nodes = parent_ids | {s for p in parent_ids for s in pred_succ.get(p, [])}
        local_forks = local_nodes & pred_forks
        considered |= local_forks
        candidates[divider] = {
            f for f in local_forks - invalid
            if _is_strongly_connected_division(pred_succ, pred_pred, f, parent_ids, daughter_ids)
        }

    pairing = _bipartite_max_matching(list(candidates), candidates)
    tp_forks = set(pairing.values())
    fp_forks = (considered | evaluable | invalid) - tp_forks
    tp = len(pairing)
    return DivisionCounts(tp=tp, fn=len(candidates) - tp, fp=len(fp_forks))

def aggregate(counts: list[DivisionCounts]) -> DivisionCounts:
    """Micro-average across runs - the official aggregation for the division half."""
    return DivisionCounts(tp=sum(c.tp for c in counts),
                          fn=sum(c.fn for c in counts),
                          fp=sum(c.fp for c in counts))
