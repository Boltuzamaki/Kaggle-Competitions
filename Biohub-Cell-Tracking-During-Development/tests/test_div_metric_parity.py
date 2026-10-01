"""Verify `biohub_div_metric` reproduces the organizers' **patched** division scorer.

The division half of the score is worth 0.1 and is the part almost nobody is
collecting, so it has to be measurable inside a Kaggle notebook - where
`tracksdata` does not exist. This test pins the port to the reference.

Real data is a poor test here: divisions are rare (151 across all 199 training
films) and the two locally-scorable videos contain almost none, so the cases
that actually separate the patched rule from the exploited one would never fire.
The graphs below are therefore synthetic and each one targets a specific clause
of `division_metrics.py`: the hub exploit, the immediate-predecessor allowance,
the distinct-branch requirement, cross-component evidence, merged branches, the
grandchild fallback, and the bipartite pairing that stops one fork claiming two
divisions.

    PYTHONPATH=src python -m pytest tests/test_div_metric_parity.py -q -s
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from biohub_ct import SCALE, TrackGraph  # noqa: E402
import biohub_div_metric as D  # noqa: E402

warnings.filterwarnings("ignore")

# One voxel step in y is 0.40625 um, so a "far" offset has to be large in voxels
# to clear the 7 um matching radius. Keep everything in voxel units, as the
# pipeline does, and let SCALE do the conversion.
FAR = 40.0   # 16.25 um - unmatchable
NEAR = 1.0   # 0.41 um - comfortably matched

def graph(nodes: dict[int, tuple[float, float, float, float]],
          edges: list[tuple[int, int]]) -> TrackGraph:
    """``nodes`` maps id -> (t, z, y, x) in voxel units."""
    ids = np.array(sorted(nodes), dtype=np.int64)
    rows = np.array([nodes[int(i)] for i in ids], dtype=np.float64)
    return TrackGraph(
        t=rows[:, 0].astype(np.int64), z=rows[:, 1], y=rows[:, 2], x=rows[:, 3],
        ids=ids,
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
        meta={},
    )

def official(pred: TrackGraph, gt: TrackGraph) -> tuple[int, int, int]:
    from local_eval import _ensure_metrics_importable, to_tracksdata

    _ensure_metrics_importable()
    from tracking_cellmot.division_metrics import evaluate_divisions

    counts = evaluate_divisions(to_tracksdata(pred), to_tracksdata(gt),
                                scale=tuple(SCALE), max_distance=7.0)
    return counts.tp, counts.fn, counts.fp

def ours(pred: TrackGraph, gt: TrackGraph) -> tuple[int, int, int]:
    c = D.evaluate_divisions(pred, gt)
    return c.tp, c.fn, c.fp

# --------------------------------------------------------------------------
# a canonical GT division:  0 -> 1 -> {2, 3};  2 -> 4,  3 -> 5
# --------------------------------------------------------------------------

GT_NODES = {
    0: (0, 0.0, 0.0, 0.0),     # grandparent
    1: (1, 0.0, 0.0, 0.0),     # divider
    2: (2, 0.0, -4.0, 0.0),    # daughter A
    3: (2, 0.0, 4.0, 0.0),     # daughter B
    4: (3, 0.0, -8.0, 0.0),    # granddaughter A
    5: (3, 0.0, 8.0, 0.0),     # granddaughter B
}
GT_EDGES = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5)]
GT = graph(GT_NODES, GT_EDGES)

def _shift(nodes: dict, dy: float) -> dict:
    return {k: (t, z, y + dy, x) for k, (t, z, y, x) in nodes.items()}

def perfect_pred() -> TrackGraph:
    """The prediction that reproduces the GT division exactly."""
    return graph({k: (t, z, y + NEAR, x) for k, (t, z, y, x) in GT_NODES.items()}, GT_EDGES)

def hub_exploit_pred() -> TrackGraph:
    """The patched-out exploit: one out-of-volume hub plus a chain of fake forks.

    The parent and both daughters are detected and everything is dragged into a
    single weakly connected component that contains forks. Under the pre-patch
    rule this scored a true positive; under the patched rule it must not, because
    no fork is the matched parent or its immediate predecessor.
    """
    nodes = {k: (t, z, y + NEAR, x) for k, (t, z, y, x) in GT_NODES.items()}
    # linear tracks only - no real fork anywhere near the division
    edges = [(0, 1), (1, 2), (2, 4), (3, 5)]
    hub, a, b, c = 100, 101, 102, 103
    nodes[hub] = (-1000, -10000.0, -10000.0, -10000.0)
    nodes[a] = (-999, -10000.0, -10000.0, -10000.0)
    nodes[b] = (-998, -10000.0, -10000.0, -10000.0)
    nodes[c] = (-998, -10000.0, -10000.0, -10001.0)
    edges += [(hub, 0), (hub, 3), (hub, a), (a, b), (a, c)]
    return graph(nodes, edges)

def fork_on_grandparent_pred() -> TrackGraph:
    """The fork sits on the grandparent, one frame early.

    ``_is_strongly_connected_division`` allows the fork to be the matched parent
    *or its immediate predecessor*, and the parent side covers both the divider
    and the grandparent, so this is still reachable.
    """
    nodes = {k: (t, z, y + NEAR, x) for k, (t, z, y, x) in GT_NODES.items()}
    return graph(nodes, [(0, 1), (0, 3), (1, 2), (2, 4), (3, 5)])

def one_daughter_lineage_pred() -> TrackGraph:
    """A local fork, but only one GT daughter lineage has any matched evidence.

    Daughter B is never detected and the fork's second branch sits at y = -30,
    out of matching range of *both* GT daughters. ``_matched_division_nodes``
    then sees fewer than two non-empty daughter sets and the division is a
    false negative even though a fork is present in the right place.
    """
    nodes = {k: (t, z, y + NEAR, x) for k, (t, z, y, x) in GT_NODES.items()}
    nodes[6] = (2, 0.0, -30.0, 0.0)
    del nodes[3]
    del nodes[5]
    return graph(nodes, [(0, 1), (1, 2), (1, 6), (2, 4)])

def duplicate_detection_pred() -> TrackGraph:
    """A duplicate detection of daughter A, with daughter B missing entirely.

    Worth pinning because it is counter-intuitive: this scores a **true
    positive**. Matching is a per-timepoint *global* assignment, not a
    nearest-neighbour lookup, so with two predictions and two GT daughters in
    range the optimal assignment pairs each prediction with a different
    daughter - the duplicate is handed to daughter A and the real detection to
    daughter B, whose own node was never predicted at all.

    The practical consequence for the fork ranker: over-proposing forks near a
    real division is not symmetric with proposing them elsewhere. A duplicate
    that lands inside the radius can complete a division the detector missed.
    """
    nodes = {k: (t, z, y + NEAR, x) for k, (t, z, y, x) in GT_NODES.items()}
    nodes[6] = (2, 0.0, -16.0, 0.0)
    del nodes[3]
    del nodes[5]
    return graph(nodes, [(0, 1), (1, 2), (1, 6), (2, 4)])

def merged_branch_pred() -> TrackGraph:
    """A child with two parents - the 'malformed' branch rule."""
    nodes = {k: (t, z, y + NEAR, x) for k, (t, z, y, x) in GT_NODES.items()}
    nodes[7] = (1, 0.0, 6.0, 0.0)
    return graph(nodes, [(0, 1), (1, 2), (1, 3), (7, 3), (2, 4), (3, 5)])

def spurious_fork_pred() -> TrackGraph:
    """A perfect division plus an extra fork on another annotated cell.

    The extra fork is 'evaluable' (it matches an annotated GT node that has a
    child), so it must be counted as a false positive even though it is nowhere
    near the real division.
    """
    nodes = {k: (t, z, y + NEAR, x) for k, (t, z, y, x) in GT_NODES.items()}
    nodes[8] = (3, 0.0, -8.0 + NEAR, 4.0)
    return graph(nodes, GT_EDGES + [(2, 8)])

def unannotated_fork_pred() -> TrackGraph:
    """A perfect division plus a fork far from any annotation.

    Spurious forks on unannotated cells are invisible to the metric, exactly as
    unannotated edges are. This is why the score tolerates a wide candidate pool
    and why a *ranking* fix is worth more than a *gating* fix.
    """
    nodes = {k: (t, z, y + NEAR, x) for k, (t, z, y, x) in GT_NODES.items()}
    nodes[10] = (0, 0.0, FAR, 0.0)
    nodes[11] = (1, 0.0, FAR - 2.0, 0.0)
    nodes[12] = (1, 0.0, FAR + 2.0, 0.0)
    return graph(nodes, GT_EDGES + [(10, 11), (10, 12)])

def two_divisions_one_fork():
    """Two GT divisions, one predicted fork that is local to both.

    The bipartite pairing must give the fork to a single division, leaving the
    other a false negative.
    """
    gt_nodes = dict(GT_NODES)
    # a second division sharing the same parent-side neighbourhood
    gt_nodes.update({
        20: (0, 0.0, 1.0, 0.0), 21: (1, 0.0, 1.0, 0.0),
        22: (2, 0.0, -3.0, 0.0), 23: (2, 0.0, 5.0, 0.0),
        24: (3, 0.0, -7.0, 0.0), 25: (3, 0.0, 9.0, 0.0),
    })
    gt_edges = GT_EDGES + [(20, 21), (21, 22), (21, 23), (22, 24), (23, 25)]
    pred = graph({k: (t, z, y + NEAR, x) for k, (t, z, y, x) in GT_NODES.items()}, GT_EDGES)
    return pred, graph(gt_nodes, gt_edges)

def no_gt_division():
    """A linear GT track with a spurious predicted fork on it."""
    gt = graph({0: (0, 0.0, 0.0, 0.0), 1: (1, 0.0, 0.0, 0.0), 2: (2, 0.0, 0.0, 0.0)},
               [(0, 1), (1, 2)])
    pred = graph({0: (0, 0.0, NEAR, 0.0), 1: (1, 0.0, NEAR, 0.0),
                  2: (2, 0.0, NEAR, 0.0), 3: (2, 0.0, NEAR + 3.0, 0.0)},
                 [(0, 1), (1, 2), (1, 3)])
    return pred, gt

def missing_daughter_pred() -> TrackGraph:
    """One daughter lineage never detected - an unreachable division."""
    nodes = {k: (t, z, y + NEAR, x) for k, (t, z, y, x) in GT_NODES.items()
             if k not in (3, 5)}
    return graph(nodes, [(0, 1), (1, 2), (2, 4)])

CASES = [
    ("perfect", perfect_pred(), GT),
    ("hub_exploit", hub_exploit_pred(), GT),
    ("fork_on_grandparent", fork_on_grandparent_pred(), GT),
    ("one_daughter_lineage", one_daughter_lineage_pred(), GT),
    ("duplicate_detection", duplicate_detection_pred(), GT),
    ("merged_branch", merged_branch_pred(), GT),
    ("spurious_evaluable_fork", spurious_fork_pred(), GT),
    ("unannotated_fork", unannotated_fork_pred(), GT),
    ("two_divisions_one_fork", *two_divisions_one_fork()),
    ("no_gt_division", *no_gt_division()),
    ("missing_daughter", missing_daughter_pred(), GT),
]

@pytest.mark.parametrize("name,pred,gt", CASES, ids=[c[0] for c in CASES])
def test_parity_with_official(name, pred, gt):
    try:
        expected = official(pred, gt)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"official division scorer unavailable: {e}")
    got = ours(pred, gt)
    print(f"  {name:26s} official tp/fn/fp={expected}  ours={got}")
    assert got == expected, f"{name}: ours {got} != official {expected}"

def test_hub_exploit_scores_zero():
    """The regression that motivates the whole file.

    Whatever the reference says, the hub must not turn a merely-detected
    division into a true positive - that is the behaviour patched out on
    2026-07-17, and the public stack's own validator still rewards it.
    """
    tp, _, _ = ours(hub_exploit_pred(), GT)
    assert tp == 0

def test_perfect_prediction_scores_one():
    assert ours(perfect_pred(), GT) == (1, 0, 0)
