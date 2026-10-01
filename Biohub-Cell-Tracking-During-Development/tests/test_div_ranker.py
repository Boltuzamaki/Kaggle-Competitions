"""Pin the fork-ranking key injected into `15_div_rank`.

The public stack spends its fork budget from the front of a queue sorted by
``parent_dist + 0.15 * sister_dist`` ascending. Measured on all 151 labelled
divisions (`13_div_inventory`): none has ``parent_dist`` below 2 um and only one
below 3 um, so that key puts a band containing no real divisions at the head of
the queue. The replacement is a Gaussian log-likelihood over the measured
division geometry.

These tests guard the two things that could silently break it:

* the **sign** - the key is consumed by ``proposals.sort()``, so lower must
  still mean "spend a slot here". Flipping it would be invisible until a
  multi-hour GPU run reported a worse score for an unrelated-looking reason.
* the **shape** - the likelihood must peak near the measured median and fall
  off on *both* sides, which is precisely what a monotone distance key cannot do.

    PYTHONPATH=src python -m pytest tests/test_div_ranker.py -q -s
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebooks" / "15_div_rank.ipynb"
START = "# ----------------------------------------------------------- DIV RANKER ---"
END = "# ------------------------------------------------------- END DIV RANKER ---"

# from 13_div_inventory over all 199 films
MEDIAN_PARENT_UM = 7.130
MEDIAN_SISTER_UM = 10.570

@pytest.fixture(scope="module")
def key():
    if not NOTEBOOK.is_file():
        pytest.skip("15_div_rank.ipynb not built (run scripts/pubfork.py)")
    src = "".join(json.load(NOTEBOOK.open())["cells"][2]["source"])
    if START not in src or END not in src:
        pytest.fail("division ranker block missing from the built notebook")
    ns: dict = dict(np=np, math=math, os=os)
    exec(src[src.index(START): src.index(END)], ns)
    return ns["safe_div_rank_key"]

def test_median_geometry_beats_a_tight_duplicate(key):
    """A textbook division must sort ahead of a 1 um duplicate detection."""
    division = key(MEDIAN_PARENT_UM, MEDIAN_SISTER_UM, MEDIAN_PARENT_UM)
    duplicate = key(1.0, 1.0, 1.0)
    print(f"  division {division:.3f}   duplicate {duplicate:.3f}")
    assert division < duplicate, "the ranker prefers the duplicate - sign is wrong"

def test_response_is_two_sided(key):
    """Too close and too far must both score worse than the median.

    This is the property the stock key lacks: being monotone in distance, it can
    only ever prefer one end.
    """
    at_median = key(MEDIAN_PARENT_UM, MEDIAN_SISTER_UM, MEDIAN_PARENT_UM)
    too_close = key(1.5, MEDIAN_SISTER_UM, 1.5)
    too_far = key(30.0, MEDIAN_SISTER_UM, 30.0)
    print(f"  close {too_close:.3f}   median {at_median:.3f}   far {too_far:.3f}")
    assert at_median < too_close
    assert at_median < too_far

def test_geometric_mode_restores_the_stock_key(key):
    """`BIOHUB_SAFE_DIV_RANK_MODE=geometric` must reproduce the original exactly,
    so the change can be A/B'd against the stack it forks."""
    if os.environ.get("BIOHUB_SAFE_DIV_RANK_MODE", "likelihood") != "likelihood":
        pytest.skip("ranker mode overridden in the environment")
    # rebuild the key with the geometric mode selected
    src = "".join(json.load(NOTEBOOK.open())["cells"][2]["source"])
    ns: dict = dict(np=np, math=math, os=os)
    saved = os.environ.get("BIOHUB_SAFE_DIV_RANK_MODE")
    os.environ["BIOHUB_SAFE_DIV_RANK_MODE"] = "geometric"
    try:
        exec(src[src.index(START): src.index(END)], ns)
        geometric = ns["safe_div_rank_key"]
        for p, s in [(3.0, 5.0), (7.1, 10.6), (12.0, 18.0)]:
            assert geometric(p, s, p) == pytest.approx(p + 0.15 * s)
    finally:
        if saved is None:
            os.environ.pop("BIOHUB_SAFE_DIV_RANK_MODE", None)
        else:
            os.environ["BIOHUB_SAFE_DIV_RANK_MODE"] = saved

def test_bad_features_do_not_dominate(key):
    """A NaN or absurd feature must be penalised, not allowed to sort first."""
    good = key(MEDIAN_PARENT_UM, MEDIAN_SISTER_UM, MEDIAN_PARENT_UM)
    for bad in (float("nan"), float("inf"), -5.0, 1e9):
        v = key(bad, MEDIAN_SISTER_UM, MEDIAN_PARENT_UM)
        assert np.isfinite(v), f"key returned {v} for parent_dist={bad}"
        assert v > good, f"parent_dist={bad} sorted ahead of the median division"

def test_ranks_real_divisions_ahead_of_tight_pairs(key):
    """End-to-end ordering check on the measured division geometry.

    The negatives are *simulated* duplicate detections, so the separation here
    is not a real-data result - it confirms the mechanism, not the size of the
    win. The honest number comes from the held-out run.
    """
    rng = np.random.default_rng(0)
    parents = rng.normal(MEDIAN_PARENT_UM, 2.2, 200).clip(2.9, 14.0)
    sisters = rng.normal(MEDIAN_SISTER_UM, 3.2, 200).clip(3.0, 20.0)
    pos = [key(p, s, p) for p, s in zip(parents, sisters)]

    dup_off = rng.uniform(0.2, 3.0, 2000)
    neg = [key(float(o), float(o), float(rng.uniform(4, 10))) for o in dup_off]

    ranked = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    top = [lab for _, lab in ranked[:50]]
    print(f"  precision in the top 50 slots: {np.mean(top):.1%}")
    assert np.mean(top) > 0.8
