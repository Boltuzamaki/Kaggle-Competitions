"""The division metric *as injected into the Kaggle notebook* must match the reference.

`test_div_metric_parity.py` pins `src/biohub_div_metric.py`. That is not enough:
the notebook carries a separate transcription that works on plain dicts and uses
the public stack's own `match_nodes_bipartite`. A divergence there would not
fail any other test - the GPU run would simply complete and report a number
that answers a different question, which is exactly the failure mode session 1
paid for.

So this extracts the injected block straight out of the built notebook, execs it
against a faithful copy of the notebook's matcher, and replays the parity cases.

    PYTHONPATH=src python -m pytest tests/test_notebook_div_metric.py -q -s
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

warnings.filterwarnings("ignore")

NOTEBOOK = REPO / "notebooks" / "12_div_probe.ipynb"
START = "# ---------------------------------------------------------------- PATCHED ---"
END = "# ------------------------------------------------------------ END PATCHED ---"

VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)

def match_nodes_bipartite(pred_nodes, gt_nodes, max_dist=7.0):
    """Byte-for-byte behaviour of the public notebook's matcher."""
    pred_by_t, gt_by_t = {}, {}
    for pid, (t, *_r) in pred_nodes.items():
        pred_by_t.setdefault(int(t), []).append(pid)
    for gid, (t, *_r) in gt_nodes.items():
        gt_by_t.setdefault(int(t), []).append(gid)
    p2g, g2p = {}, {}
    for t, p_ids in pred_by_t.items():
        g_ids = gt_by_t.get(t, [])
        if not g_ids:
            continue
        vs = np.array(VOXEL_SCALE_UM, dtype=float)
        p_pos = np.array([pred_nodes[p][1:] for p in p_ids], dtype=float) * vs
        g_pos = np.array([gt_nodes[g][1:] for g in g_ids], dtype=float) * vs
        cost = np.sqrt(((p_pos[:, None, :] - g_pos[None, :, :]) ** 2).sum(-1))
        BIG = 1e6
        cg = np.where(cost <= max_dist, cost, BIG)
        for r, c in zip(*linear_sum_assignment(cg)):
            if cg[r, c] >= BIG:
                continue
            p2g[p_ids[r]] = g_ids[c]
            g2p[g_ids[c]] = p_ids[r]
    return p2g, g2p

def injected_fn():
    if not NOTEBOOK.is_file():
        pytest.skip(f"{NOTEBOOK.name} not built (run scripts/pubfork.py)")
    src = "".join(json.load(NOTEBOOK.open())["cells"][2]["source"])
    if START not in src or END not in src:
        pytest.fail("patched division-metric block missing from the built notebook")
    block = src[src.index(START): src.index(END)]
    ns = dict(np=np, linear_sum_assignment=linear_sum_assignment,
              VOXEL_SCALE_UM=VOXEL_SCALE_UM, VALIDATOR_MATCH_RADIUS_UM=7.0,
              match_nodes_bipartite=match_nodes_bipartite)
    exec(block, ns)
    return ns["compute_division_confusion"]

def to_plain(g):
    nodes = {int(i): (int(t), float(z), float(y), float(x))
             for i, t, z, y, x in zip(g.ids, g.t, g.z, g.y, g.x)}
    return nodes, [(int(s), int(d)) for s, d in g.edges]

def _cases():
    import test_div_metric_parity as T
    return T

@pytest.mark.parametrize("idx", range(11))
def test_injected_matches_official(idx):
    T = _cases()
    if idx >= len(T.CASES):
        pytest.skip("case list shorter than expected")
    name, pred, gt = T.CASES[idx]
    try:
        o_tp, o_fn, o_fp = T.official(pred, gt)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"official division scorer unavailable: {e}")

    pn, pe = to_plain(pred)
    gn, ge = to_plain(gt)
    p2g, g2p = match_nodes_bipartite(pn, gn)
    # note the notebook's return order is (tp, fp, fn), not (tp, fn, fp)
    n_tp, n_fp, n_fn = injected_fn()(pn, pe, gn, ge, p2g, g2p)
    print(f"  {name:<26} official {o_tp}/{o_fn}/{o_fp}   notebook {n_tp}/{n_fn}/{n_fp}")
    assert (n_tp, n_fn, n_fp) == (o_tp, o_fn, o_fp), name
