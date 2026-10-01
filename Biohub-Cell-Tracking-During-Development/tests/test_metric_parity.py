"""Verify `biohub_metric` reproduces the organizers' scorer exactly.

The reimplementation exists so the pipeline can be scored on Kaggle, where
`tracksdata` is not installed. It is only trustworthy if it agrees with the
reference on real data, so this test asserts that directly and is skipped
(loudly) when the reference stack or the data is unavailable.

    PYTHONPATH=src python -m pytest tests/test_metric_parity.py -q -s
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import biohub_ct as B  # noqa: E402
import biohub_metric as M  # noqa: E402

warnings.filterwarnings("ignore")

DATA = REPO / "data" / "comp"

def scorable() -> list[str]:
    out = []
    if not (DATA / "test").is_dir():
        return out
    for zp in sorted((DATA / "test").glob("*.zarr")):
        if not (DATA / "train" / f"{zp.stem}.geff").exists():
            continue
        try:
            vol = B.open_volume(zp)
        except Exception:
            continue
        c = zp / "0" / "c"
        if c.is_dir() and len(list(c.iterdir())) >= vol.n_t:
            out.append(zp.stem)
    return out

@pytest.mark.parametrize("min_track_len,max_link_um", [(4, 7.0), (1, 5.0), (6, 8.0)])
def test_matches_official_scorer(min_track_len, max_link_um):
    names = scorable()
    if not names:
        pytest.skip("no complete local dataset with ground truth")
    try:
        from local_eval import score_dataset
    except Exception as e:  # pragma: no cover
        pytest.skip(f"official scorer unavailable: {e}")

    cfg = B.Config(min_track_len=min_track_len, max_link_um=max_link_um)
    for name in names:
        vol = B.open_volume(DATA / "test" / f"{name}.zarr")
        graph = B.build_graph(B.detect_frames(vol, cfg), cfg)
        geff = DATA / "train" / f"{name}.geff"

        try:
            ref = score_dataset(graph, geff)
        except Exception as e:  # pragma: no cover
            pytest.skip(f"official scorer failed: {e}")

        gt = B.read_geff(geff)
        mine = M.evaluate(graph, gt, n_total=gt.meta.get("estimated_number_of_nodes"))

        assert mine.edge_tp == ref["edge_tp"], f"{name}: TP {mine.edge_tp} vs {ref['edge_tp']}"
        assert mine.edge_fp == ref["edge_fp"], f"{name}: FP {mine.edge_fp} vs {ref['edge_fp']}"
        assert mine.edge_fn == ref["edge_fn"], f"{name}: FN {mine.edge_fn} vs {ref['edge_fn']}"
        assert mine.num_pred_nodes == ref["num_pred_nodes"]
        assert np.isclose(mine.edge_jaccard, ref["edge_jaccard"], atol=1e-9)
        assert np.isclose(mine.adj_edge_jaccard, ref["adj_edge_jaccard"], atol=1e-9)
        assert np.isclose(mine.node_recall, ref["node_recall"], atol=1e-9)

def test_summarise_matches_official():
    names = scorable()
    if len(names) < 2:
        pytest.skip("need at least two complete datasets")
    try:
        from local_eval import score_dataset, summarise_rows
    except Exception as e:  # pragma: no cover
        pytest.skip(f"official scorer unavailable: {e}")

    cfg = B.Config()
    ref_rows, mine_rows = [], []
    for name in names:
        vol = B.open_volume(DATA / "test" / f"{name}.zarr")
        graph = B.build_graph(B.detect_frames(vol, cfg), cfg)
        geff = DATA / "train" / f"{name}.geff"
        ref_rows.append(score_dataset(graph, geff))
        gt = B.read_geff(geff)
        mine_rows.append(M.evaluate(graph, gt, n_total=gt.meta.get("estimated_number_of_nodes")))

    ref = summarise_rows(ref_rows)
    mine = M.summarise(mine_rows)
    assert np.isclose(mine["edge_jaccard"], ref["edge_jaccard"], atol=1e-9)
    assert np.isclose(mine["adj_edge_jaccard"], ref["adj_edge_jaccard"], atol=1e-9)
