"""Unit tests for the tracking library.

Everything here runs on synthetic fixtures, so the suite needs no competition
data and no network:

    PYTHONPATH=src .venv/bin/python -m pytest tests/ -q

Real-data checks live in `notebooks/00_smoke_test.ipynb`, which runs on Kaggle.
The split matters: every failure this repo has actually hit was an *environment*
problem (wrong package version, wrong file inlined) that a local unit test
cannot see.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import biohub_ct as B  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def straight_track(n_frames: int = 10, step_um: float = 2.0) -> list[np.ndarray]:
    """One cell drifting along +x at a constant physical speed."""
    dx = step_um / B.SCALE[2]
    return [np.array([[32.0, 128.0, 100.0 + t * dx]]) for t in range(n_frames)]

def two_tracks(n_frames: int = 10, sep_um: float = 30.0) -> list[np.ndarray]:
    dy = sep_um / B.SCALE[1]
    return [np.array([[32.0, 100.0, 100.0 + t],
                      [32.0, 100.0 + dy, 100.0 + t]]) for t in range(n_frames)]

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_scale_is_physical_and_anisotropic():
    assert B.SCALE.shape == (3,)
    assert B.SCALE[0] > B.SCALE[1] == B.SCALE[2], "z must be the coarse axis"
    # xy_downsample=4 is chosen to make the grid isotropic; guard that identity.
    assert np.isclose(B.SCALE[1] * 4, B.SCALE[0])

def test_ball_footprint_covers_requested_radius():
    spacing = np.array([B.SCALE[0], B.SCALE[1] * 4, B.SCALE[2] * 4])
    fp = B._ball_footprint(5.0, spacing)
    assert fp.ndim == 3 and fp.dtype == bool
    assert fp[fp.shape[0] // 2, fp.shape[1] // 2, fp.shape[2] // 2], "centre must be included"
    assert fp.sum() > 1

def test_embryo_of():
    assert B.embryo_of("44b6_0113de3b") == "44b6"
    assert B.embryo_of("6bba_05db0fb1") == "6bba"

# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------

def test_link_motion_recovers_a_straight_track():
    frames = straight_track(10)
    g = B.link_motion(frames, max_link_um=7.0)
    assert g.n_nodes == 10
    assert g.n_edges == 9, "a single 10-frame track has exactly 9 links"
    for s, d in g.edges:
        assert g.t[list(g.ids).index(d)] - g.t[list(g.ids).index(s)] == 1

def test_link_motion_respects_the_gate():
    # 40 um per frame is far beyond any plausible gate.
    frames = straight_track(5, step_um=40.0)
    g = B.link_motion(frames, max_link_um=7.0)
    assert g.n_edges == 0

def test_link_motion_keeps_parallel_tracks_separate():
    frames = two_tracks(8)
    g = B.link_motion(frames, max_link_um=7.0)
    assert g.n_nodes == 16
    assert g.n_edges == 14, "two tracks x 7 links"
    # No node may have more than one successor here.
    out = {}
    for s, _ in g.edges:
        out[int(s)] = out.get(int(s), 0) + 1
    assert max(out.values()) == 1

def test_link_motion_handles_empty_frames():
    frames = [np.zeros((0, 3)), np.array([[1.0, 2.0, 3.0]]), np.zeros((0, 3))]
    g = B.link_motion(frames, max_link_um=7.0)
    assert g.n_nodes == 1 and g.n_edges == 0

# ---------------------------------------------------------------------------
# Gap closing
# ---------------------------------------------------------------------------

def test_close_gaps_bridges_a_single_missed_frame():
    frames = straight_track(9)
    frames[4] = np.zeros((0, 3))          # drop the detection at t=4
    g = B.link_motion(frames, max_link_um=7.0)
    assert g.n_edges == 6, "the gap splits the track into two pieces"

    bridged = B.close_gaps(g, max_gap=1, gap_um_per_frame=6.0)
    assert bridged.n_nodes == g.n_nodes + 1, "one interpolated node inserted"
    assert bridged.n_edges == g.n_edges + 2, "a bridge yields TWO matchable edges"

    # The synthetic node must sit at the missing timepoint.
    new_t = sorted(set(bridged.t.tolist()) - set(g.t.tolist()))
    assert new_t == [4]

def test_close_gaps_is_a_noop_without_gaps():
    g = B.link_motion(straight_track(8), max_link_um=7.0)
    assert B.close_gaps(g, 1, 6.0).n_edges == g.n_edges

def test_close_gaps_respects_distance_limit():
    frames = straight_track(9, step_um=5.0)
    frames[4] = np.zeros((0, 3))
    g = B.link_motion(frames, max_link_um=7.0)
    # The two ends are ~10 um apart; a 1 um/frame budget must refuse to bridge.
    assert B.close_gaps(g, max_gap=1, gap_um_per_frame=1.0).n_nodes == g.n_nodes

# ---------------------------------------------------------------------------
# Track filtering
# ---------------------------------------------------------------------------

def test_filter_short_tracks_removes_only_short_components():
    long_frames = straight_track(10)
    g_long = B.link_motion(long_frames, max_link_um=7.0)
    # An isolated speck 60 um away that never links to anything.
    speck = np.array([[32.0, 240.0, 240.0]])
    frames = [np.vstack([long_frames[t], speck]) if t == 0 else long_frames[t]
              for t in range(10)]
    g = B.link_motion(frames, max_link_um=7.0)
    filtered = B.filter_short_tracks(g, min_len=4)
    assert filtered.n_nodes == g_long.n_nodes
    assert filtered.n_edges == g_long.n_edges

def test_filter_short_tracks_min_len_one_is_a_noop():
    g = B.link_motion(straight_track(5), max_link_um=7.0)
    assert B.filter_short_tracks(g, min_len=1).n_nodes == g.n_nodes

def test_subset_drops_dangling_edges():
    g = B.link_motion(straight_track(5), max_link_um=7.0)
    keep = {int(i) for i in g.ids[:2]}
    sub = B.subset(g, keep)
    assert sub.n_nodes == 2
    ids = set(int(i) for i in sub.ids)
    assert all(int(s) in ids and int(d) in ids for s, d in sub.edges)

# ---------------------------------------------------------------------------
# Divisions
# ---------------------------------------------------------------------------

def test_add_safe_divisions_never_exceeds_out_degree_two():
    dy = 3.0 / B.SCALE[1]
    frames = []
    for t in range(6):
        pts = [[32.0, 100.0, 100.0]]
        if t >= 3:                       # three candidate daughters appear
            pts += [[32.0, 100.0 + dy, 100.0], [32.0, 100.0 + 2 * dy, 100.0]]
        frames.append(np.array(pts))
    g = B.link_motion(frames, max_link_um=7.0)
    out = B.add_safe_divisions(g, max_parent_um=8.0, max_sister_um=12.0, frame_frac_cap=1.0)
    counts = {}
    for s, _ in out.edges:
        counts[int(s)] = counts.get(int(s), 0) + 1
    assert max(counts.values(), default=0) <= 2

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _synthetic_volume(centres, shape=(64, 256, 256), sigma=(1.2, 5.0, 5.0), amp=3000):
    vol = np.full(shape, 100.0, dtype=np.float32)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    for cz, cy, cx in centres:
        vol += amp * np.exp(-(((zz - cz) / sigma[0]) ** 2
                              + ((yy - cy) / sigma[1]) ** 2
                              + ((xx - cx) / sigma[2]) ** 2) / 2)
    return vol.astype(np.uint16)

def test_detect_dog_finds_well_separated_blobs():
    centres = [(20, 60, 60), (32, 128, 128), (44, 190, 190)]
    vol = _synthetic_volume(centres)
    cfg = B.Config()
    coords, scores = B.detect_dog(vol, None, cfg.xy_downsample, cfg.dog_scales,
                                  cfg.min_distance_um, cfg.rel_threshold, cfg.max_peaks)
    assert len(coords) >= len(centres)
    gt = np.array(centres, dtype=float)
    d = np.sqrt((((coords[:, None, :] - gt[None, :, :]) * B.SCALE) ** 2).sum(2)).min(0)
    assert (d <= 7.0).all(), f"blobs not matched within the 7 um gate: {d}"

def test_detect_dog_returns_scores_in_descending_order():
    vol = _synthetic_volume([(32, 128, 128), (32, 64, 64)])
    cfg = B.Config()
    _, scores = B.detect_dog(vol, None, cfg.xy_downsample, cfg.dog_scales,
                             cfg.min_distance_um, cfg.rel_threshold, cfg.max_peaks)
    assert np.all(np.diff(scores) <= 1e-9)

def test_refine_centroids_rejects_large_shifts():
    vol = _synthetic_volume([(32, 128, 128)])
    far = np.array([[32.0, 40.0, 40.0]])            # nowhere near a blob
    out = B.refine_centroids(vol, far, max_shift_um=0.5)
    assert np.allclose(out, far), "a shift beyond the cap must be rejected"

def test_normalize_frame_uses_supplied_quantiles():
    vol = _synthetic_volume([(32, 128, 128)])
    a = B.normalize_frame(vol, {"0.01": 100.0, "0.99": 1100.0})
    b = B.normalize_frame(vol, None)
    assert a.dtype == np.float32 and np.isfinite(a).all()
    assert not np.allclose(a, b), "quantiles should change the normalisation"

# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def test_submission_roundtrip_and_validation(tmp_path):
    g = B.build_graph(straight_track(10), B.Config(min_track_len=2, safe_divisions=False))
    out = tmp_path / "submission.csv"
    with B.SubmissionWriter(out) as w:
        w.add("ds_a", g)
        w.add("ds_b", g)

    stats = B.validate_submission(out, expected_datasets=["ds_a", "ds_b"])
    assert stats["n_datasets"] == 2
    assert stats["n_nodes"] == 2 * g.n_nodes
    assert stats["n_edges"] == 2 * g.n_edges
    assert stats["edges_wrong_dt"] == 0
    assert stats["nodes_with_outdeg_gt2"] == 0

    header = out.read_text().splitlines()[0].split(",")
    assert header == B.SUBMISSION_COLUMNS

def test_validate_submission_rejects_dangling_edge(tmp_path):
    out = tmp_path / "bad.csv"
    out.write_text(
        ",".join(B.SUBMISSION_COLUMNS) + "\n"
        "0,ds,node,1,0,1,1,1,-1,-1\n"
        "1,ds,edge,-1,-1,-1,-1,-1,1,999\n"      # 999 does not exist
    )
    with pytest.raises(AssertionError, match="missing node"):
        B.validate_submission(out)

def test_validate_submission_flags_missing_dataset(tmp_path):
    out = tmp_path / "partial.csv"
    out.write_text(
        ",".join(B.SUBMISSION_COLUMNS) + "\n"
        "0,ds_a,node,1,0,1,1,1,-1,-1\n"
    )
    with pytest.raises(AssertionError, match="missing datasets"):
        B.validate_submission(out, expected_datasets=["ds_a", "ds_b"])

def test_validate_submission_counts_non_consecutive_edges(tmp_path):
    out = tmp_path / "skip.csv"
    out.write_text(
        ",".join(B.SUBMISSION_COLUMNS) + "\n"
        "0,ds,node,1,0,1,1,1,-1,-1\n"
        "1,ds,node,2,5,1,1,1,-1,-1\n"
        "2,ds,edge,-1,-1,-1,-1,-1,1,2\n"        # spans 5 frames
    )
    assert B.validate_submission(out)["edges_wrong_dt"] == 1

# ---------------------------------------------------------------------------
# Zarr v3 reader (the run-2 failure)
# ---------------------------------------------------------------------------

def _write_zarr_v3_array(path: Path, data: np.ndarray, codec: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    codecs = [{"name": "bytes", "configuration": {"endian": "little"}}]
    payload = data.tobytes()
    if codec == "zstd":
        from numcodecs import Zstd
        payload = Zstd().encode(payload)
        codecs.append({"name": "zstd", "configuration": {"level": 1}})
    (path / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "array",
        "shape": list(data.shape), "data_type": str(data.dtype),
        "chunk_grid": {"name": "regular",
                       "configuration": {"chunk_shape": list(data.shape)}},
        "codecs": codecs, "fill_value": 0,
    }))
    chunk = path / "c"
    for _ in data.shape:
        chunk = chunk / "0"
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_bytes(payload)

@pytest.mark.parametrize("codec", ["bytes", "zstd"])
def test_read_zarr_v3_array_roundtrip(tmp_path, codec):
    pytest.importorskip("numcodecs")
    data = np.arange(12, dtype=np.int64).reshape(3, 4)
    _write_zarr_v3_array(tmp_path / "arr", data, codec)
    assert np.array_equal(B._read_zarr_v3_array(tmp_path / "arr"), data)

def test_read_geff_without_the_zarr_package(tmp_path, monkeypatch):
    """The reader must not import `zarr` - Kaggle ships 2.x, which cannot read v3."""
    pytest.importorskip("numcodecs")
    root = tmp_path / "x.geff"
    n = 5
    _write_zarr_v3_array(root / "nodes" / "ids", np.arange(1, n + 1, dtype=np.uint64), "zstd")
    _write_zarr_v3_array(root / "nodes" / "props" / "t" / "values",
                         np.arange(n, dtype=np.int64), "zstd")
    for axis in ("z", "y", "x"):
        _write_zarr_v3_array(root / "nodes" / "props" / axis / "values",
                             np.full(n, 7, dtype=np.int64), "zstd")
    edges = np.stack([np.arange(1, n, dtype=np.uint64),
                      np.arange(2, n + 1, dtype=np.uint64)], axis=1)
    _write_zarr_v3_array(root / "edges" / "ids", edges, "zstd")
    (root / "zarr.json").write_text(json.dumps(
        {"attributes": {"geff": {"extra": {"estimated_number_of_nodes": 1234}}}}))

    # Make any `import zarr` explode, proving the reader never reaches for it.
    real_import = __import__

    def guarded(name, *a, **k):
        if name == "zarr":
            raise ImportError("zarr must not be imported by read_geff")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", guarded)

    g = B.read_geff(root)
    assert g.n_nodes == n
    assert g.n_edges == n - 1
    assert g.meta["estimated_number_of_nodes"] == 1234

# ---------------------------------------------------------------------------
# Notebook generation (the run-1 failure)
# ---------------------------------------------------------------------------

def test_generated_notebooks_inline_the_matching_module():
    """Each %%writefile cell must carry its own module, not another one's."""
    nb_dir = REPO / "notebooks"
    found = 0
    for nb_path in sorted(nb_dir.glob("*.ipynb")):
        nb = json.loads(nb_path.read_text())
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            if not src.startswith("%%writefile "):
                continue
            target = src.split("\n", 1)[0].split()[1]
            expected = REPO / "src" / target
            if not expected.is_file():
                continue
            body = src.split("\n", 1)[1]
            assert body.rstrip("\n") == expected.read_text().rstrip("\n"), \
                f"{nb_path.name} inlines stale/wrong content for {target}"
            found += 1
    assert found >= 4, "expected several inlined library cells"

def test_notebook_code_cells_are_valid_python():
    import ast
    for nb_path in sorted((REPO / "notebooks").glob("*.ipynb")):
        nb = json.loads(nb_path.read_text())
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if src.startswith("%%writefile"):
                src = src.split("\n", 1)[1]
            src = "\n".join(l for l in src.split("\n")
                            if not l.strip().startswith(("%", "!")))
            try:
                ast.parse(src)
            except SyntaxError as e:
                pytest.fail(f"{nb_path.name} cell {i}: {e}")
