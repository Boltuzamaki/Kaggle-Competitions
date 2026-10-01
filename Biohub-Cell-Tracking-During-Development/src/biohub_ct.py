"""Biohub cell tracking during development - core library.

Self-contained: only numpy / scipy are required at inference time.
The same file is inlined into the Kaggle notebooks so a notebook has no
external dataset dependency.

Data model
----------
Images  : OME-Zarr v3, shape (T, Z, Y, X), uint16.
          One blosc2 chunk per timepoint at ``<ds>.zarr/0/c/{t}/0/0/0``.
          Array metadata in ``<ds>.zarr/0/zarr.json``; per-dataset intensity
          quantiles in ``<ds>.zarr/zarr.json`` under ``image_statistics``.
Tracks  : ``<ds>.geff`` - a Zarr v3 group with ``nodes/ids``,
          ``nodes/props/{t,z,y,x}/values`` and ``edges/ids`` (source, target).
          Ground truth is *extremely sparse*: typically a single hand-traced
          lineage (~50 nodes) per video, against ~25k real cells.

Scoring (see metrics.md in royerlab/kaggle-cell-tracking-competition)
--------------------------------------------------------------------
    score = adjusted_edge_jaccard + 0.1 * division_jaccard

    adjusted_edge_jaccard = max(0, J * (1 - 0.1 * (N_pred - N_true) / N_true))

Two consequences drive every design choice below:

1. A predicted edge only counts as a false positive if one of its endpoints
   matches an annotated GT node (within 7 um).  Edges among the ~99.8% of
   unannotated cells are invisible to the edge Jaccard.
2. ``N_pred < N_true`` makes the adjustment factor *greater than one* (up to
   1.1).  Predicting fewer, higher-confidence nodes is rewarded twice: a
   smaller node count and cleaner tracks.  Hence the aggressive track-level
   filtering in :func:`filter_short_tracks`.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment

# (z, y, x) micrometres per voxel - constant across the whole competition.
SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

COMPETITION = "biohub-cell-tracking-during-development"

SUBMISSION_COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
    "source_id", "target_id",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def find_comp_dir(extra: list[str] | None = None) -> Path:
    """Locate the competition data root, on Kaggle or locally."""
    candidates = [
        Path(f"/kaggle/input/competitions/{COMPETITION}"),
        Path(f"/kaggle/input/{COMPETITION}"),
        Path("data/comp"),
        Path("../data/comp"),
    ]
    if extra:
        candidates = [Path(p) for p in extra] + candidates
    for c in candidates:
        if c.is_dir() and ((c / "test").is_dir() or (c / "train").is_dir()):
            return c
    raise FileNotFoundError(
        "Competition data not found. Tried: " + ", ".join(str(c) for c in candidates)
    )

def list_datasets(root: Path | str, split: str = "test") -> list[str]:
    """Dataset base-names (no ``.zarr``) under ``root/split``, sorted."""
    d = Path(root) / split
    if not d.is_dir():
        d = Path(root)
    return sorted(p.name[:-5] for p in d.glob("*.zarr"))

def embryo_of(dataset: str) -> str:
    """Dataset names are ``{embryo}_{field_of_view}``."""
    return dataset.split("_")[0]

# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

_blosc2 = None

def _get_blosc2():
    global _blosc2
    if _blosc2 is None:
        import blosc2
        _blosc2 = blosc2
    return _blosc2

@dataclass
class Volume:
    """A lazily-read (T, Z, Y, X) image volume."""

    path: Path
    shape: tuple[int, int, int, int]
    dtype: np.dtype
    quantiles: dict[str, float] = field(default_factory=dict)

    @property
    def n_t(self) -> int:
        return int(self.shape[0])

    def frame(self, t: int) -> np.ndarray:
        """Return timepoint ``t`` as a (Z, Y, X) array."""
        frame_shape = self.shape[1:]
        chunk = self.path / "0" / "c" / str(t) / "0" / "0" / "0"
        if chunk.is_file():
            try:
                raw = _get_blosc2().decompress(chunk.read_bytes())
                arr = np.frombuffer(raw, dtype=self.dtype)
                if arr.size == int(np.prod(frame_shape)):
                    return arr.reshape(frame_shape).copy()
            except Exception:
                pass
        # Fallback: let zarr deal with it (slower, but always correct).
        import zarr
        return np.asarray(zarr.open(str(self.path / "0"), mode="r")[t])

def open_volume(zarr_path: Path | str) -> Volume:
    zarr_path = Path(zarr_path)
    meta = json.loads((zarr_path / "0" / "zarr.json").read_text())
    shape = tuple(int(s) for s in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    quantiles: dict[str, float] = {}
    root_meta_path = zarr_path / "zarr.json"
    if root_meta_path.is_file():
        try:
            attrs = json.loads(root_meta_path.read_text()).get("attributes", {})
            quantiles = attrs.get("image_statistics", {}).get("quantiles", {}) or {}
        except Exception:
            pass
    return Volume(path=zarr_path, shape=shape, dtype=dtype, quantiles=quantiles)

# ---------------------------------------------------------------------------
# Track graphs
# ---------------------------------------------------------------------------

@dataclass
class TrackGraph:
    """Node coordinates in *voxel* units plus directed edges (by node id)."""

    t: np.ndarray            # (N,) int64
    z: np.ndarray            # (N,) float64
    y: np.ndarray
    x: np.ndarray
    ids: np.ndarray          # (N,) int64
    edges: np.ndarray        # (E, 2) int64 - (source_id, target_id)
    meta: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(len(self.ids))

    @property
    def n_edges(self) -> int:
        return int(len(self.edges))

    def coords(self) -> np.ndarray:
        """(N, 3) voxel coordinates in (z, y, x) order."""
        return np.stack([self.z, self.y, self.x], axis=1)

    def physical(self) -> np.ndarray:
        """(N, 3) coordinates in micrometres."""
        return self.coords() * SCALE

def _decompress(raw: bytes, codecs: list[str]) -> bytes:
    """Decode one Zarr v3 chunk given its codec names.

    Deliberately does not import ``zarr``: Kaggle's image ships zarr 2.x, which
    cannot open a Zarr **v3** store at all, and that failure is what silently
    starved the training run of data. Several zstd backends are tried because
    which one is installed varies between environments.
    """
    if "zstd" in codecs:
        errors = []
        try:
            from numcodecs import Zstd  # ships with zarr 2.x and 3.x
            return Zstd().decode(raw)
        except Exception as e:  # noqa: BLE001
            errors.append(f"numcodecs: {e}")
        try:
            import zstandard
            return zstandard.ZstdDecompressor().decompressobj().decompress(raw)
        except Exception as e:  # noqa: BLE001
            errors.append(f"zstandard: {e}")
        try:
            import pyzstd
            return pyzstd.decompress(raw)
        except Exception as e:  # noqa: BLE001
            errors.append(f"pyzstd: {e}")
        try:
            from compression.zstd import decompress as zstd_decompress  # py3.14+
            return zstd_decompress(raw)
        except Exception as e:  # noqa: BLE001
            errors.append(f"stdlib: {e}")
        raise RuntimeError("no working zstd backend - " + "; ".join(errors))
    if "blosc" in codecs:
        return _get_blosc2().decompress(raw)
    if "gzip" in codecs:
        import gzip
        return gzip.decompress(raw)
    return raw  # 'bytes' codec only

def _read_zarr_v3_array(path: Path) -> np.ndarray:
    """Read a single-chunk Zarr v3 array without the ``zarr`` package.

    Every array inside a ``.geff`` is stored as one chunk, so the chunk key is
    just ``c/0`` (or ``c/0/0`` for the 2-D edge list).
    """
    meta = json.loads((path / "zarr.json").read_text())
    shape = tuple(int(s) for s in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    chunk_shape = tuple(
        int(s) for s in meta["chunk_grid"]["configuration"]["chunk_shape"]
    )
    if any(s > c for s, c in zip(shape, chunk_shape)):
        raise NotImplementedError(f"{path} is multi-chunk; only single-chunk is supported")

    chunk_path = path / "c"
    for _ in shape:
        chunk_path = chunk_path / "0"
    if not chunk_path.is_file():
        # An all-fill-value array writes no chunk at all.
        return np.full(shape, meta.get("fill_value", 0), dtype=dtype)

    codecs = [c.get("name", "") for c in meta.get("codecs", [])]
    buf = _decompress(chunk_path.read_bytes(), codecs)
    return np.frombuffer(buf, dtype=dtype, count=int(np.prod(shape))).reshape(shape)

def read_geff(geff_path: Path | str) -> TrackGraph:
    """Read a ground-truth ``.geff`` directly from its zarr arrays."""
    geff_path = Path(geff_path)

    def arr(rel: str) -> np.ndarray:
        return _read_zarr_v3_array(geff_path / rel)

    ids = arr("nodes/ids").astype(np.int64)
    t = arr("nodes/props/t/values").astype(np.int64)
    z = arr("nodes/props/z/values").astype(np.float64)
    y = arr("nodes/props/y/values").astype(np.float64)
    x = arr("nodes/props/x/values").astype(np.float64)
    edges = arr("edges/ids").astype(np.int64).reshape(-1, 2)

    meta: dict = {}
    try:
        zj = json.loads((geff_path / "zarr.json").read_text())
        geff_meta = zj.get("attributes", {}).get("geff", {}) or {}
        meta = dict(geff_meta)
        extra = geff_meta.get("extra") or {}
        if "estimated_number_of_nodes" in extra:
            meta["estimated_number_of_nodes"] = extra["estimated_number_of_nodes"]
    except Exception:
        pass

    return TrackGraph(t=t, z=z, y=y, x=x, ids=ids, edges=edges, meta=meta)

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def normalize_frame(vol: np.ndarray, quantiles: dict | None = None,
                    lo_q: float = 1.0, hi_q: float = 99.7) -> np.ndarray:
    """Robustly scale a frame to roughly [0, 1].

    Uses the dataset-level quantiles stored in the zarr metadata when present
    (they are computed over the whole movie, so they are stable across frames
    and available for the hidden test set too); otherwise falls back to
    per-frame percentiles.
    """
    vf = vol.astype(np.float32)
    lo = hi = None
    if quantiles:
        try:
            lo = float(quantiles["0.01"])
            hi = float(quantiles["0.99"])
        except (KeyError, TypeError, ValueError):
            lo = hi = None
    if lo is None or hi is None or hi <= lo:
        lo, hi = np.percentile(vf, [lo_q, hi_q])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((vf - lo) / (hi - lo), 0.0, None).astype(np.float32)

def _ball_footprint(radius_um: float, spacing: np.ndarray) -> np.ndarray:
    """Boolean ellipsoid covering ``radius_um`` on a grid with ``spacing`` um."""
    rad = np.maximum(1, np.round(radius_um / spacing).astype(int))
    zz, yy, xx = np.ogrid[-rad[0]:rad[0] + 1, -rad[1]:rad[1] + 1, -rad[2]:rad[2] + 1]
    d2 = ((zz * spacing[0]) ** 2 + (yy * spacing[1]) ** 2 + (xx * spacing[2]) ** 2)
    return d2 <= radius_um ** 2

def detect_dog(vol: np.ndarray,
               quantiles: dict | None = None,
               xy_downsample: int = 4,
               dog_scales: tuple[tuple[float, float], ...] = ((2.0, 6.0), (3.0, 9.0)),
               min_distance_um: float = 5.5,
               rel_threshold: float = 0.02,
               max_peaks: int | None = 4000,
               adaptive_percentile: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Multi-scale Difference-of-Gaussians blob detector.

    Runs on an XY-downsampled grid: with ``xy_downsample=4`` the effective
    spacing becomes (1.625, 1.625, 1.625) um - perfectly isotropic - which
    makes the Gaussian sigmas and the NMS footprint spherical and keeps the
    volume at a cheap 64^3.

    Returns
    -------
    coords : (N, 3) float array of (z, y, x) in *original* voxel units,
             ordered by descending response.
    scores : (N,) DoG response at each peak, same order.
    """
    norm_full = normalize_frame(vol, quantiles)
    norm = norm_full[:, ::xy_downsample, ::xy_downsample]
    spacing = np.array([SCALE[0], SCALE[1] * xy_downsample, SCALE[2] * xy_downsample])

    dog = None
    for small_um, large_um in dog_scales:
        resp = (gaussian_filter(norm, sigma=small_um / spacing)
                - gaussian_filter(norm, sigma=large_um / spacing))
        dog = resp if dog is None else np.maximum(dog, resp)

    footprint = _ball_footprint(min_distance_um, spacing)
    local_max = maximum_filter(dog, footprint=footprint, mode="nearest")

    # A fixed absolute cut on the DoG response does not transfer between
    # datasets: contrast varies enough that the same threshold yielded 50
    # peaks/frame on one test video and 384 on another. Scaling the cut by the
    # frame's own high-percentile response makes it contrast-invariant, so the
    # detector adapts to each video instead of to whichever one it was tuned on.
    threshold = rel_threshold
    if adaptive_percentile is not None:
        positive = dog[dog > 0]
        if positive.size:
            threshold = rel_threshold * float(np.percentile(positive, adaptive_percentile))

    peaks = (dog == local_max) & (dog >= threshold)

    coords = np.argwhere(peaks)
    if coords.size == 0:
        return np.zeros((0, 3)), np.zeros((0,))

    scores = dog[peaks]
    order = np.argsort(scores)[::-1]
    coords, scores = coords[order], scores[order]
    if max_peaks is not None and len(coords) > max_peaks:
        coords, scores = coords[:max_peaks], scores[:max_peaks]

    out = coords.astype(np.float64)
    out[:, 1] *= xy_downsample
    out[:, 2] *= xy_downsample
    return out, scores.astype(np.float64)

def estimate_scale_um(vol_frames: list[np.ndarray], quantiles: dict | None = None,
                      xy_downsample: int = 4,
                      candidates: tuple[float, ...] = (0.9, 1.2, 1.5, 1.9, 2.4, 3.0, 3.8),
                      ratio: float = 3.0, top_k: int = 200) -> float:
    """Pick the DoG inner sigma (µm) that best matches this video's cells.

    Classic scale-space selection (Lindeberg): the *scale-normalised* response
    of a Laplacian-like filter peaks when the filter matches the blob size, so
    sweeping sigma and taking the argmax of the normalised response estimates
    object size without any labels.

    Motivation from the 60-video CV: score tracks cell density hard --
    sparse videos average **0.884** and dense ones **0.656**, with mean cell
    spacing falling from 27 µm to 13 µm. Cells shrink as the embryo divides, so
    one fixed band-pass cannot serve both ends of the range.

    Only a few frames are sampled, so this costs a fraction of a second per
    video and needs no ground truth - it works on the hidden test set.
    """
    spacing = np.array([SCALE[0], SCALE[1] * xy_downsample, SCALE[2] * xy_downsample])
    best_sigma, best_response = candidates[len(candidates) // 2], -np.inf

    for sigma in candidates:
        responses = []
        for raw in vol_frames:
            norm = normalize_frame(raw, quantiles)[:, ::xy_downsample, ::xy_downsample]
            dog = (gaussian_filter(norm, sigma=sigma / spacing)
                   - gaussian_filter(norm, sigma=sigma * ratio / spacing))
            # Scale normalisation makes responses comparable across sigma.
            dog = dog * (sigma ** 2)
            flat = dog.ravel()
            if flat.size > top_k:
                responses.append(float(np.mean(np.partition(flat, -top_k)[-top_k:])))
            elif flat.size:
                responses.append(float(flat.mean()))
        if responses:
            mean_response = float(np.mean(responses))
            if mean_response > best_response:
                best_response, best_sigma = mean_response, sigma
    return float(best_sigma)

def auto_scales(vol, n_frames: int = 4, xy_downsample: int = 4,
                **kwargs) -> tuple[tuple[float, float], ...]:
    """Two DoG scale pairs centred on this video's estimated cell size."""
    n_t = vol.n_t
    idx = np.unique(np.linspace(0, n_t - 1, min(n_frames, n_t)).astype(int))
    frames = [vol.frame(int(t)) for t in idx]
    s = estimate_scale_um(frames, vol.quantiles, xy_downsample=xy_downsample, **kwargs)
    return ((s, s * 3.0), (s * 1.65, s * 4.95))

def adaptive_min_distance(vol, cfg=None, n_frames: int = 4,
                          ratio: float = 0.20,
                          lo: float = 3.0, hi: float = 6.0) -> tuple[float, float]:
    """Scale the NMS radius to this video's own cell spacing.

    Cell density spans **45-670 per frame** across the dataset and the best NMS
    radius tracks it. Measured on 12 videos per density group:

    ======== ============== ============ ========= =========
    group     cells/frame    spacing      best NMS   ratio
    ======== ============== ============ ========= =========
    sparse             55       27.3 µm       5.5      0.201
    mid               106       22.0 µm       4.5      0.205
    dense             521       12.9 µm       3.0      0.232
    ======== ============== ============ ========= =========

    The ratio is essentially constant, so ``radius = 0.20 x spacing`` captures
    the whole relationship with one number. A fixed 3.5 µm leaves **+0.019** on
    the table for sparse videos, which score 0.926 against 0.638 for dense ones
    and so carry real weight in the aggregate.

    Density is estimated from a cheap detection pass on a few frames. That
    observable tracks the ground-truth count with **r = 0.999**, so the rule
    works on the hidden test set where `estimated_number_of_nodes` is absent.

    Returns ``(min_distance_um, observed_peaks_per_frame)``.
    """
    cfg = cfg if cfg is not None else Config()
    n_t = vol.n_t
    idx = np.unique(np.linspace(0, n_t - 1, min(n_frames, n_t)).astype(int))

    counts = []
    for t in idx:
        raw = vol.frame(int(t))
        coords, _ = detect_dog(raw, vol.quantiles, cfg.xy_downsample, cfg.dog_scales,
                               cfg.min_distance_um, cfg.rel_threshold, cfg.max_peaks,
                               adaptive_percentile=cfg.adaptive_percentile)
        counts.append(len(coords))
    density = float(np.mean(counts)) if counts else 0.0
    if density <= 0:
        return cfg.min_distance_um, density

    volume_um3 = float(np.prod(np.array(vol.shape[1:], dtype=np.float64) * SCALE))
    spacing_um = (volume_um3 / density) ** (1.0 / 3.0)
    return float(np.clip(ratio * spacing_um, lo, hi)), density

def merge_detections(coord_sets: list[np.ndarray], score_sets: list[np.ndarray],
                     min_distance_um: float = 3.5,
                     max_peaks: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Union several detection sets, then greedy non-maximum suppression in µm.

    Scores from different detectors are not comparable in absolute terms, so
    each set is rank-normalised to (0, 1] before merging; the greedy pass then
    keeps the globally best-ranked candidate and drops anything within
    ``min_distance_um`` of an already-kept point.
    """
    if not coord_sets:
        return np.zeros((0, 3)), np.zeros((0,))

    coords, scores = [], []
    for c, s in zip(coord_sets, score_sets):
        c = np.asarray(c, dtype=np.float64)
        if len(c) == 0:
            continue
        n = len(c)
        order = np.argsort(np.asarray(s, dtype=np.float64))[::-1]
        rank = np.empty(n)
        rank[order] = np.linspace(1.0, 1.0 / n, n)   # best -> 1.0
        coords.append(c)
        scores.append(rank)
    if not coords:
        return np.zeros((0, 3)), np.zeros((0,))

    coords = np.concatenate(coords, axis=0)
    scores = np.concatenate(scores, axis=0)

    order = np.argsort(scores)[::-1]
    coords, scores = coords[order], scores[order]
    phys = coords * SCALE

    keep: list[int] = []
    kept_phys = np.zeros((0, 3))
    r2 = float(min_distance_um) ** 2
    for i in range(len(coords)):
        if kept_phys.shape[0]:
            if ((kept_phys - phys[i]) ** 2).sum(axis=1).min() < r2:
                continue
        keep.append(i)
        kept_phys = np.vstack([kept_phys, phys[i]])
        if max_peaks is not None and len(keep) >= max_peaks:
            break
    idx = np.array(keep, dtype=int)
    return coords[idx], scores[idx]

def detect_dog_union(vol: np.ndarray, quantiles: dict | None = None,
                     xy_downsample: int = 4,
                     scale_sets: tuple[tuple[tuple[float, float], ...], ...] = (
                         ((1.5, 4.5),), ((2.5, 7.5),)),
                     min_distance_um: float = 3.5,
                     rel_threshold: float = 0.02,
                     max_peaks: int | None = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Run the DoG detector once per scale set and union the results.

    :func:`detect_dog` takes an element-wise max over its scales *before* NMS,
    so two cells resolved at different scales that sit closer than the NMS
    radius collapse into one peak. Detecting per scale and merging afterwards
    keeps both.

     **Measured, and it does NOT help - do not reach for this without a new
    idea.** On the two scorable videos it raises node recall substantially but
    lowers the actual score:

    ======================================  ========  ========
    configuration                            recall    adj
    ======================================  ========  ========
    baseline max-over-scales                   0.902    0.7395
    union of 2 scale sets                      0.902    0.6516
    union of 3 scale sets                      0.990    0.5383
    union of 3, capped at 400/frame            0.931    0.6172
    ======================================  ========  ========

    Tightening the linking gate does not rescue it (4/5/6/7 µm give
    0.456/0.545/0.601/0.617). The extra candidates confuse the assignment more
    than the recovered nodes gain, and the node-count penalty compounds it.

    The lesson: **detection recall is not the objective, edge Jaccard is.** What
    is needed is a detector that *ranks* true cells above spurious ones, not one
    that proposes more of both - i.e. the learned detector, not a wider
    band-pass. Kept here as a documented experiment and for use with a scorer
    that can rank candidates properly.
    """
    coord_sets, score_sets = [], []
    for scales in scale_sets:
        c, s = detect_dog(vol, quantiles, xy_downsample=xy_downsample,
                          dog_scales=scales, min_distance_um=min_distance_um,
                          rel_threshold=rel_threshold, max_peaks=max_peaks)
        coord_sets.append(c)
        score_sets.append(s)
    return merge_detections(coord_sets, score_sets,
                            min_distance_um=min_distance_um, max_peaks=max_peaks)

def refine_centroids(vol: np.ndarray, coords: np.ndarray,
                     win: tuple[int, int, int] = (1, 4, 4),
                     baseline_percentile: float = 25.0,
                     max_shift_um: float = 2.5) -> np.ndarray:
    """Background-subtracted intensity centroid refinement at full resolution.

    Peaks found on the downsampled grid land on a coarse lattice; this recovers
    sub-voxel placement, which matters because node matching uses a hard 7 um
    radius.  Shifts larger than ``max_shift_um`` are rejected as unstable.
    """
    if len(coords) == 0:
        return coords
    Z, Y, X = vol.shape
    out = coords.astype(np.float64).copy()
    wz, wy, wx = win
    for i, original in enumerate(coords):
        z, y, x = (int(round(v)) for v in original)
        z0, z1 = max(0, z - wz), min(Z, z + wz + 1)
        y0, y1 = max(0, y - wy), min(Y, y + wy + 1)
        x0, x1 = max(0, x - wx), min(X, x + wx + 1)
        patch = vol[z0:z1, y0:y1, x0:x1].astype(np.float64)
        if patch.size == 0:
            continue
        weights = np.maximum(patch - np.percentile(patch, baseline_percentile), 0.0)
        total = float(weights.sum())
        if total <= 0:
            continue
        zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
        yy = np.arange(y0, y1, dtype=np.float64)[None, :, None]
        xx = np.arange(x0, x1, dtype=np.float64)[None, None, :]
        refined = np.array([
            float((weights * zz).sum() / total),
            float((weights * yy).sum() / total),
            float((weights * xx).sum() / total),
        ])
        if np.sqrt((((refined - original) * SCALE) ** 2).sum()) <= max_shift_um:
            out[i] = refined
    return out

# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------

def _assign(cost: np.ndarray, dist: np.ndarray, max_um: float):
    """Hungarian assignment restricted to pairs within ``max_um``."""
    if cost.size == 0:
        return []
    big = float(max_um) * 1000.0 + 1.0
    ri, ci = linear_sum_assignment(np.where(dist <= max_um, cost, big))
    return [(int(r), int(c)) for r, c in zip(ri, ci) if dist[r, c] <= max_um]

def link_motion(frames: list[np.ndarray],
                max_link_um: float = 8.0,
                motion_weight: float = 0.6,
                min_margin_um: float = 0.0) -> TrackGraph:
    """Velocity-aware frame-to-frame linking.

    Each live track predicts where it should be at ``t+1`` from its last
    displacement; the assignment cost is the distance to that prediction while
    the *gate* stays on the true distance.  This resolves the dense-region
    swaps that plain nearest-neighbour linking makes.

    ``frames[t]`` is an (M_t, 3) array of (z, y, x) voxel coordinates.
    """
    node_t: list[int] = []
    node_zyx: list[np.ndarray] = []
    frame_ids: list[list[int]] = []
    next_id = 1
    for t, coords in enumerate(frames):
        ids_t = []
        for c in coords:
            node_t.append(t)
            node_zyx.append(np.asarray(c, dtype=np.float64))
            ids_t.append(next_id)
            next_id += 1
        frame_ids.append(ids_t)

    edges: list[tuple[int, int]] = []
    # velocity[node_id] -> last physical displacement (um)
    velocity: dict[int, np.ndarray] = {}

    for t in range(len(frames) - 1):
        a, b = frames[t], frames[t + 1]
        if len(a) == 0 or len(b) == 0:
            continue
        ap = np.asarray(a, dtype=np.float64) * SCALE
        bp = np.asarray(b, dtype=np.float64) * SCALE

        pred = ap.copy()
        for i, nid in enumerate(frame_ids[t]):
            v = velocity.get(nid)
            if v is not None:
                pred[i] = ap[i] + motion_weight * v

        dist = np.sqrt(((ap[:, None, :] - bp[None, :, :]) ** 2).sum(axis=2))
        cost = np.sqrt(((pred[:, None, :] - bp[None, :, :]) ** 2).sum(axis=2))

        # Ambiguity guard. When the runner-up target is nearly as close as the
        # chosen one, the assignment is close to a coin flip; a wrong guess
        # costs a false positive *and* a false negative, while declining leaves
        # a one-frame hole that close_gaps can repair for free. The margin to
        # the runner-up was the single most informative feature in the learned
        # edge-scoring experiment, so this captures that signal without a model.
        second = None
        if min_margin_um > 0 and cost.shape[1] > 1:
            part = np.partition(cost, 1, axis=1)
            second = part[:, 1]

        for r, c in _assign(cost, dist, max_link_um):
            if second is not None and (second[r] - cost[r, c]) < min_margin_um:
                continue
            src, tgt = frame_ids[t][r], frame_ids[t + 1][c]
            edges.append((src, tgt))
            velocity[tgt] = bp[c] - ap[r]

    zyx = (np.stack(node_zyx) if node_zyx else np.zeros((0, 3)))
    return TrackGraph(
        t=np.array(node_t, dtype=np.int64),
        z=zyx[:, 0], y=zyx[:, 1], x=zyx[:, 2],
        ids=np.arange(1, len(node_t) + 1, dtype=np.int64),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
    )

def link_harmonic(frames: list[np.ndarray],
                  max_link_um: float = 7.0,
                  motion_weight: float = 0.6,
                  bidir_weight: float = 0.15,
                  temperature_um: float = 2.0) -> TrackGraph:
    """Linking with harmonic forward/reverse fusion of association scores.

    Adapted from the technique the public 0.93+ notebooks converge on. Their
    version fuses a learned edge model's logits; the principle transfers to a
    distance cost and needs no training.

    For each consecutive pair of frames, the motion-compensated distance matrix
    is turned into two competing distributions:

    * **forward**  - softmax over *targets* for each source ("where did this
      cell go?")
    * **reverse**  - softmax over *sources* for each target ("where did this
      cell come from?")

    They are combined with a weighted **harmonic mean in probability space**::

        p = 1 / ((1 - w) / p_forward + w / p_reverse)

    The harmonic mean is dominated by the *smaller* term, so a pair is demoted
    unless **both** directions support it. That is exactly the failure this
    pipeline has: with denser candidates, wrong links rose from 3 to 26 while
    detection misses barely moved, because a one-directional argmin is happy to
    claim a target that some other source owns far more strongly.

    ``temperature_um`` sets how sharply distance maps to probability; at 2.0 µm
    a pair 2 µm closer than its rival is ~e times more probable, which matches
    the observed ~1.7 µm median per-frame displacement.
    """
    node_t: list[int] = []
    node_zyx: list[np.ndarray] = []
    frame_ids: list[list[int]] = []
    next_id = 1
    for t, coords in enumerate(frames):
        ids_t = []
        for c in coords:
            node_t.append(t)
            node_zyx.append(np.asarray(c, dtype=np.float64))
            ids_t.append(next_id)
            next_id += 1
        frame_ids.append(ids_t)

    edges: list[tuple[int, int]] = []
    velocity: dict[int, np.ndarray] = {}

    for t in range(len(frames) - 1):
        a, b = frames[t], frames[t + 1]
        if len(a) == 0 or len(b) == 0:
            continue
        ap = np.asarray(a, dtype=np.float64) * SCALE
        bp = np.asarray(b, dtype=np.float64) * SCALE

        pred = ap.copy()
        for i, nid in enumerate(frame_ids[t]):
            v = velocity.get(nid)
            if v is not None:
                pred[i] = ap[i] + motion_weight * v

        dist = np.sqrt(((ap[:, None, :] - bp[None, :, :]) ** 2).sum(axis=2))
        motion_cost = np.sqrt(((pred[:, None, :] - bp[None, :, :]) ** 2).sum(axis=2))

        logits = -motion_cost / max(temperature_um, 1e-6)
        # Softmax along each axis in turn; subtract the max for stability.
        fwd = np.exp(logits - logits.max(axis=1, keepdims=True))
        fwd /= np.clip(fwd.sum(axis=1, keepdims=True), 1e-12, None)
        rev = np.exp(logits - logits.max(axis=0, keepdims=True))
        rev /= np.clip(rev.sum(axis=0, keepdims=True), 1e-12, None)

        fwd = np.clip(fwd, 1e-12, None)
        rev = np.clip(rev, 1e-12, None)
        harmonic = 1.0 / ((1.0 - bidir_weight) / fwd + bidir_weight / rev)
        cost = -np.log(np.clip(harmonic, 1e-300, None))

        for r, c in _assign(cost, dist, max_link_um):
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

def close_gaps(g: TrackGraph, max_gap: int = 1, gap_um_per_frame: float = 6.0) -> TrackGraph:
    """Bridge short detection dropouts by inserting interpolated nodes.

    A track that ends at ``t`` and one that starts at ``t + gap + 1`` are joined
    through ``gap`` synthetic nodes on the straight line between them.  Since
    every GT edge spans exactly one frame, bridging a single missed frame turns
    zero matchable edges into two.
    """
    if g.n_edges == 0 or g.n_nodes == 0:
        return g

    coord = {int(i): (int(tt), zz, yy, xx)
             for i, tt, zz, yy, xx in zip(g.ids, g.t, g.z, g.y, g.x)}
    has_out = {int(s) for s in g.edges[:, 0]}
    has_in = {int(d) for d in g.edges[:, 1]}

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    for nid, (tt, *_rest) in coord.items():
        if nid not in has_out:
            ends_by_t.setdefault(tt, []).append(nid)
        if nid not in has_in:
            starts_by_t.setdefault(tt, []).append(nid)

    new_nodes: list[tuple[int, float, float, float, int]] = []
    new_edges: list[tuple[int, int]] = []
    next_id = int(g.ids.max()) + 1

    for gap in range(1, max_gap + 1):
        for tt, ends in sorted(ends_by_t.items()):
            ends = [e for e in ends if e not in has_out]
            starts = [s for s in starts_by_t.get(tt + gap + 1, []) if s not in has_in]
            if not ends or not starts:
                continue
            ec = np.array([coord[e][1:] for e in ends], dtype=np.float64) * SCALE
            sc = np.array([coord[s][1:] for s in starts], dtype=np.float64) * SCALE
            dist = np.sqrt(((ec[:, None, :] - sc[None, :, :]) ** 2).sum(axis=2))
            threshold = gap_um_per_frame * (gap + 1)
            for r, c in _assign(dist, dist, threshold):
                e_id, s_id = ends[r], starts[c]
                if e_id in has_out or s_id in has_in:
                    continue
                te, ze, ye, xe = coord[e_id]
                _ts, zs, ys, xs = coord[s_id]
                prev = e_id
                for k in range(1, gap + 1):
                    frac = k / (gap + 1)
                    nid = next_id
                    next_id += 1
                    new_nodes.append((
                        te + k,
                        ze + (zs - ze) * frac,
                        ye + (ys - ye) * frac,
                        xe + (xs - xe) * frac,
                        nid,
                    ))
                    new_edges.append((prev, nid))
                    prev = nid
                new_edges.append((prev, s_id))
                has_out.add(e_id)
                has_in.add(s_id)

    if not new_nodes:
        return g
    return TrackGraph(
        t=np.concatenate([g.t, np.array([n[0] for n in new_nodes], dtype=np.int64)]),
        z=np.concatenate([g.z, np.array([n[1] for n in new_nodes])]),
        y=np.concatenate([g.y, np.array([n[2] for n in new_nodes])]),
        x=np.concatenate([g.x, np.array([n[3] for n in new_nodes])]),
        ids=np.concatenate([g.ids, np.array([n[4] for n in new_nodes], dtype=np.int64)]),
        edges=np.concatenate([g.edges, np.array(new_edges, dtype=np.int64).reshape(-1, 2)]),
        meta=g.meta,
    )

def smooth_tracks(g: TrackGraph, weight: float = 0.8, window: int = 2) -> TrackGraph:
    """Pull each node toward a local straight-line fit of its own track.

    Every strong public notebook runs some form of this (``OUTPUT_LINEFIT_SMOOTH``,
    typically weight 0.8 / window 2). The rationale fits this metric well: node
    matching uses a hard 7 um radius, real motion is smooth at ~1.7 um/frame, so
    per-frame detection jitter is almost pure noise. Fitting a line over +/-``window``
    frames and moving each node ``weight`` of the way toward it removes jitter
    without flattening genuine curvature.

    Nodes with fewer than three neighbours in their own track are left alone, and
    smoothing runs on the *chain* through each node, so a division's two branches
    are smoothed separately rather than being averaged together.
    """
    if g.n_nodes == 0 or g.n_edges == 0 or weight <= 0:
        return g

    idx = {int(nid): i for i, nid in enumerate(g.ids)}
    succ: dict[int, list[int]] = {}
    pred: dict[int, list[int]] = {}
    for a, b in g.edges:
        succ.setdefault(int(a), []).append(int(b))
        pred.setdefault(int(b), []).append(int(a))

    coords = g.coords().astype(np.float64)
    out = coords.copy()

    for nid in g.ids:
        nid = int(nid)
        chain = [nid]
        # Walk backwards then forwards, stopping at branch points so the two
        # daughters of a division are never mixed into one fit.
        cur = nid
        for _ in range(window):
            ps = pred.get(cur, [])
            if len(ps) != 1 or len(succ.get(ps[0], [])) != 1:
                break
            cur = ps[0]
            chain.insert(0, cur)
        cur = nid
        for _ in range(window):
            ss = succ.get(cur, [])
            if len(ss) != 1:
                break
            cur = ss[0]
            chain.append(cur)
        if len(chain) < 3:
            continue

        rows = [idx[c] for c in chain if c in idx]
        if len(rows) < 3:
            continue
        ts = g.t[rows].astype(np.float64)
        if ts.max() == ts.min():
            continue
        pts = coords[rows]
        # Independent least-squares line in each axis against time.
        a1 = np.polyfit(ts, pts, 1)          # (2, 3): slope and intercept per axis
        fitted = np.polyval(a1, float(g.t[idx[nid]]))
        out[idx[nid]] = (1 - weight) * coords[idx[nid]] + weight * fitted

    return TrackGraph(t=g.t, z=out[:, 0], y=out[:, 1], x=out[:, 2],
                      ids=g.ids, edges=g.edges, meta=g.meta)

def _components(g: TrackGraph) -> dict[int, list[int]]:
    """Weakly-connected components as ``{root_id: [node_ids]}``."""
    parent = {int(i): int(i) for i in g.ids}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, d in g.edges:
        ra, rb = find(int(s)), find(int(d))
        if ra != rb:
            parent[rb] = ra

    groups: dict[int, list[int]] = {}
    for nid in parent:
        groups.setdefault(find(nid), []).append(nid)
    return groups

def filter_short_tracks(g: TrackGraph, min_len: int = 6) -> TrackGraph:
    """Drop weakly-connected components shorter than ``min_len`` nodes.

    This is the single most valuable post-processing step.  Short components
    are overwhelmingly spurious detections, and removing them helps the score
    twice over: the edge Jaccard gets cleaner, and the node count drops, which
    *increases* the adjustment factor ``1 - 0.1 * (N_pred - N_true)/N_true``.
    """
    if g.n_nodes == 0:
        return g
    keep_ids: set[int] = set()
    for members in _components(g).values():
        if len(members) >= min_len:
            keep_ids.update(members)
    return subset(g, keep_ids)

def subset(g: TrackGraph, keep_ids: set[int]) -> TrackGraph:
    """Restrict a graph to ``keep_ids`` (and the edges fully inside it)."""
    mask = np.array([int(i) in keep_ids for i in g.ids], dtype=bool)
    if g.n_edges:
        e_mask = np.array(
            [int(s) in keep_ids and int(d) in keep_ids for s, d in g.edges], dtype=bool
        )
        edges = g.edges[e_mask]
    else:
        edges = g.edges
    return TrackGraph(t=g.t[mask], z=g.z[mask], y=g.y[mask], x=g.x[mask],
                      ids=g.ids[mask], edges=edges, meta=g.meta)

def add_safe_divisions(g: TrackGraph,
                       max_parent_um: float = 4.7,
                       max_sister_um: float = 7.5,
                       frame_frac_cap: float = 0.008) -> TrackGraph:
    """Attach a second daughter to childless parents, very conservatively.

    The division Jaccard is only worth 0.1 of the score and a wrong fork can
    cost an edge TP, so this only fires when an orphan node at ``t+1`` sits
    very close to the parent *and* close to the parent's existing child.
    """
    if g.n_edges == 0:
        return g

    idx = {int(nid): i for i, nid in enumerate(g.ids)}
    phys = g.physical()
    out_deg: dict[int, list[int]] = {}
    has_in: set[int] = set()
    for s, d in g.edges:
        out_deg.setdefault(int(s), []).append(int(d))
        has_in.add(int(d))

    by_t: dict[int, list[int]] = {}
    for nid, tt in zip(g.ids, g.t):
        by_t.setdefault(int(tt), []).append(int(nid))

    new_edges: list[tuple[int, int]] = []
    for tt in sorted(by_t):
        parents = [p for p in by_t.get(tt, []) if len(out_deg.get(p, [])) == 1]
        orphans = [o for o in by_t.get(tt + 1, []) if o not in has_in]
        if not parents or not orphans:
            continue
        cap = max(1, int(frame_frac_cap * len(by_t.get(tt, []))))
        o_phys = phys[[idx[o] for o in orphans]]
        added = 0
        used: set[int] = set()
        for p in parents:
            if added >= cap:
                break
            p_phys = phys[idx[p]]
            child = out_deg[p][0]
            c_phys = phys[idx[child]]
            d_parent = np.sqrt(((o_phys - p_phys) ** 2).sum(axis=1))
            d_sister = np.sqrt(((o_phys - c_phys) ** 2).sum(axis=1))
            ok = (d_parent <= max_parent_um) & (d_sister <= max_sister_um)
            for j in np.argsort(d_parent):
                if not ok[j] or orphans[j] in used:
                    continue
                new_edges.append((p, orphans[j]))
                used.add(orphans[j])
                has_in.add(orphans[j])
                added += 1
                break

    if not new_edges:
        return g
    return TrackGraph(
        t=g.t, z=g.z, y=g.y, x=g.x, ids=g.ids,
        edges=np.concatenate([g.edges, np.array(new_edges, dtype=np.int64).reshape(-1, 2)]),
        meta=g.meta,
    )

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def probe_heatmap(hm: np.ndarray, coord_vox: np.ndarray, search_um: float = 4.0,
                  xy_downsample: int = 4) -> tuple[np.ndarray, float]:
    """Find the strongest heatmap peak near a predicted position.

    ``hm`` is on the downsampled isotropic grid; ``coord_vox`` is in original
    voxel units. Returns ``(position_in_original_voxels, response)``; the
    response is 0.0 when the search box falls outside the volume.
    """
    spacing = np.array([SCALE[0], SCALE[1] * xy_downsample, SCALE[2] * xy_downsample])
    c = np.asarray(coord_vox, dtype=np.float64).copy()
    c[1] /= xy_downsample
    c[2] /= xy_downsample

    rad = np.maximum(1, np.ceil(search_um / spacing).astype(int))
    lo = np.maximum(0, np.round(c).astype(int) - rad)
    hi = np.minimum(np.array(hm.shape), np.round(c).astype(int) + rad + 1)
    if np.any(lo >= hi):
        return np.asarray(coord_vox, dtype=np.float64), 0.0

    box = np.asarray(hm[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.float32)
    if box.size == 0:
        return np.asarray(coord_vox, dtype=np.float64), 0.0

    k = np.unravel_index(int(np.argmax(box)), box.shape)
    peak = np.array([lo[0] + k[0], lo[1] + k[1], lo[2] + k[2]], dtype=np.float64)
    out = peak.copy()
    out[1] *= xy_downsample
    out[2] *= xy_downsample
    return out, float(box[k])

def repair_tracks_with_evidence(g: TrackGraph, heatmaps: list[np.ndarray],
                                max_gap: int = 2,
                                gap_um_per_frame: float = 6.0,
                                search_um: float = 4.0,
                                min_response: float = 0.30,
                                motion_weight: float = 1.0,
                                xy_downsample: int = 4) -> TrackGraph:
    """Close gaps using *evidence from the detector*, not blind interpolation.

    This is the ITEC idea - let tracking improve detection rather than merely
    consume it. ``close_gaps`` bridges a dropout by inserting a node on the
    straight line between the two track ends, which is a guess. Here the
    trajectory instead *predicts* where the cell should be, the U-Net heatmap is
    inspected around that prediction, and a node is inserted at the real local
    peak when one exists above ``min_response``.

    The detector already found these cells - they were simply ranked below the
    per-frame top-K budget. Recovering them costs nothing in the node-count
    penalty relative to interpolation, but places the node on actual image
    evidence rather than on a chord, which matters against a 7 um matching gate.

    Falls back to the interpolated position when no peak clears the threshold,
    so this can only be better-informed than :func:`close_gaps`, never blinder.
    """
    if g.n_edges == 0 or g.n_nodes == 0 or not heatmaps:
        return g

    coord = {int(i): np.array([float(zz), float(yy), float(xx)])
             for i, zz, yy, xx in zip(g.ids, g.z, g.y, g.x)}
    time_of = {int(i): int(tt) for i, tt in zip(g.ids, g.t)}
    has_out = {int(s_) for s_ in g.edges[:, 0]}
    has_in = {int(d_) for d_ in g.edges[:, 1]}

    # Last displacement of each track end, for the motion prediction.
    prev_of: dict[int, int] = {}
    for a, b in g.edges:
        prev_of[int(b)] = int(a)

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    for nid, tt in time_of.items():
        if nid not in has_out:
            ends_by_t.setdefault(tt, []).append(nid)
        if nid not in has_in:
            starts_by_t.setdefault(tt, []).append(nid)

    new_nodes: list[tuple[int, float, float, float, int]] = []
    new_edges: list[tuple[int, int]] = []
    next_id = int(g.ids.max()) + 1
    n_evidence = n_interp = 0

    for gap in range(1, max_gap + 1):
        for tt in sorted(ends_by_t):
            ends = [e for e in ends_by_t[tt] if e not in has_out]
            starts = [s_ for s_ in starts_by_t.get(tt + gap + 1, []) if s_ not in has_in]
            if not ends or not starts:
                continue
            ec = np.array([coord[e] for e in ends]) * SCALE
            sc = np.array([coord[s_] for s_ in starts]) * SCALE
            dist = np.sqrt(((ec[:, None, :] - sc[None, :, :]) ** 2).sum(axis=2))
            threshold = gap_um_per_frame * (gap + 1)

            for r, c in _assign(dist, dist, threshold):
                e_id, s_id = ends[r], starts[c]
                if e_id in has_out or s_id in has_in:
                    continue
                p_end, p_start = coord[e_id], coord[s_id]

                # Velocity of the ending track, if it has any history.
                vel = np.zeros(3)
                pv = prev_of.get(e_id)
                if pv is not None and motion_weight > 0:
                    vel = (p_end - coord[pv]) * motion_weight

                prev_node = e_id
                for k in range(1, gap + 1):
                    frac = k / (gap + 1)
                    chord = p_end + (p_start - p_end) * frac
                    predicted = 0.5 * chord + 0.5 * (p_end + vel * k)

                    t_k = tt + k
                    pos, resp = (chord, 0.0)
                    if 0 <= t_k < len(heatmaps):
                        cand, resp = probe_heatmap(heatmaps[t_k], predicted,
                                                   search_um, xy_downsample)
                        if resp >= min_response:
                            pos = cand
                            n_evidence += 1
                        else:
                            pos = chord
                            n_interp += 1
                    nid = next_id
                    next_id += 1
                    new_nodes.append((t_k, pos[0], pos[1], pos[2], nid))
                    new_edges.append((prev_node, nid))
                    prev_node = nid
                new_edges.append((prev_node, s_id))
                has_out.add(e_id)
                has_in.add(s_id)

    if not new_nodes:
        return g
    g_out = TrackGraph(
        t=np.concatenate([g.t, np.array([n[0] for n in new_nodes], dtype=np.int64)]),
        z=np.concatenate([g.z, np.array([n[1] for n in new_nodes])]),
        y=np.concatenate([g.y, np.array([n[2] for n in new_nodes])]),
        x=np.concatenate([g.x, np.array([n[3] for n in new_nodes])]),
        ids=np.concatenate([g.ids, np.array([n[4] for n in new_nodes], dtype=np.int64)]),
        edges=np.concatenate([g.edges, np.array(new_edges, dtype=np.int64).reshape(-1, 2)]),
        meta=dict(g.meta),
    )
    g_out.meta["repair_from_evidence"] = n_evidence
    g_out.meta["repair_interpolated"] = n_interp
    return g_out

def build_graph_iterative(frames: list[np.ndarray], cfg: Config,
                          heatmaps: list[np.ndarray] | None = None,
                          rounds: int = 2) -> TrackGraph:
    """Link, repair from image evidence, then re-link on the repaired node set.

    The ITEC premise: tracking should *improve* detection, not just consume it.
    A recovered node is fed back into the candidate pool so the next linking
    round sees it as an ordinary detection, which lets a bridged track keep
    growing instead of ending at the patch.
    """
    work = [np.asarray(f, dtype=np.float64).copy() for f in frames]
    g = None
    for _ in range(max(1, rounds)):
        if cfg.bidir_weight > 0:
            g = link_harmonic(work, cfg.max_link_um, cfg.motion_weight,
                              cfg.bidir_weight, cfg.bidir_temperature_um)
        else:
            g = link_motion(work, cfg.max_link_um, cfg.motion_weight, cfg.min_margin_um)

        if heatmaps:
            g = repair_tracks_with_evidence(
                g, heatmaps, max_gap=cfg.max_gap,
                gap_um_per_frame=cfg.gap_um_per_frame,
                search_um=cfg.repair_search_um,
                min_response=cfg.repair_min_response,
                xy_downsample=cfg.xy_downsample)
        elif cfg.max_gap > 0:
            g = close_gaps(g, cfg.max_gap, cfg.gap_um_per_frame)

        # Feed recovered nodes back into the candidate pool for the next round.
        work = [[] for _ in range(len(frames))]
        for tt, zz, yy, xx in zip(g.t, g.z, g.y, g.x):
            work[int(tt)].append([zz, yy, xx])
        work = [np.array(v, dtype=np.float64).reshape(-1, 3) for v in work]

    if cfg.min_track_len > 1:
        g = filter_short_tracks(g, cfg.min_track_len)
    if cfg.smooth_weight > 0:
        g = smooth_tracks(g, cfg.smooth_weight, cfg.smooth_window)
    if cfg.safe_divisions:
        g = add_safe_divisions(g, cfg.div_parent_um, cfg.div_sister_um,
                               cfg.div_frame_frac_cap)
    return g

@dataclass
class Config:
    """Every knob of the classical pipeline, in one place."""

    # 4 makes the grid isotropic at 1.625 um; measured better than 2, which is
    # finer in XY but leaves the anisotropy that distorts the DoG kernels.
    xy_downsample: int = 4
    # Small scales: the nuclei are ~5-8 um across, so a (1.5, 4.5) um band-pass
    # sits on them. Larger kernels merge neighbours and halve node recall.
    dog_scales: tuple[tuple[float, float], ...] = ((1.5, 4.5), (2.5, 7.5))
    # When True, dog_scales is replaced per video by auto_scales(), which sizes
    # the band-pass to that video's own cells. Cell density spans 45-670 per
    # frame across the dataset and score tracks it hard (sparse 0.884, dense
    # 0.656 on the 60-video CV), so a fixed band-pass cannot fit both ends.
    auto_scale: bool = False
    # Scale the NMS radius to each video's own cell spacing (radius = 0.20 x
    # spacing, clipped). Measured worth +0.019 on sparse videos and +0.006 on
    # dense ones; see adaptive_min_distance for the numbers.
    adaptive_nms: bool = False
    min_distance_um: float = 3.5
    # Best of {0.01, 0.02, 0.04} in a 162-config sweep. The effect is small
    # (0.733 vs 0.706/0.714) and measured on only two videos - unlike
    # min_distance_um, whose 3.5-vs-4.5/5.5 advantage is large and consistent.
    rel_threshold: float = 0.02
    # None = absolute cut. A value like 99.5 makes rel_threshold a fraction of
    # the frame's own DoG response at that percentile (contrast-invariant).
    adaptive_percentile: float | None = None
    max_peaks: int | None = 4000
    refine: bool = True
    # 7.0 um is the p99 of the annotated frame-to-frame displacement, so the
    # gate admits essentially every real move without inviting identity swaps.
    max_link_um: float = 7.0
    motion_weight: float = 0.6
    # Refuse a link when the runner-up target is within this margin (um) of the
    # chosen one. 0 disables. See link_motion for the rationale.
    min_margin_um: float = 0.0
    # Harmonic forward/reverse association fusion (see link_harmonic).
    # Measured +0.0006 on 30 videos - noise, so off. The public notebooks fuse a
    # *learned* edge model's logits, which carry appearance information; fusing a
    # symmetric distance matrix adds almost nothing the forward pass did not have.
    bidir_weight: float = 0.0
    bidir_temperature_um: float = 2.0
    max_gap: int = 1
    gap_um_per_frame: float = 6.0
    # Long enough to remove detector noise, short enough not to delete a real
    # lineage that the detector only picked up intermittently.
    min_track_len: int = 4
    # Off by default. Measured on the two scorable videos it produced 0 division
    # TPs and 1 FP, costing 0.0066. That is not just a noise-level fit: only 5
    # divisions are annotated across 65 training videos, so a geometric rule has
    # almost no real forks to find and every mistake is a wasted edge. Revisit
    # with a learned division classifier, not with tighter radii.
    safe_divisions: bool = False
    # Local line-fit smoothing of node positions (see smooth_tracks).
    # Measured +0.0135 on a 30-video CV - the largest post-processing gain
    # since gap closing, and well clear of the ~0.005 noise floor.
    smooth_weight: float = 0.8
    smooth_window: int = 3
    # Division geometry. The earlier narrow radii (4.7 / 7.5) produced only
    # false positives; public 0.93+ notebooks use much wider gates (7.0 / 12.0).
    div_parent_um: float = 4.7
    div_sister_um: float = 7.5
    div_frame_frac_cap: float = 0.008
    # ITEC-style evidence-based repair (see repair_tracks_with_evidence).
    repair_search_um: float = 4.0
    repair_min_response: float = 0.30
    repair_rounds: int = 2
    t_limit: int | None = None

def detect_frames(vol: Volume, cfg: Config, t_limit: int | None = None,
                  progress=None) -> list[np.ndarray]:
    """Run detection on every timepoint; returns per-frame (M, 3) voxel coords."""
    scales = cfg.dog_scales
    if cfg.auto_scale:
        scales = auto_scales(vol, xy_downsample=cfg.xy_downsample)
    min_dist = cfg.min_distance_um
    if cfg.adaptive_nms:
        min_dist, _density = adaptive_min_distance(vol, cfg)
    n_t = vol.n_t if t_limit is None else min(vol.n_t, t_limit)
    frames = []
    rng = range(n_t)
    if progress is not None:
        rng = progress(rng)
    for t in rng:
        raw = vol.frame(t)
        coords, _scores = detect_dog(
            raw, vol.quantiles,
            xy_downsample=cfg.xy_downsample,
            dog_scales=scales,
            min_distance_um=min_dist,
            rel_threshold=cfg.rel_threshold,
            max_peaks=cfg.max_peaks,
            adaptive_percentile=cfg.adaptive_percentile,
        )
        if cfg.refine and len(coords):
            coords = refine_centroids(raw, coords)
        frames.append(coords)
    return frames

def build_graph(frames: list[np.ndarray], cfg: Config) -> TrackGraph:
    """Link detections into a track graph and post-process it."""
    if cfg.bidir_weight > 0:
        g = link_harmonic(frames, max_link_um=cfg.max_link_um,
                          motion_weight=cfg.motion_weight,
                          bidir_weight=cfg.bidir_weight,
                          temperature_um=cfg.bidir_temperature_um)
    else:
        g = link_motion(frames, max_link_um=cfg.max_link_um, motion_weight=cfg.motion_weight,
                        min_margin_um=cfg.min_margin_um)
    if cfg.max_gap > 0:
        g = close_gaps(g, max_gap=cfg.max_gap, gap_um_per_frame=cfg.gap_um_per_frame)
    if cfg.min_track_len > 1:
        g = filter_short_tracks(g, min_len=cfg.min_track_len)
    if cfg.smooth_weight > 0:
        g = smooth_tracks(g, weight=cfg.smooth_weight, window=cfg.smooth_window)
    if cfg.safe_divisions:
        g = add_safe_divisions(g, max_parent_um=cfg.div_parent_um,
                               max_sister_um=cfg.div_sister_um,
                               frame_frac_cap=cfg.div_frame_frac_cap)
    return g

def run_dataset(zarr_path: Path | str, cfg: Config, progress=None) -> TrackGraph:
    vol = open_volume(zarr_path)
    frames = detect_frames(vol, cfg, t_limit=cfg.t_limit, progress=progress)
    return build_graph(frames, cfg)

# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

class SubmissionWriter:
    """Streaming writer for the competition CSV.

    Kept streaming because a full submission can reach millions of rows and
    building one giant DataFrame is a reliable way to hit the Kaggle memory
    ceiling.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._fh = self.path.open("w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(SUBMISSION_COLUMNS)
        self._row_id = 0
        self.n_nodes = 0
        self.n_edges = 0

    def add(self, dataset: str, g: TrackGraph) -> None:
        w = self._writer
        for nid, tt, zz, yy, xx in zip(g.ids, g.t, g.z, g.y, g.x):
            w.writerow([self._row_id, dataset, "node", int(nid), int(tt),
                        f"{zz:.4f}", f"{yy:.4f}", f"{xx:.4f}", -1, -1])
            self._row_id += 1
        self.n_nodes += g.n_nodes
        for s, d in g.edges:
            w.writerow([self._row_id, dataset, "edge", -1, -1, -1, -1, -1,
                        int(s), int(d)])
            self._row_id += 1
        self.n_edges += g.n_edges

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

def validate_submission(path: Path | str, expected_datasets: list[str] | None = None) -> dict:
    """Sanity-check a submission CSV; raises AssertionError on a schema problem.

    Catches the failure modes that silently score zero: a wrong header, edges
    that reference missing nodes, non-consecutive edges, and out-degree > 2.
    """
    path = Path(path)
    nodes: dict[tuple[str, int], int] = {}   # (dataset, node_id) -> t
    out_deg: dict[tuple[str, int], int] = {}
    datasets: set[str] = set()
    n_nodes = n_edges = 0

    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        assert header == SUBMISSION_COLUMNS, f"bad header: {header}"
        for row in reader:
            ds, row_type = row[1], row[2]
            datasets.add(ds)
            if row_type == "node":
                nodes[(ds, int(row[3]))] = int(row[4])
                n_nodes += 1
            elif row_type == "edge":
                key = (ds, int(row[8]))
                out_deg[key] = out_deg.get(key, 0) + 1
                n_edges += 1
            else:
                raise AssertionError(f"unknown row_type: {row_type}")

    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader)
        bad_ref = bad_dt = 0
        for row in reader:
            if row[2] != "edge":
                continue
            ds, s, d = row[1], int(row[8]), int(row[9])
            ts, td_ = nodes.get((ds, s)), nodes.get((ds, d))
            if ts is None or td_ is None:
                bad_ref += 1
            elif td_ - ts != 1:
                bad_dt += 1

    assert bad_ref == 0, f"{bad_ref} edges reference a missing node"
    over = sum(1 for v in out_deg.values() if v > 2)

    stats = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_datasets": len(datasets),
        "datasets": sorted(datasets),
        "edges_wrong_dt": bad_dt,
        "nodes_with_outdeg_gt2": over,
    }
    if expected_datasets is not None:
        missing = set(expected_datasets) - datasets
        assert not missing, f"missing datasets in submission: {sorted(missing)}"
    return stats
