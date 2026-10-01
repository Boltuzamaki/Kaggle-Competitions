#!/usr/bin/env python
"""Build a labelled candidate-pair dataset for the learned edge scorer.

For each annotated timepoint pair (t, t+1) in a video: run the detector, match
the detections to ground-truth nodes within 7 µm, and emit every candidate pair
whose *source* is a matched GT node. The label is 1 when the ground truth joins
the two matched nodes.

Restricting positives and negatives to pairs anchored on a matched GT node is
what keeps the labels honest: for any other source we simply do not know the
right answer, and treating "unknown" as negative would teach the model to
suppress correct links between unannotated cells.

    PYTHONPATH=src python scripts/build_edge_dataset.py --out edge_pairs.npz
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import biohub_ct as B  # noqa: E402
from biohub_link import FEATURE_NAMES, pair_features  # noqa: E402

MATCH_UM = 7.0

def match_frame(pred: np.ndarray, gt: np.ndarray) -> dict[int, int]:
    """GT row index -> predicted row index, optimal within 7 µm."""
    if len(pred) == 0 or len(gt) == 0:
        return {}
    d = np.sqrt((((gt[:, None, :] - pred[None, :, :]) * B.SCALE) ** 2).sum(2))
    ri, ci = linear_sum_assignment(np.where(d <= MATCH_UM, d, 1e6))
    return {int(r): int(c) for r, c in zip(ri, ci) if d[r, c] <= MATCH_UM}

def build_for_dataset(zarr_path: Path, geff_path: Path, cfg: B.Config,
                      use_appearance: bool = True) -> tuple[np.ndarray, np.ndarray]:
    vol = B.open_volume(zarr_path)
    g = B.read_geff(geff_path)

    gt_by_t: dict[int, list[tuple[int, list[float]]]] = {}
    for nid, tt, zz, yy, xx in zip(g.ids, g.t, g.z, g.y, g.x):
        gt_by_t.setdefault(int(tt), []).append((int(nid), [float(zz), float(yy), float(xx)]))
    gt_edges = {(int(s), int(d)) for s, d in g.edges}

    # Only timepoints where an annotated edge starts are useful.
    id_to_t = {int(nid): int(tt) for nid, tt in zip(g.ids, g.t)}
    useful_t = sorted({id_to_t[s] for s, _ in gt_edges if s in id_to_t})

    X, y = [], []
    frame_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def detections(t: int):
        if t not in frame_cache:
            raw = vol.frame(t)
            c, _ = B.detect_dog(raw, vol.quantiles, cfg.xy_downsample, cfg.dog_scales,
                                cfg.min_distance_um, cfg.rel_threshold, cfg.max_peaks)
            if len(c):
                c = B.refine_centroids(raw, c)
            frame_cache[t] = (c, raw if use_appearance else None)
        return frame_cache[t]

    for t in useful_t:
        if t + 1 >= vol.n_t:
            continue
        src_det, src_raw = detections(t)
        tgt_det, tgt_raw = detections(t + 1)
        if len(src_det) == 0 or len(tgt_det) == 0:
            continue

        gt_src = gt_by_t.get(t, [])
        gt_tgt = gt_by_t.get(t + 1, [])
        if not gt_src or not gt_tgt:
            continue

        m_src = match_frame(src_det, np.array([c for _, c in gt_src]))
        m_tgt = match_frame(tgt_det, np.array([c for _, c in gt_tgt]))
        if not m_src:
            continue

        pairs, feats, _dist = pair_features(
            src_det, tgt_det, velocity=None, src_ids=None, gate_um=12.0,
            src_vol=src_raw, tgt_vol=tgt_raw)
        if len(pairs) == 0:
            continue

        # predicted-row -> GT node id, for the two frames
        src_row_to_gt = {v: gt_src[k][0] for k, v in m_src.items()}
        tgt_row_to_gt = {v: gt_tgt[k][0] for k, v in m_tgt.items()}

        for (si, ti), f in zip(pairs, feats):
            gs = src_row_to_gt.get(int(si))
            if gs is None:
                continue                      # source is not an annotated cell -> label unknown
            gt_t = tgt_row_to_gt.get(int(ti))
            label = 1 if (gt_t is not None and (gs, gt_t) in gt_edges) else 0
            X.append(f)
            y.append(label)

    if not X:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32), np.zeros(0, np.int8)
    return np.asarray(X, np.float32), np.asarray(y, np.int8)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/comp"))
    ap.add_argument("--out", type=Path, default=Path("edge_pairs.npz"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-appearance", action="store_true")
    ap.add_argument("--min-distance-um", type=float, default=None,
                    help="detector NMS radius; use a small value (2.0-2.5) to train in the "
                         "dense-candidate regime where linking actually fails")
    ap.add_argument("--gate-um", type=float, default=12.0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    cfg = B.Config()
    if args.min_distance_um is not None:
        cfg.min_distance_um = args.min_distance_um
    print(f'detector min_distance_um={cfg.min_distance_um}')
    train = args.root / "train"
    names = []
    for zp in sorted(train.glob("*.zarr")):
        if not (train / f"{zp.stem}.geff").exists():
            continue
        c = zp / "0" / "c"
        try:
            vol = B.open_volume(zp)
        except Exception:
            continue
        if c.is_dir() and len(list(c.iterdir())) >= vol.n_t:
            names.append(zp.stem)
    # the placeholder test videos also live under train/ with full volumes
    for zp in sorted((args.root / "test").glob("*.zarr")):
        if (train / f"{zp.stem}.geff").exists() and zp.stem not in names:
            try:
                vol = B.open_volume(zp)
            except Exception:
                continue
            c = zp / "0" / "c"
            if c.is_dir() and len(list(c.iterdir())) >= vol.n_t:
                names.append(zp.stem)

    if args.limit:
        names = names[:args.limit]
    print(f"building from {len(names)} dataset(s)")

    Xs, ys, used = [], [], []
    for name in names:
        zarr_path = train / f"{name}.zarr"
        if not zarr_path.is_dir():
            zarr_path = args.root / "test" / f"{name}.zarr"
        t0 = time.time()
        X, y = build_for_dataset(zarr_path, train / f"{name}.geff", cfg,
                                 use_appearance=not args.no_appearance)
        if len(X):
            Xs.append(X); ys.append(y); used.append(name)
            print(f"  {name}: {len(X):6d} pairs, {int(y.sum()):3d} positive "
                  f"({100*y.mean():.2f}%)  ({time.time()-t0:.0f}s)")

    if not Xs:
        raise SystemExit("no labelled pairs produced - check that volumes and geffs are complete")
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    np.savez_compressed(args.out, X=X, y=y, features=np.array(FEATURE_NAMES), datasets=np.array(used))
    print(f"\n{len(X)} pairs, {int(y.sum())} positive ({100*y.mean():.2f}%) -> {args.out}")

if __name__ == "__main__":
    main()
