#!/usr/bin/env python
"""Score the classical pipeline locally against ground truth.

Any dataset that has both a ``.zarr`` and a ``.geff`` can be scored, so this
works on the placeholder test videos *and* on any training video you have
downloaded.

    PYTHONPATH=src python scripts/run_local_cv.py --limit 8
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import biohub_ct as B  # noqa: E402
from local_eval import format_summary, score_dataset, summarise_rows  # noqa: E402

def scorable(root: Path, split: str) -> list[str]:
    """Datasets under ``root/split`` whose ground truth exists in ``root/train``.

    A complete ``.zarr`` is required: partially downloaded volumes are skipped
    rather than silently scored on missing frames.
    """
    d = root / split
    train = root / "train"
    out = []
    for zarr_path in sorted(d.glob("*.zarr")):
        name = zarr_path.stem
        if not (train / f"{name}.geff").exists():
            continue
        try:
            vol = B.open_volume(zarr_path)
        except Exception:
            continue
        n_chunks = len(list((zarr_path / "0" / "c").iterdir())) if (zarr_path / "0" / "c").is_dir() else 0
        if n_chunks < vol.n_t:
            print(f"  skip {name}: {n_chunks}/{vol.n_t} chunks present")
            continue
        out.append(name)
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/comp"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-link-um", type=float, default=None)
    ap.add_argument("--min-track-len", type=int, default=None)
    ap.add_argument("--rel-threshold", type=float, default=None)
    ap.add_argument("--min-distance-um", type=float, default=None)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")

    cfg = B.Config()
    for attr, val in [("max_link_um", args.max_link_um),
                      ("min_track_len", args.min_track_len),
                      ("rel_threshold", args.rel_threshold),
                      ("min_distance_um", args.min_distance_um)]:
        if val is not None:
            setattr(cfg, attr, val)

    names = scorable(args.root, args.split)
    if args.limit:
        names = names[:args.limit]
    print(f"scoring {len(names)} dataset(s) from {args.root/args.split}\n")

    rows = []
    for name in names:
        t0 = time.time()
        vol = B.open_volume(args.root / args.split / f"{name}.zarr")
        g = B.build_graph(B.detect_frames(vol, cfg), cfg)
        r = score_dataset(g, args.root / "train" / f"{name}.geff")
        rows.append(r)
        print(f"  {name}: adj={r['adj_edge_jaccard']:.4f} J={r['edge_jaccard']:.4f} "
              f"recall={r['node_recall']:.3f} nodes={g.n_nodes} "
              f"TP/FP/FN={r['edge_tp']}/{r['edge_fp']}/{r['edge_fn']} ({time.time()-t0:.0f}s)")

    if rows:
        print("\n" + format_summary(summarise_rows(rows)))

if __name__ == "__main__":
    main()
