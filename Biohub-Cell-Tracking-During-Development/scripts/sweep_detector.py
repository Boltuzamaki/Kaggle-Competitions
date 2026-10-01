#!/usr/bin/env python
"""Sweep the DoG scale-space, which error attribution identified as the bottleneck.

13.1% of ground-truth edges are lost to detection misses versus 4.0% to linking
mistakes, so the band-pass geometry is what matters. Scored end-to-end with the
official metric, not on a recall proxy - a scale set that raises recall while
inflating the node count can still lose on the adjustment factor.

    PYTHONPATH=src python scripts/sweep_detector.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import biohub_ct as B  # noqa: E402
from local_eval import score_dataset, summarise_rows  # noqa: E402

SCALE_SETS = {
    "current 2-scale":      ((1.5, 4.5), (2.5, 7.5)),
    "fine 2-scale":         ((1.0, 3.0), (1.75, 5.25)),
    "3-scale fine":         ((1.0, 3.0), (1.5, 4.5), (2.5, 7.5)),
    "4-scale wide":         ((1.0, 3.0), (1.5, 4.5), (2.0, 6.0), (2.5, 7.5)),
    "3-scale tight-ratio":  ((1.2, 2.8), (1.8, 4.2), (2.6, 6.1)),
    "single small":         ((1.5, 4.5),),
}

def complete(root: Path, split: str) -> list[str]:
    out = []
    for zp in sorted((root / split).glob("*.zarr")):
        if not (root / "train" / f"{zp.stem}.geff").exists():
            continue
        try:
            vol = B.open_volume(zp)
        except Exception:
            continue
        c = zp / "0" / "c"
        if c.is_dir() and len(list(c.iterdir())) >= vol.n_t:
            out.append(zp.stem)
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/comp"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("sweep_detector.csv"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    names = complete(args.root, args.split)
    print(f"scoring on {len(names)} dataset(s): {names}\n")

    results = []
    for label, scales in SCALE_SETS.items():
        for min_dist in (3.0, 3.5, 4.5):
            for rel in (0.01, 0.02):
                cfg = B.Config(dog_scales=scales, min_distance_um=min_dist, rel_threshold=rel)
                t0 = time.time()
                rows, peaks, recalls = [], [], []
                for name in names:
                    vol = B.open_volume(args.root / args.split / f"{name}.zarr")
                    frames = B.detect_frames(vol, cfg)
                    g = B.build_graph(frames, cfg)
                    r = score_dataset(g, args.root / "train" / f"{name}.geff")
                    rows.append(r)
                    peaks.append(np.mean([len(f) for f in frames]))
                    recalls.append(r["node_recall"])
                s = summarise_rows(rows)
                rec = {"scales": label, "n_scales": len(scales),
                       "min_distance_um": min_dist, "rel_threshold": rel,
                       "adj": s["adj_edge_jaccard"], "edge_j": s["edge_jaccard"],
                       "recall": s["node_recall"],
                       "recall_hard": recalls[-1] if len(recalls) > 1 else float("nan"),
                       "peaks_per_frame": float(np.mean(peaks))}
                results.append(rec)
                print(f"{label:<20} md={min_dist} rel={rel}  adj={rec['adj']:.4f}  "
                      f"recall={rec['recall']:.3f} (hard {rec['recall_hard']:.3f})  "
                      f"peaks={rec['peaks_per_frame']:.0f}  ({time.time()-t0:.0f}s)")

    import pandas as pd
    df = pd.DataFrame(results).sort_values("adj", ascending=False)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}\n")
    print(df.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
