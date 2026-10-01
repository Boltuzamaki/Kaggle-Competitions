#!/usr/bin/env python
"""Joint sweep over detection and linking parameters, scored with the official metric.

Detections are the expensive part, so they are computed once per detection
setting and every linking setting is then evaluated on top of the cached result.

    PYTHONPATH=src python scripts/sweep.py --out sweep.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import biohub_ct as B  # noqa: E402
from local_eval import score_dataset, summarise_rows  # noqa: E402

DETECT_GRID = {
    "min_distance_um": [3.5, 4.5, 5.5],
    "rel_threshold": [0.01, 0.02, 0.04],
}
LINK_GRID = {
    "max_link_um": [5.0, 6.0, 7.0],
    "min_track_len": [1, 4, 6],
    "gap_um_per_frame": [5.0, 6.0],
}

def complete_datasets(root: Path, split: str) -> list[str]:
    out = []
    for zp in sorted((root / split).glob("*.zarr")):
        if not (root / "train" / f"{zp.stem}.geff").exists():
            continue
        try:
            vol = B.open_volume(zp)
        except Exception:
            continue
        cdir = zp / "0" / "c"
        if cdir.is_dir() and len(list(cdir.iterdir())) >= vol.n_t:
            out.append(zp.stem)
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/comp"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("sweep_results.csv"))
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    names = complete_datasets(args.root, args.split)
    print(f"sweeping on {len(names)} dataset(s): {names}\n")

    det_keys = list(DETECT_GRID)
    link_keys = list(LINK_GRID)
    results = []

    for det_vals in itertools.product(*(DETECT_GRID[k] for k in det_keys)):
        det = dict(zip(det_keys, det_vals))
        cfg = B.Config(**det)
        t0 = time.time()
        cache = {}
        for name in names:
            vol = B.open_volume(args.root / args.split / f"{name}.zarr")
            cache[name] = B.detect_frames(vol, cfg)
        n_peaks = np.mean([len(f) for fs in cache.values() for f in fs])
        print(f"detect {det}  peaks/frame={n_peaks:.0f}  ({time.time()-t0:.0f}s)")

        for link_vals in itertools.product(*(LINK_GRID[k] for k in link_keys)):
            link = dict(zip(link_keys, link_vals))
            cfg2 = B.Config(**det, **link)
            rows = []
            for name in names:
                g = B.build_graph(cache[name], cfg2)
                rows.append(score_dataset(g, args.root / "train" / f"{name}.geff"))
            s = summarise_rows(rows)
            rec = {**det, **link,
                   "score": s["score"], "adj": s["adj_edge_jaccard"],
                   "edge_j": s["edge_jaccard"], "recall": s["node_recall"],
                   "peaks_per_frame": float(n_peaks)}
            results.append(rec)

        results.sort(key=lambda r: -(r["adj"] if r["adj"] == r["adj"] else -1))
        print("   best so far:", json.dumps(results[0], default=float))

    import pandas as pd
    df = pd.DataFrame(results).sort_values("adj", ascending=False)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}\n")
    print(df.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
