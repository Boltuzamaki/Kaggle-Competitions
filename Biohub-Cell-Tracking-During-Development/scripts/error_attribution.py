#!/usr/bin/env python
"""Attribute every missed ground-truth edge to a specific cause.

Edge Jaccard alone says *how much* is lost, not *where*. This walks each GT
edge and classifies the failure:

    detection_miss_source / detection_miss_target
        an endpoint has no predicted node within 7 um - a detector problem
    not_linked
        both endpoints matched predicted nodes, but no edge joins them --
        a linking problem the linker could have solved
    wrong_link
        the source's predicted node links somewhere else - an identity swap
    pruned
        both endpoints were detected but a post-processing step removed them

Run:
    PYTHONPATH=src python scripts/error_attribution.py
"""

from __future__ import annotations

import argparse
import collections
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import biohub_ct as B  # noqa: E402

MAX_MATCH_UM = 7.0

def match_nodes(pred: B.TrackGraph, gt: B.TrackGraph) -> dict[int, int]:
    """GT node id -> predicted node id, optimal per timepoint within 7 um."""
    pred_by_t: dict[int, list[int]] = {}
    for i, t in enumerate(pred.t):
        pred_by_t.setdefault(int(t), []).append(i)
    gt_by_t: dict[int, list[int]] = {}
    for i, t in enumerate(gt.t):
        gt_by_t.setdefault(int(t), []).append(i)

    p_phys, g_phys = pred.physical(), gt.physical()
    mapping: dict[int, int] = {}
    for t, g_idx in gt_by_t.items():
        p_idx = pred_by_t.get(t, [])
        if not p_idx:
            continue
        d = np.sqrt(((g_phys[g_idx][:, None, :] - p_phys[p_idx][None, :, :]) ** 2).sum(2))
        cost = np.where(d <= MAX_MATCH_UM, d, 1e6)
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if d[r, c] <= MAX_MATCH_UM:
                mapping[int(gt.ids[g_idx[r]])] = int(pred.ids[p_idx[c]])
    return mapping

def attribute(pred: B.TrackGraph, gt: B.TrackGraph, pre_prune: B.TrackGraph | None = None) -> dict:
    mapping = match_nodes(pred, gt)
    pred_edges = {(int(s), int(d)) for s, d in pred.edges}
    pred_out: dict[int, list[int]] = collections.defaultdict(list)
    for s, d in pred.edges:
        pred_out[int(s)].append(int(d))

    pre_map = match_nodes(pre_prune, gt) if pre_prune is not None else {}

    causes = collections.Counter()
    for s, d in gt.edges:
        s, d = int(s), int(d)
        ps, pd = mapping.get(s), mapping.get(d)
        if ps is not None and pd is not None and (ps, pd) in pred_edges:
            causes["TP"] += 1
            continue
        if ps is None and pd is None:
            causes["detection_miss_both"] += 1
        elif ps is None:
            causes["detection_miss_source"] += 1
        elif pd is None:
            causes["detection_miss_target"] += 1
        elif pred_out.get(ps):
            causes["wrong_link"] += 1
        else:
            causes["not_linked"] += 1

    # How many of the detection misses were actually detected before pruning?
    recovered = 0
    if pre_map:
        for s, d in gt.edges:
            s, d = int(s), int(d)
            if (mapping.get(s) is None or mapping.get(d) is None) \
                    and pre_map.get(s) is not None and pre_map.get(d) is not None:
                recovered += 1
    causes["(of those, present before pruning)"] = recovered

    causes["gt_edges"] = gt.n_edges
    causes["gt_nodes_matched"] = len(mapping)
    causes["gt_nodes"] = gt.n_nodes
    return dict(causes)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/comp"))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    cfg = B.Config()
    totals = collections.Counter()

    for zp in sorted((args.root / args.split).glob("*.zarr")):
        name = zp.stem
        geff = args.root / "train" / f"{name}.geff"
        if not geff.exists():
            continue
        try:
            vol = B.open_volume(zp)
        except Exception:
            continue
        cdir = zp / "0" / "c"
        if not cdir.is_dir() or len(list(cdir.iterdir())) < vol.n_t:
            continue

        frames = B.detect_frames(vol, cfg)
        linked = B.link_motion(frames, cfg.max_link_um, cfg.motion_weight)
        bridged = B.close_gaps(linked, cfg.max_gap, cfg.gap_um_per_frame)
        final = B.build_graph(frames, cfg)
        gt = B.read_geff(geff)

        res = attribute(final, gt, pre_prune=bridged)
        print(f"\n=== {name} ===")
        for k, v in sorted(res.items(), key=lambda kv: -kv[1] if isinstance(kv[1], int) else 0):
            print(f"  {k:<34} {v}")
        for k, v in res.items():
            totals[k] += v

    print("\n=== TOTAL ===")
    gt_edges = totals["gt_edges"]
    for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
        pct = f"({100*v/gt_edges:5.1f}% of GT edges)" if gt_edges and k != "gt_edges" else ""
        print(f"  {k:<34} {v:5d} {pct}")

if __name__ == "__main__":
    main()
