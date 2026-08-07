"""Paired per-task comparison of two benchmark runs.

Paired because the tasks are the population we care about: a mean difference over
16 shared tasks is far more informative than two independent means, and the sign
test below does not assume the per-task deltas are normal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: str) -> dict:
    return {r["task"]: r for r in json.loads(Path(path).read_text())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--challenger", required=True)
    ap.add_argument("--metric", default="selected_private")
    args = ap.parse_args()

    base, chal = load(args.base), load(args.challenger)
    shared = sorted(set(base) & set(chal))

    print(f"{'task':<10} {'base':>9} {'challenger':>11} {'delta':>9}   picks(challenger)")
    print("-" * 78)
    deltas, missing = [], []
    for task in shared:
        b, c = base[task].get(args.metric), chal[task].get(args.metric)
        if b is None or c is None:
            missing.append(task)
            state = "BASE-FAIL" if b is None else "CHAL-FAIL"
            print(f"{task:<10} {'--' if b is None else f'{b:9.5f}'} "
                  f"{'--' if c is None else f'{c:11.5f}'} {state:>9}")
            continue
        d = c - b
        deltas.append(d)
        flag = "  <<" if d < -1e-9 else ("  ++" if d > 1e-9 else "")
        print(f"{task:<10} {b:9.5f} {c:11.5f} {d:+9.5f}{flag}   "
              f"{','.join(chal[task].get('selected', []))[:34]}")

    if not deltas:
        print("\nno comparable tasks")
        return
    deltas = np.array(deltas)
    wins = int((deltas > 1e-9).sum())
    losses = int((deltas < -1e-9).sum())
    print("-" * 78)
    print(f"tasks compared      : {len(deltas)}   (skipped: {missing or 'none'})")
    print(f"base mean           : {np.mean([base[t][args.metric] for t in shared if base[t].get(args.metric) is not None and chal[t].get(args.metric) is not None]):.6f}")
    print(f"challenger mean     : {np.mean([chal[t][args.metric] for t in shared if base[t].get(args.metric) is not None and chal[t].get(args.metric) is not None]):.6f}")
    print(f"mean delta          : {deltas.mean():+.6f}")
    print(f"median delta        : {np.median(deltas):+.6f}")
    print(f"std of delta        : {deltas.std(ddof=1):.6f}")
    print(f"win / tie / loss    : {wins} / {len(deltas) - wins - losses} / {losses}")
    if deltas.std(ddof=1) > 0:
        t = deltas.mean() / (deltas.std(ddof=1) / np.sqrt(len(deltas)))
        print(f"paired t-statistic  : {t:+.3f}   (|t|>2.13 ~ p<0.05 at n=16)")
    print(f"worst task regression: {deltas.min():+.6f}")


if __name__ == "__main__":
    main()
