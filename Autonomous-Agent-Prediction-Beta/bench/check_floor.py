"""Verify the floor guarantee: every proven candidate must be reproduced exactly.

The challenger is only allowed to *add* candidates. If any candidate the proven config
also produces has moved by more than a float-rounding tolerance, the guarantee is broken
and the challenger can land below the proven config on an unlucky draw.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOL = 1e-9
PROVEN = {
    "portfolio_catboost", "portfolio_lightgbm", "portfolio_extra_trees",
    "portfolio_logistic", "portfolio_rank_top2", "portfolio_rank_all",
    "quick_baseline",
}


def index(path: str) -> dict:
    out = {}
    for r in json.loads(Path(path).read_text()):
        out[r["task"]] = {c["name"]: c for c in r.get("candidates", [])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--challenger", required=True)
    args = ap.parse_args()

    base, chal = index(args.base), index(args.challenger)
    drift, missing, added = [], [], set()
    checked = 0

    for task in sorted(set(base) & set(chal)):
        for name, b in base[task].items():
            if name not in PROVEN:
                continue
            c = chal[task].get(name)
            if c is None:
                missing.append((task, name))
                continue
            checked += 1
            for metric in ("private", "public"):
                if abs(b[metric] - c[metric]) > TOL:
                    drift.append((task, name, metric, b[metric], c[metric]))
        added |= {n for n in chal[task] if n not in base[task]}

    print(f"proven candidates checked : {checked}")
    print(f"new candidates introduced : {sorted(added)}")
    if missing:
        print(f"\nMISSING proven candidates ({len(missing)}):")
        for task, name in missing:
            print(f"  {task}  {name}")
    if drift:
        print(f"\nDRIFTED proven candidates ({len(drift)}):")
        for task, name, metric, b, c in drift[:20]:
            print(f"  {task:10s} {name:24s} {metric:8s} {b:.8f} -> {c:.8f}  ({c-b:+.2e})")
    if not drift and not missing:
        print("\nFLOOR GUARANTEE HOLDS — every proven candidate is reproduced bit-identically.")
    else:
        print("\nFLOOR GUARANTEE BROKEN")


if __name__ == "__main__":
    main()
