"""Screen many decks against the strong public agents, seats balanced, shared seeds.

A full arena round-robin over 27 decks is 465 pairings; we only care how each deck
performs against the two ~950-rated benchmark agents, which is 54 matchups. Same
seed set for every deck, so the comparison across decks is itself paired.

Screening only — unpaired win-rates have overstated effects in 6 of 6 cases this
project. Use this to pick candidates, then confirm with tools/crn/deck_duel.py.
"""
from __future__ import annotations

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from paired_eval import play, build  # noqa: E402

OPPONENTS = ["public-archaludon", "public-alakazam"]
NSEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def main():
    import arena  # noqa: F401  (populates COMPETITORS)
    names = sys.argv[1].split(",")
    seeds = [90000 + i for i in range(NSEEDS)]
    opps = {o: build(o) for o in OPPONENTS}

    rows = []
    for name in names:
        try:
            fa, deck_a = build(name)
        except SystemExit:
            print(f"  {name}: not registered")
            continue
        total_w = total_g = 0
        per = {}
        for oname, (fo, deck_o) in opps.items():
            w = g = 0
            for s in seeds:
                for seat in (0, 1):
                    random.seed(s)
                    if seat == 0:
                        r = play(s, fa, fo, deck_a, deck_o)
                        w += int(r == 0)
                    else:
                        r = play(s, fo, fa, deck_o, deck_a)
                        w += int(r == 1)
                    g += 1
            per[oname] = w / max(g, 1)
            total_w += w
            total_g += g
        rows.append((name, total_w / max(total_g, 1), per, total_g))
        print(f"  {name:18s} {100*total_w/max(total_g,1):5.1f}%  "
              + "  ".join(f"{k.split('-')[1][:6]} {100*v:4.1f}%" for k, v in per.items()),
              flush=True)

    print("\n=== ranked vs public agents ===")
    for name, wr, per, g in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:18s} {100*wr:5.1f}%  n={g}")


if __name__ == "__main__":
    main()
