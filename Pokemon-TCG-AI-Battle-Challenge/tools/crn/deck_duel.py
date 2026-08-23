"""Paired deck comparison: v3 search on deck X vs v3 search on deck Y, same seeds.

Policy is held constant, so every discordant seed is attributable to the deck.
Runs each candidate against a panel of fixed opponents and reports the aggregate.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from paired_eval import build, paired, paired_decks, mcnemar  # noqa: E402

CANDIDATES = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "sweep-garchomp", "sweep-grimmsnarl", "sweep-dragapult", "sweep-dudunsparce",
]
OPPONENTS = sys.argv[2].split(",") if len(sys.argv) > 2 else [
    "meta-alakazam", "meta-grimmsnarl",
]
NSEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
OUT = os.path.join(ROOT, "scratchpad", "scrape_20260804", "deck_duel.json")

seeds = [5000 + i for i in range(NSEEDS)]
built = {n: build(n) for n in set(CANDIDATES) | set(OPPONENTS)}
results = []

for a, b in itertools.combinations(CANDIDATES, 2):
    agg = {"a": a, "b": b, "a_wins": 0, "b_wins": 0,
           "a_only": 0, "b_only": 0, "games": 0}
    for opp in OPPONENTS:
        fa, da = built[a]
        fb, db = built[b]
        fo, do = built[opp]
        if da != db:
            # decks differ -- that is the point; each plays its own list
            r = paired_decks(fa, da, fb, db, fo, do, seeds)
        else:
            r = paired(fa, fb, fo, da, do, seeds)
        for k in ("a_wins", "b_wins", "a_only", "b_only", "games"):
            agg[k] += r[k]
        print(f"  {a} vs {b} @ {opp}: "
              f"{r['a_wins']}-{r['b_wins']} of {r['games']}, "
              f"discordant {r['a_only']}/{r['b_only']}", flush=True)
    agg["p"] = mcnemar(agg["a_only"], agg["b_only"])
    results.append(agg)
    print(f"{a} vs {b}: A {agg['a_wins']} / B {agg['b_wins']} of {agg['games']} each, "
          f"discordant {agg['a_only']}/{agg['b_only']}, p={agg['p']:.4f}\n", flush=True)

results.sort(key=lambda r: -(r["a_wins"] - r["b_wins"]))
json.dump(results, open(OUT, "w"), indent=1)
print("wrote", OUT)
