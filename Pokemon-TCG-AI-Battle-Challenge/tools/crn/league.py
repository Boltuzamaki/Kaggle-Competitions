"""League benchmark: score a candidate against a DIVERSE frozen opponent pool.

Why this exists: we optimised against two public agents, got Ogerpon over Garchomp
at p=0.0146, and it scores 578 on the ladder versus Garchomp's 594. Two opponents
is not a benchmark, it is a target to overfit.

Both 1st-place write-ups the user supplied name this exact failure and the same
fix:

  Lux AI S3: "we also let the agent face a pool of older opponents at some
  fraction of games. This improved the agent's robustness, preventing overfitting
  to a single self-play style."

  Orbit Wars (author's stated regret): "I would have ... added league-play against
  past checkpoints to help prevent strategic cycles and self-overfitting."

The pool deliberately mixes STRENGTH and STYLE:
  * two frozen public ~950 agents (the strong anchor),
  * several top-team decks piloted by hybrid (diverse archetypes),
  * v3 and domain-policy variants (different policy classes),
  * a first-index bot (degenerate baseline that punishes fragile lines).

Every candidate plays the same seeds against every pool member, both seats, so the
comparison is paired across the whole league. We report per-opponent win-rates
plus a field-weighted aggregate, because an equal-weighted average across
opponents hides matchup structure.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from paired_eval import play, build, mcnemar  # noqa: E402

# name -> weight. Weights approximate how much each style matters on the ladder:
# the strong public agents anchor the top, the meta decks represent the field.
LEAGUE = OrderedDict([
    ("public-archaludon", 2.0),
    ("public-alakazam", 2.0),
    ("td-td_00", 1.5),          # #1 team's deck
    ("td-td_08", 1.5),          # keidroid's deck
    ("hybs-grimmsnarl", 1.0),   # largest field archetype
    ("hybs-dudunsparce", 1.0),
    ("v3-grimmsnarl", 1.0),     # different policy class
    ("meta-crustle", 0.5),
    ("first-garchomp", 0.5),    # degenerate baseline
])


def score(cand_name, seeds, pool):
    fa, da = build(cand_name)
    per = OrderedDict()
    per_seed = {}
    for oname, (fo, do, w) in pool.items():
        wins = games = 0
        outs = []
        for s in seeds:
            got = 0
            for seat in (0, 1):
                random.seed(s)
                if seat == 0:
                    got += int(play(s, fa, fo, da, do) == 0)
                else:
                    got += int(play(s, fo, fa, do, da) == 1)
            wins += got
            games += 2
            outs.append(got)
        per[oname] = wins / max(games, 1)
        per_seed[oname] = outs
    tw = sum(pool[o][2] for o in per)
    weighted = sum(per[o] * pool[o][2] for o in per) / max(tw, 1)
    return weighted, per, per_seed


def main():
    import arena  # noqa: F401
    cands = sys.argv[1].split(",")
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    seeds = [80000 + i for i in range(nseeds)]

    pool = OrderedDict()
    for name, w in LEAGUE.items():
        try:
            fo, do = build(name)
            pool[name] = (fo, do, w)
        except SystemExit:
            print(f"  (skipping unregistered {name})")
    print(f"league: {len(pool)} opponents, {nseeds} seeds, "
          f"{2*nseeds*len(pool)} games per candidate\n")

    results = {}
    for c in cands:
        wt, per, ps = score(c, seeds, pool)
        results[c] = (wt, per, ps)
        print(f"{c}:  weighted {100*wt:5.1f}%")
        for o, v in per.items():
            print(f"    {o:20s} {100*v:5.1f}%")
        print(flush=True)

    if len(cands) == 2:
        a, b = cands
        aa = sum(results[a][2][o][i] > results[b][2][o][i]
                 for o in results[a][2] for i in range(len(seeds)))
        bb = sum(results[a][2][o][i] < results[b][2][o][i]
                 for o in results[a][2] for i in range(len(seeds)))
        print(f"paired across league: {a} better {aa}, {b} better {bb}, "
              f"p={mcnemar(aa, bb):.4f}")

    print("\n=== league ranking ===")
    for c, (wt, per, _) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"  {c:22s} {100*wt:5.1f}%")
    json.dump({c: {"weighted": w, "per": p} for c, (w, p, _) in results.items()},
              open(os.path.join(ROOT, "scratchpad", "scrape_20260804",
                                "league_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
