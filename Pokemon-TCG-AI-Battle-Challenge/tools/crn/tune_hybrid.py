"""Sweep hybrid's search parameters against the STRONG benchmark, paired on seeds.

hybrid_agent exposes three knobs that were set by hand and never tuned:

    _TOP_K   = 6    root candidates taken from the domain policy
    _DET     = 3    determinizations averaged per decision
    _MAXROLL = 24   cap on simulated rollout steps

The earlier attempt to scale search (search_scaled.py) came back null, but that
was v3's crude leaf evaluator AND a saturated opponent -- the two conditions most
likely to hide an effect. hybrid has a better evaluator, and we now measure
against public agents, so the question is worth re-asking under conditions that
can actually answer it.

Every configuration plays the SAME seeds against the SAME opponents as the
baseline, so differences are attributable to the parameters. Timing is reported
because a config that wins but blows the clock is a loss (600 s/game budget).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from paired_eval import play, build, mcnemar  # noqa: E402
import hybrid_agent  # noqa: E402
import meta_decks  # noqa: E402

OPPONENTS = ["public-archaludon", "public-alakazam"]
DECK = meta_decks.OTHER          # the live deck
NSEEDS = int(os.environ.get("TUNE_SEEDS", "24"))

# (top_k, det, maxroll) -- baseline first
GRID = [
    (6, 3, 24),      # baseline (current live config)
    (10, 3, 24),     # wider root
    (6, 6, 24),      # more determinizations
    (6, 3, 40),      # longer rollouts
    (10, 6, 24),     # wider + more samples
    (4, 3, 24),      # narrower root (search can hurt a strong policy)
    (6, 8, 40),      # substantially more compute
]


def run_config(cfg, opps, seeds):
    hybrid_agent._TOP_K, hybrid_agent._DET, hybrid_agent._MAXROLL = cfg

    def fa(obs):
        return hybrid_agent.hybrid_agent(obs, DECK)

    wins = games = 0
    per_seed = []
    t0 = time.time()
    for oname, (fo, deck_o) in opps.items():
        for s in seeds:
            got = 0
            for seat in (0, 1):
                random.seed(s)
                if seat == 0:
                    got += int(play(s, fa, fo, DECK, deck_o) == 0)
                else:
                    got += int(play(s, fo, fa, deck_o, DECK) == 1)
            wins += got
            games += 2
            per_seed.append(got)
    return wins / max(games, 1), per_seed, (time.time() - t0) / max(games, 1)


def main():
    import arena  # noqa: F401
    opps = {o: build(o) for o in OPPONENTS}
    seeds = [70000 + i for i in range(NSEEDS)]

    base_cfg = GRID[0]
    base_wr, base_seeds, base_t = run_config(base_cfg, opps, seeds)
    print(f"baseline {base_cfg}: {100*base_wr:5.1f}%  {base_t:.2f}s/game", flush=True)

    results = [(base_cfg, base_wr, 0, 0, 1.0, base_t)]
    for cfg in GRID[1:]:
        wr, ps, t = run_config(cfg, opps, seeds)
        a = sum(1 for x, y in zip(ps, base_seeds) if x > y)
        b = sum(1 for x, y in zip(ps, base_seeds) if x < y)
        p = mcnemar(a, b)
        results.append((cfg, wr, a, b, p, t))
        print(f"  {str(cfg):14s} {100*wr:5.1f}%  (base {100*base_wr:4.1f}%)  "
              f"discordant {a}/{b}  p={p:.3f}  {t:.2f}s/game", flush=True)

    print("\n=== ranked ===")
    for cfg, wr, a, b, p, t in sorted(results, key=lambda r: -r[1]):
        flag = "  <-- significant" if p < 0.05 and wr > base_wr else ""
        print(f"  {str(cfg):14s} {100*wr:5.1f}%  p={p:.3f}  {t:.2f}s/game{flag}")
    json.dump([{"cfg": list(c), "wr": w, "a": a, "b": b, "p": p, "sec_per_game": t}
               for c, w, a, b, p, t in results],
              open(os.path.join(ROOT, "scratchpad", "scrape_20260804",
                                "hybrid_tuning.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
