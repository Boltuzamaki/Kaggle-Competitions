"""High-power confirmation of the tech_basics deck change, plus decomposition.

First pass (40 seeds x 4 opponents, seeds 552xxx): 215 vs 196, discordant 47/31,
p=0.0894. Trending but underpowered -- at that 60/40 split it needs roughly 200
discordant seeds to resolve, and it had 78.

tech_basics = -2 Night Stretcher, +1 Fezandipiti ex, +1 Shaymin. Two mechanisms
could be doing the work and the bundle cannot tell them apart:
  * more Basic Pokemon lowers the mulligan rate (the stock list is basic-light);
  * Fezandipiti ex is a draw engine that triggers on knockouts.
So the full change is run alongside each half on its own.

Seeds are DISJOINT from the first pass and the opponent panel is wider. The stock
arm is played once per seed/opponent and shared by all three comparisons, which
cuts the cost by a third.
"""
from __future__ import annotations

import importlib.util as ilu
import multiprocessing as mp
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

from paired_eval import play, build, mcnemar  # noqa: E402

FP = os.path.join(ROOT, "agent", "fork", "fork_main.py")
STOCK = [int(x) for x in open(os.path.join(ROOT, "agent", "fork", "deck.csv")) if x.strip()]

NIGHT_STRETCHER, FEZANDIPITI, SHAYMIN = 1097, 140, 343

VARIANTS = {
    "tech_basics": {NIGHT_STRETCHER: -2, FEZANDIPITI: +1, SHAYMIN: +1},
    "fez_only":    {NIGHT_STRETCHER: -1, FEZANDIPITI: +1},
    "shaymin_only": {NIGHT_STRETCHER: -1, SHAYMIN: +1},
}
OPPS = ["td-td_08", "public-archaludon", "meta-grimmsnarl", "public-alakazam"]


def apply_delta(delta):
    c = Counter(STOCK)
    for cid, d in delta.items():
        c[cid] = c.get(cid, 0) + d
        if c[cid] <= 0:
            c.pop(cid, None)
    deck = []
    for cid in sorted(c):
        deck += [cid] * c[cid]
    return deck


DECKS = {k: apply_delta(v) for k, v in VARIANTS.items()}


def _bind(m, deck):
    def w(o):
        return list(deck) if o.get("select") is None else m.agent(o)
    return w


def chunk(args):
    seeds, = args
    import arena  # noqa: F401
    s = ilu.spec_from_file_location("fm_tb", FP)
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)
    arms = {"stock": _bind(m, STOCK)}
    for k, d in DECKS.items():
        arms[k] = _bind(m, d)
    wins = {k: 0 for k in arms}
    disc = {k: [0, 0] for k in DECKS}
    for opp in OPPS:
        fo, do = build(opp)
        for seed in seeds:
            got = {}
            for tag, fn in arms.items():
                deck = STOCK if tag == "stock" else DECKS[tag]
                g = 0
                for seat in (0, 1):
                    random.seed(seed)
                    g += int(play(seed, fn, fo, deck, do) == 0) if seat == 0 else \
                         int(play(seed, fo, fn, do, deck) == 1)
                got[tag] = g
                wins[tag] += g
            for k in DECKS:
                if got[k] > got["stock"]:
                    disc[k][0] += 1
                elif got["stock"] > got[k]:
                    disc[k][1] += 1
    return wins, disc


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    for k, d in DECKS.items():
        assert len(d) == 60 and max(Counter(d).values()) <= 4, (k, len(d))
    seeds = [136000 + i for i in range(n)]      # disjoint from the 552xxx pass
    parts = [seeds[i::workers] for i in range(workers)]
    print(f"{n} seeds x {len(OPPS)} opponents x {len(DECKS)+1} arms, {workers} workers "
          f"= {n*len(OPPS)*(len(DECKS)+1)*2} games", flush=True)

    wins = {}
    disc = {k: [0, 0] for k in DECKS}
    with mp.get_context("fork").Pool(workers) as pool:
        for w, d in pool.imap_unordered(chunk, [(p,) for p in parts]):
            for k, v in w.items():
                wins[k] = wins.get(k, 0) + v
            for k, v in d.items():
                disc[k][0] += v[0]
                disc[k][1] += v[1]

    ng = n * len(OPPS) * 2
    print(f"\nstock: {wins.get('stock',0)}/{ng}")
    for k in DECKS:
        a, b = disc[k]
        p = mcnemar(a, b)
        tag = "BETTER" if (a > b and p < 0.05) else ("worse" if b > a else "flat")
        print(f"  {k:13s} {wins.get(k,0):4d}/{ng}  disc {a:3d}/{b:3d}  p={p:.4f}  {tag}",
              flush=True)


if __name__ == "__main__":
    main()
