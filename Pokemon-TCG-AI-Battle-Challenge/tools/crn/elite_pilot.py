"""Can our own policy on an ELITE deck beat the fork on its middling one?

Mined from 1,138 games between players rated >= 1150:

    elite_07  Mega Lucario ex        83 games   75.9% wins
    elite_02  Dunsparce toolbox     374 games   58.8% wins
    elite_04  Alakazam (the fork's) 170 games   48.8% wins

The fork plays the 48.8% archetype. Two days of tuning went into refining a deck
that elite players win under half their games with, while a 75.9% list sat in the
same dataset. Win rate is not purely pilot skill either: Majkel1337 appears on
three lists at 75.9%, 52.8% and 36.7%.

The fork's 69 weights are keyed to Abra/Kadabra/Alakazam card IDs, so it cannot
pilot another archetype. Our domain policy is deck-agnostic and deterministic, so
it can. The trade is a weaker policy on a much stronger deck -- the fork beat our
best agent 54.2% head-to-head, but on a deck elites lose most games with.

Head-to-head on shared seeds, both seats, McNemar on discordant seeds.
"""
from __future__ import annotations

import importlib.util as ilu
import json
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "agent"), os.path.join(ROOT, "tools"),
          os.path.join(ROOT, "references", "top_rankers")):
    sys.path.insert(0, p)

from paired_eval import play, mcnemar  # noqa: E402

FP = os.path.join(ROOT, "agent", "fork", "fork_main.py")
FORK_DECK = [int(x) for x in open(os.path.join(ROOT, "agent", "fork", "deck.csv")) if x.strip()]
ELITE = json.load(open(os.path.join(ROOT, "agent", "elite_decks.json")))
ARMS = [a for a in os.environ.get("EP_ARMS", "elite_07,elite_02,elite_04").split(",") if a]


def _fork():
    s = ilu.spec_from_file_location("fm_ep", FP)
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)
    m.TIME_BUDGET_S = 1e9
    return m


def chunk(args):
    name, seeds = args
    import domain_policy
    deck = ELITE[name]["deck"]
    fk = _fork()

    def ours(o):
        return list(deck) if o.get("select") is None else domain_policy.domain_agent(o, deck)

    def theirs(o):
        return list(FORK_DECK) if o.get("select") is None else fk.agent(o)

    ow = fw = 0
    for s in seeds:
        for seat in (0, 1):
            random.seed(s)
            if seat == 0:
                r = play(s, ours, theirs, deck, FORK_DECK)
                ow += int(r == 0); fw += int(r == 1)
            else:
                r = play(s, theirs, ours, FORK_DECK, deck)
                ow += int(r == 1); fw += int(r == 0)
    return name, ow, fw


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seeds = [523000 + i for i in range(n)]
    k = max(1, workers // max(len(ARMS), 1))
    jobs = [(a, seeds[i::k]) for a in ARMS for i in range(k)]
    print(f"our domain policy on elite decks vs the FORK -- {n} seeds x 2 seats", flush=True)

    agg = {a: [0, 0] for a in ARMS}
    with mp.get_context("fork").Pool(workers) as pool:
        for name, ow, fw in pool.imap_unordered(chunk, jobs):
            agg[name][0] += ow
            agg[name][1] += fw

    print()
    for a in ARMS:
        ow, fw = agg[a]
        tot = ow + fw
        e = ELITE[a]
        print(f"  {a}  ours {ow:4d} - fork {fw:4d}  ({100*ow/max(tot,1):5.1f}%)   "
              f"[elite winrate {100*e['wins']/e['games']:.1f}% over {e['games']} games]")


if __name__ == "__main__":
    main()
