"""Single-concept deck transplants from the LB-1170.6 list, tested one at a time.

Swapping M Sato's whole list in scored 102-125 (p=0.055, rejected). But that test
changed eleven card slots at once, so it could only answer "is their entire list
better under this policy" -- not "is any individual difference an improvement".
A net-negative bundle can easily contain a positive component.

Their list differs from the fork's in exactly four 60-card-neutral concepts:

  dunsparce_line   Dunsparce(305) 1->3, Dunsparce(65) 2->0
  mine_over_dudu   Dudunsparce 4->2, Nighttime Mine 0->2
  tech_basics      Night Stretcher 3->1, +Fezandipiti ex, +Shaymin
  hammer_pkg       -Lillie's Determination, -Neutralization Zone,
                   +Enriching Energy, +Enhanced Hammer (3->4)

Each is played against the stock list under the IDENTICAL stock policy, paired on
shared seeds across a four-opponent panel. One process per variant.

Deliberately NOT touching the Alakazam/Abra/Kadabra/Rare Candy/Poffin/energy
engine that the 69 card-specific weights key on -- that is what made the wholesale
swap uninterpretable.
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

DUNSPARCE_305, DUNSPARCE_65, DUDUNSPARCE = 305, 65, 66
NIGHT_STRETCHER, NIGHTTIME_MINE = 1097, 1266
FEZANDIPITI, SHAYMIN = 140, 343
LILLIES, NEUTRAL_ZONE = 1227, 1247
ENRICHING_ENERGY, ENHANCED_HAMMER = 13, 1081

VARIANTS = {
    "dunsparce_line": {DUNSPARCE_305: +2, DUNSPARCE_65: -2},
    "mine_over_dudu": {DUDUNSPARCE: -2, NIGHTTIME_MINE: +2},
    "tech_basics":    {NIGHT_STRETCHER: -2, FEZANDIPITI: +1, SHAYMIN: +1},
    "hammer_pkg":     {LILLIES: -1, NEUTRAL_ZONE: -1,
                       ENRICHING_ENERGY: +1, ENHANCED_HAMMER: +1},
}

OPPS = ["td-td_08", "meta-grimmsnarl", "public-archaludon", "hybs-other"]


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


def run(args):
    name, deck, n = args
    import arena  # noqa: F401
    s = ilu.spec_from_file_location("fm_" + name, FP)
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)

    def bind(d):
        def w(o):
            return list(d) if o.get("select") is None else m.agent(o)
        return w

    A, B = bind(deck), bind(STOCK)
    a = b = aw = bw = 0
    for opp in OPPS:
        fo, do = build(opp)
        for seed in [552000 + i for i in range(n)]:
            x = y = 0
            for seat in (0, 1):
                random.seed(seed)
                x += int(play(seed, A, fo, deck, do) == 0) if seat == 0 else \
                     int(play(seed, fo, A, do, deck) == 1)
            for seat in (0, 1):
                random.seed(seed)
                y += int(play(seed, B, fo, STOCK, do) == 0) if seat == 0 else \
                     int(play(seed, fo, B, do, STOCK) == 1)
            aw += x
            bw += y
            if x > y:
                a += 1
            elif y > x:
                b += 1
    return name, aw, bw, a, b


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    jobs = []
    for name, delta in VARIANTS.items():
        deck = apply_delta(delta)
        if len(deck) != 60:
            print(f"SKIP {name}: {len(deck)} cards")
            continue
        jobs.append((name, deck, n))
    print(f"{len(jobs)} variants x {n} seeds x {len(OPPS)} opponents "
          f"= {n*len(OPPS)*4} games each", flush=True)

    with mp.get_context("fork").Pool(len(jobs)) as pool:
        for name, aw, bw, a, b in pool.imap_unordered(run, jobs):
            p = mcnemar(a, b)
            tag = "BETTER" if (a > b and p < 0.05) else ("worse" if b > a else "flat")
            print(f"  {name:16s} {aw:4d} vs stock {bw:4d}  disc {a:3d}/{b:3d}  "
                  f"p={p:.4f}  {tag}", flush=True)


if __name__ == "__main__":
    main()
