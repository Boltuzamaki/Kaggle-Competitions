"""Is there an ORDER effect in the paired harness?

The stock-vs-stock control keeps coming back biased in one direction: 4/5 then
1/6 discordant, both favouring the arm that plays SECOND. Noise would scatter;
this does not. Every candidate tested so far played first, so a second-mover
advantage would bias every result AGAINST the candidate -- which would mean the
long run of "flat or slightly worse" verdicts is partly an artifact.

Suspect: in clean_battery.chunk() the opponent is built ONCE per chunk and reused
for both arms' games. If that opponent module carries state across games (the fork
keeps pre_turn / ability_used_* / my_deck at module level, and the public agents
may do likewise), the second arm plays a differently-warmed opponent.

Two conditions, identical in every other respect:
    normal   A plays first, B second
    swapped  B plays first, A second
Both arms are STOCK, so any consistent asymmetry is harness bias, not strength.

If the bias follows the ORDER, the fix is to rebuild the opponent per game.
"""
from __future__ import annotations

import importlib.util as ilu
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "agent"), os.path.join(ROOT, "tools"),
          os.path.join(ROOT, "references", "top_rankers")):
    sys.path.insert(0, p)

from paired_eval import play, build  # noqa: E402

FP = os.path.join(ROOT, "agent", "fork", "fork_main.py")
DECK = [int(x) for x in open(os.path.join(ROOT, "agent", "fork", "deck.csv")) if x.strip()]
PANEL = ["public-archaludon", "meta-grimmsnarl", "public-alakazam"]


def _mk(tag):
    s = ilu.spec_from_file_location("fm_" + tag, FP)
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)
    m._TEMPLATES = []; m._TEMPLATE_SIG = []
    m.TIME_BUDGET_S = 1e9

    def w(o, _m=m):
        return list(DECK) if o.get("select") is None else _m.agent(o)
    return w


def chunk(args):
    mode, seeds = args
    import arena  # noqa: F401
    A, B = _mk("a" + mode), _mk("b" + mode)
    a = b = 0
    for opp in PANEL:
        fo, do = build(opp)
        for s in seeds:
            def run(agent):
                g = 0
                for seat in (0, 1):
                    random.seed(s)
                    g += int(play(s, agent, fo, DECK, do) == 0) if seat == 0 else \
                         int(play(s, fo, agent, do, DECK) == 1)
                return g
            if mode == "normal":
                x = run(A); y = run(B)
            elif mode == "swapped":
                y = run(B); x = run(A)
            else:  # fresh opponent rebuilt before EACH arm
                fo, do = build(opp)
                x = run(A)
                fo, do = build(opp)
                y = run(B)
            if x > y: a += 1
            elif y > x: b += 1
    return mode, a, b


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    seeds = [188000 + i for i in range(n)]
    modes = ["normal", "swapped", "fresh"]
    k = max(1, workers // len(modes))
    jobs = [(m, seeds[i::k]) for m in modes for i in range(k)]
    print(f"order-effect test: {n} seeds x {len(PANEL)} opponents, both arms STOCK",
          flush=True)
    agg = {m: [0, 0] for m in modes}
    with mp.get_context("fork").Pool(workers) as pool:
        for mode, a, b in pool.imap_unordered(chunk, jobs):
            agg[mode][0] += a; agg[mode][1] += b
    print()
    for m in modes:
        a, b = agg[m]
        note = ""
        if m == "normal":
            note = "  (A first)"
        elif m == "swapped":
            note = "  (B first -- bias should FLIP if order-driven)"
        else:
            note = "  (opponent rebuilt per arm -- should be ~0/0 if that is the cause)"
        print(f"  {m:8s} discordant A:{a:3d} / B:{b:3d}{note}")


if __name__ == "__main__":
    main()
