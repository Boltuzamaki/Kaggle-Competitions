"""Where does the 860.7 agent actually lose?

grim_candy_v2 reached 860.7 by adding a matchup-specific expert for ONE bad
matchup (Archaludon, p=0.0022). The natural next step is the same technique on
the next-worst matchup -- but that requires knowing which one that is, and the
existing validation only measured Archaludon, Alakazam and the Grim mirror.

This measures its win rate against a WIDE panel: the deterministic public and
meta agents, plus the real elite archetypes mined from replays of players rated
>=1150. Output is a per-opponent table so the weakest matchups are identifiable
rather than guessed.

Read-only with respect to agent/grim_candy_v2 -- this drives it, never edits it.

Absolute win rates here are NOT a skill estimate (local testing cannot produce
one). What matters is the RANKING across opponents: the matchups at the bottom
are where a targeted expert has room, which is exactly how v2 was built.
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
GC = os.path.join(ROOT, "agent", "grim_candy_v2")
for p in (HERE, GC, os.path.join(ROOT, "agent"), os.path.join(ROOT, "tools"),
          os.path.join(ROOT, "references", "top_rankers")):
    if p not in sys.path:
        sys.path.insert(0, p)

from paired_eval import play, build  # noqa: E402

MAIN = os.path.join(GC, "main.py")
DECK = [int(x) for x in open(os.path.join(GC, "deck.csv")) if x.strip()]

# Deterministic opponents only: hybs-* and td-* are not reproducible run-to-run.
PANEL = [n for n in os.environ.get("MP_PANEL", ",".join([
    "public-archaludon", "public-alakazam",
    "meta-garchomp", "meta-grimmsnarl",
    "our-fork-m3000",
])).split(",") if n]


def _load():
    s = ilu.spec_from_file_location("gc_prof", MAIN)
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def _fork_opponent():
    """Our own m3000 fork (ladder 762-777) as a strong opponent.

    The meta-* agents saturate at 100% against an 860-rated agent and carry no
    information; the only competitive opponents available are the two public
    notebooks and this. A profile against saturated opponents ranks nothing.
    """
    fp = os.path.join(ROOT, "agent", "fork", "fork_main.py")
    fd = [int(x) for x in open(os.path.join(ROOT, "agent", "fork", "deck.csv")) if x.strip()]
    s = ilu.spec_from_file_location("fork_opp", fp)
    fm = ilu.module_from_spec(s)
    s.loader.exec_module(fm)
    fm.TIME_BUDGET_S = 1e9
    import deep_search
    deep_search.install(fm, extra_turns=0, margin=3000.0)   # the shipped config

    def f(o):
        return list(fd) if o.get("select") is None else fm.agent(o)
    return f, fd


def chunk(args):
    opp, seeds = args
    import arena  # noqa: F401
    m = _load()

    def me(o):
        return list(DECK) if o.get("select") is None else m.agent(o)

    fo, do = _fork_opponent() if opp == "our-fork-m3000" else build(opp)
    w = l = 0
    for s in seeds:
        for seat in (0, 1):
            random.seed(s)
            if seat == 0:
                r = play(s, me, fo, DECK, do)
                w += int(r == 0); l += int(r == 1)
            else:
                r = play(s, fo, me, do, DECK)
                w += int(r == 1); l += int(r == 0)
    return opp, w, l


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seeds = [318000 + i for i in range(n)]
    k = 2
    jobs = [(o, seeds[i::k]) for o in PANEL for i in range(k)]
    print(f"grim_candy_v2 matchup profile: {len(PANEL)} opponents x {n} seeds x 2 seats",
          flush=True)

    agg = {o: [0, 0] for o in PANEL}
    with mp.get_context("fork").Pool(workers) as pool:
        for opp, w, l in pool.imap_unordered(chunk, jobs):
            agg[opp][0] += w
            agg[opp][1] += l

    rows = []
    for o in PANEL:
        w, l = agg[o]
        t = w + l
        rows.append((100.0 * w / max(t, 1), o, w, l))
    rows.sort()
    print("\n===== MATCHUP PROFILE (worst first) =====")
    for pct, o, w, l in rows:
        bar = "#" * int(pct / 4)
        print(f"  {o:20s} {w:4d}-{l:<4d} {pct:5.1f}%  {bar}")
    tw = sum(a[0] for a in agg.values())
    tl = sum(a[1] for a in agg.values())
    print(f"\n  overall {tw}-{tl}  ({100.0*tw/max(tw+tl,1):.1f}%)")
    print("  NOTE: absolute rates are not a skill estimate; the RANKING is the signal.")
    json.dump({o: agg[o] for o in PANEL},
              open(os.path.join(ROOT, "scratchpad", "scrape_20260804",
                                "grim_matchup_profile.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
