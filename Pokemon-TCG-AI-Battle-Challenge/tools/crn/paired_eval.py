"""Paired A/B evaluation over common random numbers.

Standard arena runs draw fresh shuffles for every game, so most of the observed
win-rate spread is deck luck rather than policy quality -- which is why 40x more
search, tuned weights and a working opponent model all came back inside the noise
band at n=120-200.

Here both candidates play the SAME seeds against the SAME opponent. Shuffles,
coin flips and prize layouts are identical across the pair, so the difference
that remains is attributable to the policies. We report the paired outcome table
and a McNemar-style test on the games where the two candidates disagree, which
needs far fewer games than an unpaired comparison.

Usage:
    .venv/bin/python tools/crn/paired_eval.py A B --deck garchomp \
        --opponent meta-alakazam --seeds 200
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(ROOT, "agent"), os.path.join(ROOT, "tools"),
          os.path.join(ROOT, "references", "top_rankers")):
    if p not in sys.path:
        sys.path.insert(0, p)


class StartData(ctypes.Structure):
    _fields_ = [("battlePtr", ctypes.c_void_p),
                ("errorPlayer", ctypes.c_int), ("errorType", ctypes.c_int)]


class SerialData(ctypes.Structure):
    _fields_ = [("json", ctypes.c_char_p),
                ("data", ctypes.POINTER(ctypes.c_ubyte)),
                ("count", ctypes.c_int), ("selectPlayer", ctypes.c_int)]


_lib = ctypes.cdll.LoadLibrary(os.path.join(HERE, "libcrn.so"))
_lib.GameInitialize()
_lib.CrnBattleStart.restype = StartData
_lib.CrnBattleStart.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint]
_lib.BattleFinish.argtypes = [ctypes.c_void_p]
_lib.GetBattleData.restype = SerialData
_lib.GetBattleData.argtypes = [ctypes.c_void_p]
_lib.Select.restype = ctypes.c_int
_lib.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

MAX_STEPS = 6000


def play(seed, agent0, agent1, deck0, deck1):
    """One reproducible game. Returns 0/1 winner, or None."""
    cards = (ctypes.c_int * 120)(*(list(deck0) + list(deck1)))
    start = _lib.CrnBattleStart(cards, ctypes.c_uint(seed & 0xFFFFFFFF or 1))
    if start.errorPlayer >= 0:
        return None
    ptr = start.battlePtr
    agents = (agent0, agent1)
    try:
        for _ in range(MAX_STEPS):
            sd = _lib.GetBattleData(ptr)
            if not sd.json:
                return None
            obs = json.loads(ctypes.string_at(sd.json).decode("utf-8", "replace"))
            # cg/game.py attaches this in Python, not in the JSON. Without it
            # to_observation_class yields a non-agent observation and every
            # search_begin() raises, silently demoting search agents to their
            # heuristic fallback -- which invalidated the first CRN results.
            obs["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode("ascii")
            cur = obs.get("current") or {}
            if cur.get("result", -1) >= 0:
                return cur["result"]
            sel = obs.get("select")
            if sel is None:
                return None
            n = len(sel.get("option") or [])
            if n == 0:
                return None
            who = sd.selectPlayer if sd.selectPlayer in (0, 1) else 0
            try:
                choice = agents[who](obs)
            except Exception:
                choice = [0]
            choice = [c for c in (choice or [0]) if isinstance(c, int) and 0 <= c < n]
            if not choice:
                choice = [0]
            arr = (ctypes.c_int * len(choice))(*choice)
            if _lib.Select(ptr, arr, len(choice)) != 0:
                return 1 - who          # illegal action loses
        return None
    finally:
        _lib.BattleFinish(ptr)


def mcnemar(b, c):
    """Two-sided exact-ish p for discordant pairs (b wins for A only, c for B only)."""
    n = b + c
    if n == 0:
        return 1.0
    # normal approximation with continuity correction
    chi = (abs(b - c) - 1) ** 2 / n
    return math.erfc(math.sqrt(chi / 2.0))


def paired(cand_a, cand_b, opponent, deck, opp_deck, seeds):
    """Play A and B over identical seeds against the same opponent, both seats."""
    a_wins = b_wins = 0
    both = neither = a_only = b_only = 0
    for s in seeds:
        outcomes = []
        for cand in (cand_a, cand_b):
            res = []
            for seat in (0, 1):
                random.seed(s)                     # fix agent-internal randomness too
                if seat == 0:
                    w = play(s, cand, opponent, deck, opp_deck)
                    res.append(w == 0 if w is not None else None)
                else:
                    w = play(s, opponent, cand, opp_deck, deck)
                    res.append(w == 1 if w is not None else None)
            wins = sum(1 for r in res if r)
            outcomes.append(wins)
        a_wins += outcomes[0]
        b_wins += outcomes[1]
        if outcomes[0] > outcomes[1]:
            a_only += 1
        elif outcomes[1] > outcomes[0]:
            b_only += 1
        elif outcomes[0] == 2:
            both += 1
        else:
            neither += 1
    return dict(a_wins=a_wins, b_wins=b_wins, games=2 * len(seeds),
                a_only=a_only, b_only=b_only, both=both, neither=neither,
                p=mcnemar(a_only, b_only))


def paired_decks(fa, deck_a, fb, deck_b, fo, deck_o, seeds):
    """Like paired(), but each candidate pilots its own deck.

    The seed still fixes the engine's RNG stream, so both candidates face the
    same opponent draws; their own shuffles differ only because their decks do,
    which is exactly the effect being measured.
    """
    a_wins = b_wins = a_only = b_only = both = neither = 0
    for s in seeds:
        counts = []
        for cand, deck in ((fa, deck_a), (fb, deck_b)):
            wins = 0
            for seat in (0, 1):
                random.seed(s)
                if seat == 0:
                    w = play(s, cand, fo, deck, deck_o)
                    wins += int(w == 0)
                else:
                    w = play(s, fo, cand, deck_o, deck)
                    wins += int(w == 1)
            counts.append(wins)
        a_wins += counts[0]
        b_wins += counts[1]
        if counts[0] > counts[1]:
            a_only += 1
        elif counts[1] > counts[0]:
            b_only += 1
        elif counts[0] == 2:
            both += 1
        else:
            neither += 1
    return dict(a_wins=a_wins, b_wins=b_wins, games=2 * len(seeds),
                a_only=a_only, b_only=b_only, both=both, neither=neither,
                p=mcnemar(a_only, b_only))


def build(name):
    """Resolve a competitor name from tools/arena.py's registry."""
    import arena
    for n, fn, dk, kind in arena.COMPETITORS:
        if n == name:
            if kind == "search":
                import search_agent
                return arena.bind_deck(fn, dk, search_agent), list(dk)
            if kind == "arg":
                return arena.bind_deck_arg(fn, dk), list(dk)
            return arena.bind_deck(fn, dk), list(dk)
    raise SystemExit(f"unknown competitor: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--opponent", default="meta-alakazam")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=1000)
    args = ap.parse_args()

    fa, deck_a = build(args.a)
    fb, deck_b = build(args.b)
    fo, deck_o = build(args.opponent)
    seeds = [args.seed0 + i for i in range(args.seeds)]

    # Cross-deck comparisons MUST use paired_decks: paired() feeds deck_a to the
    # engine for both candidates, so a differing deck_b would be silently ignored
    # and that candidate would play someone else's list.
    if list(deck_a) != list(deck_b):
        r = paired_decks(fa, deck_a, fb, deck_b, fo, deck_o, seeds)
    else:
        r = paired(fa, fb, fo, deck_a, deck_o, seeds)
    n = len(seeds) * 2
    print(f"paired over {len(seeds)} seeds x 2 seats = {n} games each, vs {args.opponent}")
    print(f"  {args.a:22s} {r['a_wins']:4d}/{n}  {100*r['a_wins']/n:5.1f}%")
    print(f"  {args.b:22s} {r['b_wins']:4d}/{n}  {100*r['b_wins']/n:5.1f}%")
    print(f"  seed-level: A better {r['a_only']}, B better {r['b_only']}, "
          f"both win {r['both']}, both lose {r['neither']}")
    print(f"  McNemar p = {r['p']:.4f} on {r['a_only']+r['b_only']} discordant seeds")
    print("  verdict:", "difference detected" if r["p"] < 0.05 else "no detectable difference")


if __name__ == "__main__":
    main()
