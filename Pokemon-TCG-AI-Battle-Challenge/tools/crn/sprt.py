"""SPRT over paired CRN games: stop as soon as a change is proven good or bad.

Adopted from the chess-engine methodology in the Approvers write-up ("we used
SPRT to determine whether a change is statistically beneficial and SPSA for
tuning ... around 20M games").

Why we need it: our fixed-N paired tests keep landing at p = 0.15-0.35, which
answers nothing, and then I have to guess whether to spend more compute. SPRT
removes the guess. It accumulates evidence game by game and stops the moment the
log-likelihood ratio crosses an accept or reject boundary, so obvious wins and
obvious losses cost few games and only genuinely marginal changes cost many.

Hypotheses are stated in Elo, the natural scale for "is this change better":

    H0: elo difference = elo0   (null, typically 0)
    H1: elo difference = elo1   (the smallest gain worth shipping, e.g. +10)

We test on PAIRED seeds using the CRN harness, and score each seed as a
trinomial: candidate better / equal / worse. Draws (equal seeds) carry no
information about direction, which is exactly why paired testing converges so
much faster than unpaired win-rate comparison.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from paired_eval import play, build  # noqa: E402


def elo_to_p(elo):
    return 1.0 / (1.0 + 10 ** (-elo / 400.0))


def llr(wins, losses, draws, elo0, elo1):
    """Log-likelihood ratio for a paired win/loss/draw record (Wald SPRT)."""
    n = wins + losses + draws
    if n == 0 or wins + losses == 0:
        return 0.0
    # model draws as uninformative; score on the decisive subset
    p0 = elo_to_p(elo0)
    p1 = elo_to_p(elo1)
    w, l = wins, losses
    try:
        return (w * math.log(p1 / p0) + l * math.log((1 - p1) / (1 - p0)))
    except (ValueError, ZeroDivisionError):
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("baseline")
    ap.add_argument("--opponents", default="public-archaludon,public-alakazam")
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=15.0,
                    help="smallest gain worth shipping, in Elo")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--max-seeds", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=120000)
    a = ap.parse_args()

    import arena  # noqa: F401
    fa, da = build(a.candidate)
    fb, db = build(a.baseline)
    opps = [build(o) for o in a.opponents.split(",")]

    upper = math.log((1 - a.beta) / a.alpha)      # accept H1 (candidate better)
    lower = math.log(a.beta / (1 - a.alpha))      # accept H0 (no gain)
    print(f"SPRT  H0: {a.elo0:+.0f} Elo   H1: {a.elo1:+.0f} Elo   "
          f"bounds [{lower:.2f}, {upper:.2f}]")
    print(f"candidate={a.candidate}  baseline={a.baseline}")
    print(f"opponents={a.opponents}  max {a.max_seeds} seeds\n")

    w = l = d = 0
    for i in range(a.max_seeds):
        s = a.seed0 + i
        for fo, do in opps:
            ca = cb = 0
            for seat in (0, 1):
                random.seed(s)
                if seat == 0:
                    ca += int(play(s, fa, fo, da, do) == 0)
                else:
                    ca += int(play(s, fo, fa, do, da) == 1)
            for seat in (0, 1):
                random.seed(s)
                if seat == 0:
                    cb += int(play(s, fb, fo, db, do) == 0)
                else:
                    cb += int(play(s, fo, fb, do, db) == 1)
            if ca > cb:
                w += 1
            elif ca < cb:
                l += 1
            else:
                d += 1

        L = llr(w, l, d, a.elo0, a.elo1)
        if (i + 1) % 5 == 0 or L > upper or L < lower:
            print(f"  seeds {i+1:4d}  W{w} L{l} D{d}  LLR {L:+.2f}", flush=True)
        if L > upper:
            print(f"\nACCEPT H1 after {i+1} seeds: {a.candidate} is better "
                  f"than {a.baseline} by >= {a.elo1:.0f} Elo (W{w} L{l} D{d})")
            return
        if L < lower:
            print(f"\nREJECT after {i+1} seeds: no {a.elo1:.0f}+ Elo gain "
                  f"(W{w} L{l} D{d})")
            return

    L = llr(w, l, d, a.elo0, a.elo1)
    print(f"\nINCONCLUSIVE at {a.max_seeds} seeds: W{w} L{l} D{d}, LLR {L:+.2f}. "
          f"Effect is smaller than {a.elo1:.0f} Elo -- not worth shipping.")


if __name__ == "__main__":
    main()
