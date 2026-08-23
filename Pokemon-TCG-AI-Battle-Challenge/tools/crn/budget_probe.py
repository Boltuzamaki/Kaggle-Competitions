"""How much search compute is the fork actually spending, and how much is available?

The graded episodes say `actTimeout = 0` and `runTimeout = 2000` -- there is no
per-action limit, only a 2000 s ceiling for the whole episode, across a median of
162 (max 248) decision steps.

The fork ships N_DET=3, K_OPP=3, MAX_SUBSTEPS=40, TIME_BUDGET_S=0.80. But a full
fork game runs in well under a second locally, so the 0.80 s budget is never
reached -- the search is bounded by the branching constants, not by time. Raising
TIME_BUDGET_S alone would therefore change nothing.

This measures ms/decision at several branching settings so we can size a config
that converts the unused episode headroom into real search depth, with margin.
"""
from __future__ import annotations

import importlib.util as ilu
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

from paired_eval import play, build  # noqa: E402

FP = os.path.join(ROOT, "agent", "fork", "fork_main.py")
DECK = [int(x) for x in open(os.path.join(ROOT, "agent", "fork", "deck.csv")) if x.strip()]

# worst-case decisions by OUR agent in one episode (248 steps observed, we act on
# roughly half) plus headroom
WORST_DECISIONS = 140
RUN_TIMEOUT_S = 2000.0
# only claim a fraction of the episode ceiling: the opponent and the engine also
# run inside it, and a single timeout forfeits the game outright
SAFE_FRACTION = 0.35

CONFIGS = [
    ("stock",  3,  3, 40, 0.80),
    ("x2",     6,  4, 40, 4.0),
    ("x4",    12,  5, 48, 4.0),
    ("x8",    24,  6, 56, 8.0),
]


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    import arena  # noqa: F401
    fo, do = build("td-td_08")
    print(f"budget: {RUN_TIMEOUT_S:.0f}s/episode, assume <= {WORST_DECISIONS} of our "
          f"decisions, claim {SAFE_FRACTION:.0%} -> "
          f"{RUN_TIMEOUT_S*SAFE_FRACTION/WORST_DECISIONS*1000:.0f} ms/decision available",
          flush=True)
    for tag, nd, ko, sb, tb in CONFIGS:
        s = ilu.spec_from_file_location("fm_" + tag, FP)
        m = ilu.module_from_spec(s)
        s.loader.exec_module(m)
        m.N_DET, m.K_OPP, m.MAX_SUBSTEPS, m.TIME_BUDGET_S = nd, ko, sb, tb
        calls = [0]
        tsum = [0.0]
        base = m.agent

        def wrapped(o, _b=base, _c=calls, _t=tsum):
            if o.get("select") is None:
                return list(DECK)
            t0 = time.perf_counter()
            r = _b(o)
            _t[0] += time.perf_counter() - t0
            _c[0] += 1
            return r

        t0 = time.perf_counter()
        for seed in range(770001, 770001 + n_games):
            random.seed(seed)
            play(seed, wrapped, fo, DECK, do)
        wall = (time.perf_counter() - t0) / n_games
        dec = calls[0] / n_games
        ms = 1000 * tsum[0] / max(calls[0], 1)
        proj = ms * WORST_DECISIONS / 1000.0
        verdict = "SAFE" if proj <= RUN_TIMEOUT_S * SAFE_FRACTION else "TOO SLOW"
        print(f"  {tag:6s} N_DET={nd:2d} K_OPP={ko} SUBS={sb:2d} TB={tb:.1f}  "
              f"{wall:6.2f}s/game  {dec:5.1f} dec/game  {ms:8.2f} ms/dec  "
              f"worst episode ~{proj:6.1f}s  {verdict}", flush=True)


if __name__ == "__main__":
    main()
