"""Is the fork reproducible run-to-run, and does our depth-0 rebuild match it?

`_search_decide` is bounded by `deadline = t0 + TIME_BUDGET_S` and checks
`time.monotonic()` inside its loops. If that deadline is ever reached the number
of determinizations actually completed depends on machine load, which would make
the agent nondeterministic and make any "identical rebuild" check meaningless.

Run 1 vs run 2 of the SAME stock agent answers that. Only if stock reproduces
itself does a depth-0 mismatch indicate a real transcription error in
deep_search.install().
"""
from __future__ import annotations

import importlib.util as ilu
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

from paired_eval import play, build  # noqa: E402
import deep_search  # noqa: E402

FP = os.path.join(ROOT, "agent", "fork", "fork_main.py")
DECK = [int(x) for x in open(os.path.join(ROOT, "agent", "fork", "deck.csv")) if x.strip()]


def arm(tag, depth):
    s = ilu.spec_from_file_location("fm_" + tag, FP)
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)
    if depth is not None:
        deep_search.install(m, extra_turns=depth)

    def w(o, _m=m):
        return list(DECK) if o.get("select") is None else _m.agent(o)
    return w


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    import arena  # noqa: F401
    fo, do = build("td-td_08")
    seeds = [881001 + i for i in range(n)]
    runs = {}
    for tag, depth in (("stock_1", None), ("stock_2", None), ("depth0", 0)):
        out = []
        a = arm(tag, depth)
        for s in seeds:
            random.seed(s)
            out.append(play(s, a, fo, DECK, do))
        runs[tag] = out
        print(f"  {tag:8s} {out}", flush=True)
    same = runs["stock_1"] == runs["stock_2"]
    ctrl = runs["stock_1"] == runs["depth0"]
    print(f"\n  stock reproduces itself : {same}")
    print(f"  depth0 matches stock    : {ctrl}")
    if not same:
        print("  => agent is NOT deterministic (time budget binds under load);")
        print("     a depth0 mismatch proves nothing. Compare statistically instead.")
    elif not ctrl:
        print("  => stock IS deterministic but depth0 differs: REAL BUG in the rebuild.")
    else:
        print("  => rebuild is faithful; depth differences are attributable to depth.")


if __name__ == "__main__":
    main()
