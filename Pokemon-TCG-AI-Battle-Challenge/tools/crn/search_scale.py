"""Does spending the unused episode compute on deeper search actually win games?

Graded episodes: actTimeout=0, runTimeout=2000 s, median 162 / max 248 steps.
Measured: the stock fork spends 43 ms per decision and about 6 s per episode.
Even claiming only 35% of the ceiling leaves ~700 s -- the shipped agent uses
under 1% of the compute it is allowed.

TIME_BUDGET_S=0.80 is never reached, so it is not the limiter; N_DET (number of
determinizations), K_OPP (opponent branching at ply-2) and MAX_SUBSTEPS are.

More determinizations should reduce the variance of every value estimate, which
is close to free strength IF the underlying search is sound. That "if" is the
whole question: an earlier knob sweep over this surface came back flat, and our
own scaled-damage fix made the agent WORSE because its shallow search was relying
on a compensating bias. So this is measured, not assumed.

Both scaled arms are compared against ONE shared stock arm on identical seeds, so
the stock games are played once rather than twice.
"""
from __future__ import annotations

import importlib.util as ilu
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

from paired_eval import play, build, mcnemar  # noqa: E402

FP = os.path.join(ROOT, "agent", "fork", "fork_main.py")
DECK = [int(x) for x in open(os.path.join(ROOT, "agent", "fork", "deck.csv")) if x.strip()]

STOCK_CFG = ("stock", 3, 3, 40, 0.80)
SCALED = [
    ("x2", 6, 4, 40, 4.0),
    ("x4", 12, 5, 48, 6.0),
]
OPPS = ["td-td_08", "public-archaludon", "meta-grimmsnarl"]


def _load(tag, nd, ko, sb, tb):
    s = ilu.spec_from_file_location("fm_" + tag, FP)
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)
    m.N_DET, m.K_OPP, m.MAX_SUBSTEPS, m.TIME_BUDGET_S = nd, ko, sb, tb

    def w(o):
        return list(DECK) if o.get("select") is None else m.agent(o)
    return w


def chunk(args):
    seeds, = args
    import arena  # noqa: F401
    agents = {STOCK_CFG[0]: _load(*STOCK_CFG)}
    for cfg in SCALED:
        agents[cfg[0]] = _load(*cfg)
    # per-arm: wins, and per-scaled-arm discordant counts vs stock
    res = {k: 0 for k in agents}
    disc = {cfg[0]: [0, 0] for cfg in SCALED}
    for opp in OPPS:
        fo, do = build(opp)
        for s in seeds:
            got = {}
            for tag, fn in agents.items():
                g = 0
                for seat in (0, 1):
                    random.seed(s)
                    g += int(play(s, fn, fo, DECK, do) == 0) if seat == 0 else \
                         int(play(s, fo, fn, do, DECK) == 1)
                got[tag] = g
                res[tag] += g
            for cfg in SCALED:
                t = cfg[0]
                if got[t] > got["stock"]:
                    disc[t][0] += 1
                elif got["stock"] > got[t]:
                    disc[t][1] += 1
    return res, disc


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seeds = [418000 + i for i in range(n)]
    parts = [seeds[i::workers] for i in range(workers)]
    print(f"{n} seeds x {len(OPPS)} opponents, {workers} workers; "
          f"arms: stock + {[c[0] for c in SCALED]}", flush=True)

    tot = {}
    td = {c[0]: [0, 0] for c in SCALED}
    with mp.get_context("fork").Pool(workers) as pool:
        for res, disc in pool.imap_unordered(chunk, [(p,) for p in parts]):
            for k, v in res.items():
                tot[k] = tot.get(k, 0) + v
            for k, v in disc.items():
                td[k][0] += v[0]
                td[k][1] += v[1]

    ng = n * len(OPPS) * 2
    print(f"\nstock: {tot.get('stock',0)}/{ng} wins")
    for cfg in SCALED:
        t = cfg[0]
        a, b = td[t]
        p = mcnemar(a, b)
        tag = "BETTER" if (a > b and p < 0.05) else ("worse" if b > a else "flat")
        print(f"  {t:4s} (N_DET={cfg[1]},K_OPP={cfg[2]}): {tot.get(t,0)}/{ng} wins  "
              f"disc {a}/{b}  p={p:.4f}  {tag}")


if __name__ == "__main__":
    main()
