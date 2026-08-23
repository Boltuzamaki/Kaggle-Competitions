"""Pick the strongest available PILOT for each field deck, by measurement.

The field metric assumed `domain_policy` for any archetype we lack a tuned policy
for. That covers the Dunsparce family -- field_02/03/06, 22.3% of the >=1100 field
-- and it is almost certainly too weak: every agent we test scores 94-100% against
those cells, while elite players win only 58.8% with that toolbox over 374 games.
So a fifth of the metric is measuring "can you beat a bad impersonation", which
inflates every agent and taints the R^2=0.788 calibration.

The fix is to stop assuming. For each field deck, run a round-robin among the
pilots we own and keep the one that actually plays that deck best:

    domain   generic hand policy (the current default)
    prob_v2  beam+MCTS search, mostly generic leaf evaluator
    grim     the 850-rated Grimmsnarl agent
    fork     the Alakazam agent at margin 3000

A search agent with a largely generic evaluator may drive a foreign deck far
better than a generic hand policy, even without deck-specific knowledge -- that is
the hypothesis being tested, and it is cheap to check.

Writes agent/field_pilots.json for field_eval.py to consume.
"""
from __future__ import annotations

import importlib.util as ilu
import itertools
import json
import multiprocessing as mp
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "agent"), os.path.join(ROOT, "tools"),
          os.path.join(ROOT, "references", "top_rankers")):
    if p not in sys.path:
        sys.path.insert(0, p)

from paired_eval import play  # noqa: E402

_ST = os.path.join(ROOT, "scratchpad", "scrape_20260804", "elo_stage")
FIELD = json.load(open(os.path.join(ROOT, "agent", "field_decks.json")))
PKG = {"prob_v2": os.path.join(_ST, "prob_v2"),
       "grim": os.path.join(_ST, "grim_v2"),
       "fork": os.path.join(_ST, "fork_m3000")}
PILOTS = [p for p in os.environ.get("PS_PILOTS", "domain,prob_v2,grim,fork").split(",") if p]


def _mk(kind, deck):
    """Build `kind` as a pilot for `deck` (the FIELD deck, not the agent's own)."""
    if kind == "domain":
        import domain_policy
        return lambda o, _d=deck: list(_d) if o.get("select") is None \
            else domain_policy.domain_agent(o, _d)
    if kind == "ogerpon":
        import ogerpon_policy
        return lambda o, _d=deck: list(_d) if o.get("select") is None \
            else ogerpon_policy.ogerpon_agent(o, _d)
    d = PKG[kind]
    before = set(sys.modules)
    sys.path.insert(0, d)
    try:
        s = ilu.spec_from_file_location("pil_" + kind, os.path.join(d, "main.py"))
        m = ilu.module_from_spec(s)
        s.loader.exec_module(m)
    finally:
        try:
            sys.path.remove(d)
        except ValueError:
            pass
        for k in set(sys.modules) - before:
            sys.modules.pop(k, None)
    for a in ("TIME_BUDGET_S", "TIME_BUDGET", "SEARCH_TIME_BUDGET", "SEARCH_TIME_BUDGET_S"):
        if hasattr(m, a):
            try:
                setattr(m, a, 1e9 if "BUDGET_S" in a else 2.6)
            except Exception:
                pass
    return lambda o, _f=m.agent, _d=deck: list(_d) if o.get("select") is None else _f(o)


def _cell(args, q):
    try:
        q.put(job(args))
    except Exception:
        pass


def job(args):
    fkey, a, b, seeds = args
    deck = FIELD[fkey]["deck"]
    fa, fb = _mk(a, deck), _mk(b, deck)
    aw = bw = 0
    for s in seeds:
        for seat in (0, 1):
            random.seed(s)
            if seat == 0:
                r = play(s, fa, fb, deck, deck)
                aw += int(r == 0); bw += int(r == 1)
            else:
                r = play(s, fb, fa, deck, deck)
                aw += int(r == 1); bw += int(r == 0)
    return fkey, a, b, aw, bw


def main():
    n = int(os.environ.get("PS_SEEDS", "10"))
    workers = int(os.environ.get("PS_WORKERS", "8"))
    keys = [k for k in os.environ.get("PS_KEYS", "field_02,field_03,field_06").split(",") if k]
    seeds = [529000 + i for i in range(n)]
    jobs = [(k, a, b, seeds) for k in keys for a, b in itertools.combinations(PILOTS, 2)]
    print(f"pilot selection: {len(keys)} field decks x {len(PILOTS)} pilots "
          f"({len(jobs)} mirror duels) x {n} seeds x 2 seats", flush=True)

    ctx = mp.get_context("fork")
    wins = {k: {p: 0 for p in PILOTS} for k in keys}
    pend, run = list(jobs), []
    while pend or run:
        while pend and len(run) < workers:
            a = pend.pop(0)
            q = ctx.Queue()
            pr = ctx.Process(target=_cell, args=(a, q), daemon=True)
            pr.start()
            run.append((pr, q, a, time.time()))
        time.sleep(0.5)
        keep = []
        for pr, q, a, t0 in run:
            try:
                r = q.get_nowait()
            except Exception:
                r = None
            if r is not None:
                fk, x, y, xw, yw = r
                wins[fk][x] += xw; wins[fk][y] += yw
                print(f"    {fk} {x} {xw}-{yw} {y}", flush=True)
                pr.join(timeout=1)
            elif not pr.is_alive():
                print(f"    {a[0]} {a[1]} vs {a[2]} CRASH -- dropped", flush=True)
            elif time.time() - t0 > float(os.environ.get("PS_TIMEOUT", "2400")):
                pr.terminate()
                print(f"    {a[0]} {a[1]} vs {a[2]} TIMEOUT -- dropped", flush=True)
            else:
                keep.append((pr, q, a, t0))
        run = keep

    print("\n===== BEST PILOT PER FIELD DECK =====")
    best = {}
    for k in keys:
        tot = sorted(wins[k].items(), key=lambda kv: -kv[1])
        best[k] = tot[0][0]
        line = "  ".join(f"{p}:{w}" for p, w in tot)
        print(f"  {k} ({100*FIELD[k]['share']:.1f}% of field) -> {tot[0][0]:8s}   {line}")
    out = os.path.join(ROOT, "agent", "field_pilots.json")
    prev = json.load(open(out)) if os.path.exists(out) else {}
    prev.update(best)
    json.dump(prev, open(out, "w"), indent=1)
    print(f"\n  wrote {out}")
    print("  Re-run field_eval with these pilots: cells whose pilot changes")
    print("  will move, and the calibration must be refitted.")


if __name__ == "__main__":
    main()
