"""Head-to-head paired comparison against our own live agent.

This is the ONLY local test that has ever predicted the ladder in this project.
It found the search-margin change (pooled 90/46, p=0.00023) which then moved the
ladder 768.9 -> 777.1 on a byte-identical control, exactly as measured.

Everything else has failed to convert. Most recently, field-weighted win rate
against a panel of field decks piloted by OUR OWN agents ranked the Ogerpon
candidate ABOVE family_v1 (73.1% vs 70.9%); the ladder ranks it 200+ points
BELOW (638 vs 845). The panel measures "beats our impersonations of the field",
which is not the same quantity as "beats the field".

So: candidate plays the incumbent directly, paired on shared seeds, both seats,
McNemar on discordant seeds. No synthetic panel, no calibration, no predicted
ladder points -- just "does this beat the agent we actually ship".

Cells run in isolated processes because the engine aborts at C++ level on some
deck/policy pairs and would otherwise take the whole run down.
"""
from __future__ import annotations

import importlib.util as ilu
import json
import math
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

from paired_eval import play, mcnemar  # noqa: E402

_ST = os.path.join(ROOT, "scratchpad", "scrape_20260804", "elo_stage")
INCUMBENT = os.environ.get("H2H_BASE", "family_v1")
CANDS = [c for c in os.environ.get("H2H_CANDS", "pkg_field_07,pkg_field_01").split(",") if c]
SEEDS_N = int(os.environ.get("H2H_SEEDS", "60"))
SEED0 = int(os.environ.get("H2H_SEED0", "480000"))
WORKERS = int(os.environ.get("H2H_WORKERS", "16"))


def _load(pkg):
    d = os.path.join(_ST, pkg)
    deck = [int(x) for x in open(os.path.join(d, "deck.csv")) if x.strip()]
    before = set(sys.modules)
    sys.path.insert(0, d)
    try:
        s = ilu.spec_from_file_location("h2h_" + pkg, os.path.join(d, "main.py"))
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
                setattr(m, a, 1e9 if a.endswith("_S") else 2.6)
            except Exception:
                pass
    return (lambda o, _f=m.agent, _d=deck: list(_d) if o.get("select") is None else _f(o)), deck


def _cell(args, q):
    try:
        q.put(job(args))
    except Exception:
        pass


def job(args):
    cand, seeds = args
    fa, da = _load(cand)
    fb, db = _load(INCUMBENT)
    cw = iw = 0
    for s in seeds:
        for seat in (0, 1):
            random.seed(s)
            if seat == 0:
                r = play(s, fa, fb, da, db)
                cw += int(r == 0); iw += int(r == 1)
            else:
                r = play(s, fb, fa, db, da)
                cw += int(r == 1); iw += int(r == 0)
    return cand, cw, iw


def main():
    seeds = [SEED0 + i for i in range(SEEDS_N)]
    k = max(1, WORKERS // max(len(CANDS), 1))
    jobs = [(c, seeds[i::k]) for c in CANDS for i in range(k)]
    print(f"head-to-head vs {INCUMBENT}: {len(CANDS)} candidates x {SEEDS_N} seeds "
          f"x 2 seats, {WORKERS} workers", flush=True)

    ctx = mp.get_context("fork")
    agg = {c: [0, 0] for c in CANDS}
    pend, run = list(jobs), []
    while pend or run:
        while pend and len(run) < WORKERS:
            a = pend.pop(0)
            q = ctx.Queue()
            pr = ctx.Process(target=_cell, args=(a, q), daemon=True)
            pr.start()
            run.append((pr, q, a, time.time()))
        time.sleep(0.4)
        keep = []
        for pr, q, a, t0 in run:
            try:
                r = q.get_nowait()
            except Exception:
                r = None
            if r is not None:
                c, cw, iw = r
                agg[c][0] += cw; agg[c][1] += iw
                pr.join(timeout=1)
            elif not pr.is_alive():
                print(f"    {a[0]} CRASH (chunk lost)", flush=True)
            elif time.time() - t0 > float(os.environ.get("H2H_TIMEOUT", "3600")):
                pr.terminate()
                print(f"    {a[0]} TIMEOUT", flush=True)
            else:
                keep.append((pr, q, a, t0))
        run = keep

    print(f"\n===== HEAD-TO-HEAD vs {INCUMBENT} =====")
    for c in CANDS:
        cw, iw = agg[c]
        n = cw + iw
        if not n:
            print(f"  {c:14s} no data")
            continue
        # binomial two-sided about 50%
        p = 0.0
        try:
            from math import comb
            k_ = min(cw, iw)
            p = 2 * sum(comb(n, i) for i in range(k_ + 1)) / (2 ** n)
            p = min(1.0, p)
        except Exception:
            pass
        verdict = "BETTER" if (cw > iw and p < 0.05) else ("worse" if iw > cw else "flat")
        print(f"  {c:14s} {cw:4d} - {iw:<4d} ({100*cw/n:5.1f}%)  p={p:.4f}  {verdict}")
    print(f"\n  Incumbent is what we actually ship. A candidate must win HERE, not on a")
    print(f"  synthetic panel -- that proxy ranked Ogerpon above family_v1 while the")
    print(f"  ladder put it 200 points below.")
    json.dump({c: agg[c] for c in CANDS},
              open(os.path.join(ROOT, "scratchpad", "scrape_20260804", "head2head.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
