"""Which DECK in the real metagame has the best field-weighted profile?

Every agent we own has exactly one structural hole, and each hole is a type
weakness rather than a tuning failure:

    family_v1 (Darkness)  loses Ogerpon 33%   -- Grass doubles Darkness
    prob_v2   (Fighting)  loses Munkidori 46% -- Psychic doubles Fighting

You cannot tune out a 2x weakness, so the question is whether some deck in the
actual field has a profile with no such hole. That was previously untestable: a
strong deck needs a competent pilot and ours scored 4-8% on foreign decks.

pilot_select.py removed the blocker -- `grim` won 4 of 8 field decks as a pilot,
beating each deck's own specialist, so it transfers well enough to drive an
arbitrary list. This scans every field deck under that single fixed pilot and
scores each by share-weighted win rate against the same 8-deck field.

Holding the pilot constant is the point: differences are attributable to the DECK,
which is the lever with the largest historical effect in this project (+112).

Mirror cells (deck playing itself) are skipped -- they are 50% by construction and
would flatten the comparison.
"""
from __future__ import annotations

import importlib.util as ilu
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
PILOTS = json.load(open(os.path.join(ROOT, "agent", "field_pilots.json")))
PKG = {"prob_v2": os.path.join(_ST, "prob_v2"), "grim": os.path.join(_ST, "grim_v2"),
       "fork": os.path.join(_ST, "fork_m3000")}
SUBJECT_PILOT = os.environ.get("MG_PILOT", "grim")
KEYS = [k for k in FIELD][:int(os.environ.get("MG_TOPK", "8"))]
CANDS = [k for k in FIELD][:int(os.environ.get("MG_CANDS", "10"))]


def _mk(kind, deck):
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
        s = ilu.spec_from_file_location("mg_" + kind, os.path.join(d, "main.py"))
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
    return lambda o, _f=m.agent, _d=deck: list(_d) if o.get("select") is None else _f(o)


def _cell(args, q):
    try:
        q.put(job(args))
    except Exception:
        pass


def job(args):
    cand, okey, seeds = args
    cdeck = FIELD[cand]["deck"]
    odeck = FIELD[okey]["deck"]
    me = _mk(SUBJECT_PILOT, cdeck)
    opp = _mk(PILOTS.get(okey, "domain"), odeck)
    w = l = 0
    for s in seeds:
        for seat in (0, 1):
            random.seed(s)
            if seat == 0:
                r = play(s, me, opp, cdeck, odeck)
                w += int(r == 0); l += int(r == 1)
            else:
                r = play(s, opp, me, odeck, cdeck)
                w += int(r == 1); l += int(r == 0)
    return cand, okey, w, l


def main():
    n = int(os.environ.get("MG_SEEDS", "12"))
    workers = int(os.environ.get("MG_WORKERS", "8"))
    seeds = [int(os.environ.get("MG_SEED0", "674000")) + i for i in range(n)]
    jobs = [(c, k, seeds) for c in CANDS for k in KEYS if c != k]
    print(f"metagame scan: {len(CANDS)} candidate decks x {len(KEYS)} field opponents "
          f"(pilot={SUBJECT_PILOT}) x {n} seeds x 2 seats = {len(jobs)} cells", flush=True)

    ctx = mp.get_context("fork")
    res, pend, run = {}, list(jobs), []
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
                c, k, w, l = r
                res[(c, k)] = (w, l)
                pr.join(timeout=1)
            elif not pr.is_alive():
                print(f"    {a[0]} vs {a[1]} CRASH", flush=True)
            elif time.time() - t0 > float(os.environ.get("MG_TIMEOUT", "2400")):
                pr.terminate()
                print(f"    {a[0]} vs {a[1]} TIMEOUT", flush=True)
            else:
                keep.append((pr, q, a, t0))
        run = keep

    print("\n===== DECK PROFILES (field-weighted, one fixed pilot) =====")
    rows = []
    for c in CANDS:
        num = den = 0.0
        worst = (1.0, None)
        for k in KEYS:
            if c == k:
                continue
            v = res.get((c, k))
            if not v or v[0] + v[1] == 0:
                continue
            wr = v[0] / (v[0] + v[1])
            sh = FIELD[k]["share"]
            num += sh * wr; den += sh
            if wr < worst[0]:
                worst = (wr, k)
        if den:
            rows.append((100 * num / den, c, worst))
    for fw, c, (wr, wk) in sorted(rows, reverse=True):
        print(f"  {c} share {100*FIELD[c]['share']:4.1f}%   field-weighted {fw:5.1f}%"
              f"   worst matchup {wk} {100*wr:3.0f}%")
    json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()},
              open(os.path.join(ROOT, "scratchpad", "scrape_20260804",
                                "metagame_scan.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
