"""Self-stacking metagame deck search: keeps testing decks until told to stop.

Rationale, from this project's own measurements: deck choice is the only lever
with a large, reproducible effect. Type matchups swing a cell from 33% to 94%,
while ~60 policy/search/weight experiments all landed inside noise. So the
question worth compute is "which 60-card list has the best profile against the
real field", not "how do we tune the policy".

Method:
  * candidate decks come from replays where BOTH players were rated >=1100,
    ranked by observed share;
  * every candidate is driven by the SAME pilot (prob_v2's search policy), so a
    difference is attributable to the deck;
  * each is scored as share-weighted win rate against the top-8 field decks, each
    of those driven by the strongest pilot we own for it (agent/field_pilots.json);
  * cells run in isolated processes -- the engine aborts at C++ level on some
    deck/pilot pairs and would otherwise take the whole run down.

It processes candidates in batches, appends every result to a JSON ledger, and
keeps going. Restarting resumes from the ledger rather than repeating work.

IMPORTANT measurement note: field_05 (Crustle/Kangaskhan) is the cell that breaks
the current leader -- an all-ex deck loses the prize race to a healing wall. It
crashes out of metagame_scan, which is why that tool reported 84-87% for a
candidate worth 80%. This pipeline includes it and treats a crashed cell as
MISSING, never as a pass.
"""
from __future__ import annotations

import importlib.util as ilu
import json
import multiprocessing as mp
import os
import random
import shutil
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
SRC = os.path.join(ROOT, "scratchpad", "scrape_20260804", "ogerpon_candidate")
LEDGER = os.path.join(ROOT, "scratchpad", "scrape_20260804", "deck_pipeline.json")

_ALL = os.path.join(ROOT, "agent", "field_decks_all.json")
_TOP = os.path.join(ROOT, "agent", "field_decks.json")
CAND_SRC = json.load(open(_ALL)) if os.path.exists(_ALL) else json.load(open(_TOP))
FIELD = json.load(open(_TOP))                       # the 8-deck opponent panel
PILOTS = json.load(open(os.path.join(ROOT, "agent", "field_pilots.json")))
PANEL = [k for k in FIELD][:8]

PKG = {"prob_v2": os.path.join(_ST, "prob_v2"), "grim": os.path.join(_ST, "grim_v2"),
       "fork": os.path.join(_ST, "fork_m3000")}
SEEDS_N = int(os.environ.get("DP_SEEDS", "16"))
WORKERS = int(os.environ.get("DP_WORKERS", "16"))
SEED0 = int(os.environ.get("DP_SEED0", "620000"))
BATCH = int(os.environ.get("DP_BATCH", "6"))


def _load(pkgdir, deck):
    before = set(sys.modules)
    sys.path.insert(0, pkgdir)
    try:
        s = ilu.spec_from_file_location("dp_" + os.path.basename(pkgdir),
                                        os.path.join(pkgdir, "main.py"))
        m = ilu.module_from_spec(s)
        s.loader.exec_module(m)
    finally:
        try:
            sys.path.remove(pkgdir)
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


def _pilot(kind, deck):
    if kind == "domain":
        import domain_policy
        return lambda o, _d=deck: list(_d) if o.get("select") is None \
            else domain_policy.domain_agent(o, _d)
    if kind == "ogerpon":
        import ogerpon_policy
        return lambda o, _d=deck: list(_d) if o.get("select") is None \
            else ogerpon_policy.ogerpon_agent(o, _d)
    return _load(PKG.get(kind, PKG["prob_v2"]), deck)


def _cell(args, q):
    try:
        q.put(job(args))
    except Exception:
        pass


def job(args):
    cand_key, cdeck, okey, seeds = args
    odeck = FIELD[okey]["deck"]
    me = _load(SRC, cdeck)                       # prob_v2 policy, candidate deck
    opp = _pilot(PILOTS.get(okey, "domain"), odeck)
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
    return cand_key, okey, w, l


def score(cells, cand_key):
    """Share-weighted win rate. Mirror counted at 50%; missing cells excluded and
    REPORTED, because a silently dropped cell is how the leader looked 7 points
    better than it is."""
    num = den = 0.0
    missing = []
    for k in PANEL:
        v = cells.get(k)
        if v is None or v[0] + v[1] == 0:
            missing.append(k)
            continue
        num += FIELD[k]["share"] * v[0] / (v[0] + v[1])
        den += FIELD[k]["share"]
    return (100.0 * num / den if den else 0.0), missing


def main():
    ledger = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {}
    keys = [k for k in CAND_SRC]
    todo = [k for k in keys if k not in ledger]
    print(f"deck pipeline: {len(keys)} candidates, {len(ledger)} already done, "
          f"{len(todo)} to run | panel {len(PANEL)} | {SEEDS_N} seeds | {WORKERS} workers",
          flush=True)
    ctx = mp.get_context("fork")

    while todo:
        batch = todo[:BATCH]
        todo = todo[BATCH:]
        seeds = [SEED0 + i for i in range(SEEDS_N)]
        jobs = [(c, CAND_SRC[c]["deck"], k, seeds) for c in batch for k in PANEL]
        cells = {c: {} for c in batch}
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
                    ck, ok, w, l = r
                    cells[ck][ok] = (w, l)
                    pr.join(timeout=1)
                elif not pr.is_alive():
                    print(f"    {a[0]} vs {a[2]} CRASH (cell missing, not counted)", flush=True)
                elif time.time() - t0 > float(os.environ.get("DP_TIMEOUT", "2400")):
                    pr.terminate()
                    print(f"    {a[0]} vs {a[2]} TIMEOUT", flush=True)
                else:
                    keep.append((pr, q, a, t0))
            run = keep

        for c in batch:
            fw, missing = score(cells[c], c)
            ledger[c] = {"field": fw, "share": CAND_SRC[c]["share"],
                         "cells": {k: list(v) for k, v in cells[c].items()},
                         "missing": missing}
            flag = f"  MISSING {missing}" if missing else ""
            print(f"  {c} share {100*CAND_SRC[c]['share']:5.2f}%  field {fw:5.1f}%{flag}",
                  flush=True)
            json.dump(ledger, open(LEDGER, "w"), indent=1)

        done = sorted(ledger.items(), key=lambda kv: -kv[1]["field"])[:5]
        print("  --- leaderboard so far ---", flush=True)
        for k, v in done:
            print(f"      {k:10s} {v['field']:5.1f}%   share {100*v['share']:5.2f}%"
                  f"{'  (incomplete)' if v['missing'] else ''}", flush=True)

    print("\nALL CANDIDATES DONE", flush=True)


if __name__ == "__main__":
    main()
