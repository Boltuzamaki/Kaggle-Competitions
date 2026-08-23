"""One-card deck search scored by FIELD-WEIGHTED win rate, not a single opponent.

Why this objective: field-weighted win rate against the real >=1100 ladder field
predicts observed ladder score at R^2 = 0.788 (residual sigma 57.7), versus 0.387
for a round-robin against our own agents. It is the only local number we have that
has been shown to track the ladder.

Why deck cards: the biggest recoverable pool is `field_00` -- 30.6% of the field,
where our best agent wins only 58%. That deck differs from ours by exactly two
cards (Tool Scrapper 0->1, Rare Candy 4->3), i.e. it is effectively the mirror, so
the gap is deck-and-policy skill rather than a type wall. By contrast the Ogerpon
hole (33%) is structural: Marnie's Grimmsnarl ex has a GRASS weakness and Ogerpon
is Grass, so it takes double damage and no move ordering fixes that.

Each arm changes ONE card from the base list, keeping 60 cards and <=4 copies, and
is scored on the same 8 field decks weighted by observed share. Cells run in
isolated processes because the engine can abort at C++ level and take a shared
pool down with it.

Screening only. Anything promising must then clear a disjoint-seed paired
confirmation against the base before it earns a submission.
"""
from __future__ import annotations

import importlib.util as ilu
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "agent"), os.path.join(ROOT, "tools"),
          os.path.join(ROOT, "references", "top_rankers")):
    if p not in sys.path:
        sys.path.insert(0, p)

from paired_eval import play  # noqa: E402

_ST = os.path.join(ROOT, "scratchpad", "scrape_20260804", "elo_stage")
BASE_PKG = os.path.join(_ST, os.environ.get("FDS_BASE", "family_v1"))
BASE_DECK = [int(x) for x in open(os.path.join(BASE_PKG, "deck.csv")) if x.strip()]
FIELD = json.load(open(os.path.join(ROOT, "agent", "field_decks.json")))
KEYS = [k for k in FIELD][:int(os.environ.get("FDS_TOPK", "8"))]
PILOT = {"field_00": "grim", "field_01": "ogerpon", "field_07": "ogerpon",
         "field_04": "fork"}

# Candidate one-card edits. Cards already in the list can be trimmed; a small set
# of plausible techs can be added. Restricted to singles so each arm is one
# coordinate and any effect is attributable.
ADD_POOL = [int(x) for x in os.environ.get(
    "FDS_ADD", "1137,1097,1123,1174,1122,1087,1129,1184").split(",") if x]


def _load(d, deck=None):
    main = os.path.join(d, "main.py")
    dk = deck if deck is not None else [int(x) for x in open(os.path.join(d, "deck.csv")) if x.strip()]
    before = set(sys.modules)
    sys.path.insert(0, d)
    try:
        s = ilu.spec_from_file_location("fds_" + os.path.basename(d), main)
        m = ilu.module_from_spec(s)
        s.loader.exec_module(m)
    finally:
        try:
            sys.path.remove(d)
        except ValueError:
            pass
        for k in set(sys.modules) - before:
            sys.modules.pop(k, None)
    for a in ("TIME_BUDGET_S", "TIME_BUDGET", "SEARCH_TIME_BUDGET_S"):
        if hasattr(m, a):
            try:
                setattr(m, a, 1e9)
            except Exception:
                pass

    def w(o, _f=m.agent, _d=dk):
        return list(_d) if o.get("select") is None else _f(o)
    return w


def _pilot(key, deck):
    kind = PILOT.get(key, "domain")
    if kind in ("fork", "grim"):
        pkg = os.path.join(_ST, "fork_m3000" if kind == "fork" else "grim_v2")
        return _load(pkg, deck)
    if kind == "ogerpon":
        import ogerpon_policy
        return (lambda o, _d=deck: list(_d) if o.get("select") is None
                else ogerpon_policy.ogerpon_agent(o, _d))
    import domain_policy
    return (lambda o, _d=deck: list(_d) if o.get("select") is None
            else domain_policy.domain_agent(o, _d))


def apply_edit(edit):
    c = Counter(BASE_DECK)
    rem, add = edit
    if rem is not None:
        c[rem] -= 1
        if c[rem] <= 0:
            c.pop(rem, None)
    if add is not None:
        c[add] = c.get(add, 0) + 1
    deck = []
    for k in sorted(c):
        deck += [k] * c[k]
    # The 4-copy limit does NOT apply to Basic Energy (card ids 1-8), and the base
    # list runs 10 Basic {D}. Enforcing <=4 on everything rejected every arm --
    # including the base deck itself -- and silently produced a zero-arm search.
    over = [k for k, v in c.items() if v > 4 and not (1 <= k <= 8)]
    return deck if len(deck) == 60 and not over else None


def _cell(args, q):
    try:
        q.put(job(args))
    except Exception:
        pass


def job(args):
    label, deck, seeds = args
    me = _load(BASE_PKG, deck)
    num = den = 0.0
    detail = {}
    for k in KEYS:
        fd = FIELD[k]["deck"]
        opp = _pilot(k, fd)
        w = l = 0
        for s in seeds:
            for seat in (0, 1):
                random.seed(s)
                if seat == 0:
                    r = play(s, me, opp, deck, fd)
                    w += int(r == 0); l += int(r == 1)
                else:
                    r = play(s, opp, me, fd, deck)
                    w += int(r == 1); l += int(r == 0)
        if w + l:
            sh = FIELD[k]["share"]
            num += sh * w / (w + l); den += sh
            detail[k] = (w, l)
    return label, 100.0 * num / max(den, 1e-9), detail


def main():
    n = int(os.environ.get("FDS_SEEDS", "16"))
    workers = int(os.environ.get("FDS_WORKERS", "8"))
    seeds = [431000 + i for i in range(n)]

    # A pure removal leaves 59 cards and is rejected, so every arm is a SWAP:
    # remove one copy of X, add one copy of Y. Y ranges over plausible techs plus
    # cards already in the list (a count adjustment, e.g. 3->4 Rare Candy, which is
    # exactly the edit the base agent's own deck search found).
    arms = [("base", list(BASE_DECK))]
    counts = Counter(BASE_DECK)
    removable = [c for c in sorted(counts)]
    addable = list(dict.fromkeys(
        ADD_POOL + [c for c in sorted(counts) if counts[c] < 4 or 1 <= c <= 8]))
    seen = set()
    for rem in removable:
        for add in addable:
            if rem == add:
                continue
            d = apply_edit((rem, add))
            if not d:
                continue
            key = tuple(sorted(d))
            if key in seen:
                continue
            seen.add(key)
            arms.append((f"-{rem}+{add}", d))
    cap = int(os.environ.get("FDS_MAXARMS", "24"))
    arms = arms[:cap]
    print(f"field deck search: {len(arms)} arms x {len(KEYS)} field decks x {n} seeds x 2 seats",
          flush=True)

    ctx = mp.get_context("fork")
    out = {}
    pend = [(lb, dk, seeds) for lb, dk in arms]
    run = []
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
                lb, fw, det = r
                out[lb] = fw
                print(f"    {lb:14s} field {fw:5.1f}%", flush=True)
                pr.join(timeout=1)
            elif not pr.is_alive():
                print(f"    {a[0]:14s} CRASH -- dropped", flush=True)
            elif time.time() - t0 > float(os.environ.get("FDS_TIMEOUT", "3600")):
                pr.terminate()
                print(f"    {a[0]:14s} TIMEOUT -- dropped", flush=True)
            else:
                keep.append((pr, q, a, t0))
        run = keep

    base = out.get("base")
    print("\n===== ARMS (best first) =====")
    for lb, fw in sorted(out.items(), key=lambda kv: -kv[1]):
        d = f"{fw-base:+5.1f}" if base is not None and lb != "base" else "  base"
        print(f"  {lb:14s} {fw:5.1f}%   {d}")
    if base is not None:
        print(f"\n  base = {base:.1f}%  -> predicted ladder {7.6*base+290.5:.0f}")
        print("  Screening only: any winner needs a disjoint-seed paired confirmation.")
    json.dump(out, open(os.path.join(ROOT, "scratchpad", "scrape_20260804",
                                     "field_deck_search.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
