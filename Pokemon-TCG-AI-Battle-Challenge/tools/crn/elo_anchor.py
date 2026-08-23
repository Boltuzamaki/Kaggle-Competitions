"""Estimate a candidate's LADDER score from local play, with honest error bars.

Local paired testing answers "does A beat B", never "what will A score". This
maps the former onto the latter by regressing local strength against agents whose
ladder scores we have actually observed.

    1. round-robin every agent against every other, paired on common random
       numbers, both seats;
    2. fit Bradley-Terry ratings (logistic MLE) from the win matrix -> a local
       Elo scale;
    3. least-squares fit local Elo -> observed ladder score over the ANCHORS;
    4. read a candidate's predicted score off that line, with a prediction
       interval built from the anchors' own scatter.

The interval is the point of this tool, not the point estimate. Identical agents
have scored 128 points apart on this ladder (grim_candy_v2: 863.3 and 734.9;
router_v12: 688.4 and 565.6), so any claim of the form "this is a 950 agent"
carries at least that much uncertainty. A candidate is only credibly top-200
(cutoff 958.8) if the LOWER bound clears it, not the point estimate.

Module-name collision note: these packages each ship their own `domain_policy`,
`coalition_expert`, etc. Loading two in one process would make the second import
resolve to the first one's modules. So after loading each agent, the modules it
added are removed from sys.modules; the already-bound module globals keep working.
"""
from __future__ import annotations

import importlib.util as ilu
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "agent"), os.path.join(ROOT, "tools"),
          os.path.join(ROOT, "references", "top_rankers")):
    if p not in sys.path:
        sys.path.insert(0, p)

from paired_eval import play  # noqa: E402

# name -> (package dir, observed ladder scores). Empty list = candidate to predict.
# Anchors are the EXACT submitted tarballs, extracted, so the locally-played
# agent is byte-identical to the one that earned the observed score. Working
# directories can drift from what was packaged.
_ST = "scratchpad/scrape_20260804/elo_stage"
AGENTS = {
    "family_v1":  (f"{_ST}/family_v1",  [866.8]),
    "grim_v2":    (f"{_ST}/grim_v2",    [863.3, 734.9]),
    "fork_m3000": (f"{_ST}/fork_m3000", [762.7, 697.6]),
    "router_v12": (f"{_ST}/router_v12", [688.4, 565.6]),
    "family_v2":  (f"{_ST}/family_v2",  []),          # candidate: predict this
}


def _load_agent(pkg):
    """Load one agent package, then unregister its private modules.

    Returns (callable, deck). The callable keeps working after the unregister
    because module globals are already bound.
    """
    d = os.path.join(ROOT, pkg)
    main = os.path.join(d, "main.py")
    deck_p = os.path.join(d, "deck.csv")
    deck = [int(x) for x in open(deck_p) if x.strip()]
    before = set(sys.modules)
    sys.path.insert(0, d)
    try:
        s = ilu.spec_from_file_location("agent_" + os.path.basename(d), main)
        m = ilu.module_from_spec(s)
        s.loader.exec_module(m)
    finally:
        try:
            sys.path.remove(d)
        except ValueError:
            pass
        for k in set(sys.modules) - before:
            sys.modules.pop(k, None)

    fn = getattr(m, "agent", None)
    if fn is None:
        raise RuntimeError(f"{pkg} has no agent()")
    # unbounded time budget where the package exposes one: a deadline that binds
    # under load makes the agent nondeterministic and breaks pairing
    for attr in ("TIME_BUDGET_S", "TIME_BUDGET", "SEARCH_TIME_BUDGET_S"):
        if hasattr(m, attr):
            try:
                setattr(m, attr, 1e9)
            except Exception:
                pass

    def w(o, _f=fn, _d=deck):
        return list(_d) if o.get("select") is None else _f(o)
    return w, deck


def duel(args):
    a_name, b_name, seeds = args
    fa, da = _load_agent(AGENTS[a_name][0])
    fb, db = _load_agent(AGENTS[b_name][0])
    aw = bw = 0
    for s in seeds:
        for seat in (0, 1):
            random.seed(s)
            if seat == 0:
                r = play(s, fa, fb, da, db)
                aw += int(r == 0); bw += int(r == 1)
            else:
                r = play(s, fb, fa, db, da)
                aw += int(r == 1); bw += int(r == 0)
    return a_name, b_name, aw, bw


def bradley_terry(wins, names, iters=500):
    """Logistic MLE for latent strengths; returned on a 400/decade Elo scale."""
    r = {n: 0.0 for n in names}
    for _ in range(iters):
        for n in names:
            num = den = 0.0
            for m in names:
                if m == n:
                    continue
                w_nm = wins.get((n, m), 0)
                w_mn = wins.get((m, n), 0)
                tot = w_nm + w_mn
                if tot == 0:
                    continue
                p = 1.0 / (1.0 + math.exp(-(r[n] - r[m])))
                num += w_nm - tot * p
                den += tot * p * (1 - p)
            if den > 1e-9:
                r[n] += max(-0.5, min(0.5, num / den))
    mean = sum(r.values()) / len(r)
    return {n: (r[n] - mean) * 400.0 / math.log(10) for n in names}


def fit_and_predict(elo, anchors):
    """Weighted least squares score = a*elo + b over anchors; predict the rest."""
    xs, ys, ws = [], [], []
    for n, obs in anchors.items():
        if not obs:
            continue
        xs.append(elo[n]); ys.append(sum(obs) / len(obs)); ws.append(len(obs))
    if len(xs) < 2:
        return None
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    a = sxy / sxx if sxx > 1e-9 else 0.0
    b = my - a * mx
    resid = [y - (a * x + b) for x, y in zip(xs, ys)]
    dof = max(1, len(xs) - 2)
    sigma = math.sqrt(sum(r * r for r in resid) / dof)
    return a, b, sigma, list(zip(xs, ys, resid))


def main():
    n = int(os.environ.get("EA_SEEDS", "30"))
    workers = int(os.environ.get("EA_WORKERS", "8"))
    names = [k for k in AGENTS]
    seeds = [845000 + i for i in range(n)]
    pairs = list(itertools.combinations(names, 2))
    jobs = [(a, b, seeds[i::2]) for a, b in pairs for i in range(2)]
    print(f"round-robin: {len(names)} agents, {len(pairs)} pairs, {n} seeds x 2 seats",
          flush=True)

    wins = {}
    with mp.get_context("fork").Pool(workers, maxtasksperchild=1) as pool:
        for a, b, aw, bw in pool.imap_unordered(duel, jobs):
            wins[(a, b)] = wins.get((a, b), 0) + aw
            wins[(b, a)] = wins.get((b, a), 0) + bw
            print(f"    {a} {aw}-{bw} {b}", flush=True)

    elo = bradley_terry(wins, names)
    print("\n===== LOCAL BRADLEY-TERRY RATINGS =====")
    for nme, e in sorted(elo.items(), key=lambda kv: -kv[1]):
        obs = AGENTS[nme][1]
        tag = f"observed {'/'.join(f'{o:.1f}' for o in obs)}" if obs else "CANDIDATE"
        print(f"  {nme:12s} elo {e:+7.1f}   {tag}")

    fit = fit_and_predict(elo, {k: v[1] for k, v in AGENTS.items()})
    if fit is None:
        print("\n  too few anchors to fit")
        return
    a, b, sigma, pts = fit
    print(f"\n===== ANCHOR FIT =====")
    print(f"  score = {a:.3f} * elo + {b:.1f}   residual sigma = {sigma:.1f}")
    for x, y, r in pts:
        print(f"    anchor elo {x:+7.1f} observed {y:6.1f}  residual {r:+6.1f}")
    print("\n===== PREDICTIONS =====")
    for nme, (_pkg, obs) in AGENTS.items():
        if obs:
            continue
        pred = a * elo[nme] + b
        lo, hi = pred - 2 * sigma, pred + 2 * sigma
        verdict = "CREDIBLY 950+" if lo >= 950 else ("possible" if hi >= 950 else "NOT 950+")
        print(f"  {nme:12s} predicted {pred:6.1f}   95%-ish [{lo:6.1f}, {hi:6.1f}]  {verdict}")
    print("\n  Top-200 cutoff is 958.8. A candidate counts only if the LOWER bound clears it.")
    json.dump({"elo": elo, "wins": {f"{k[0]}|{k[1]}": v for k, v in wins.items()},
               "fit": {"a": a, "b": b, "sigma": sigma}},
              open(os.path.join(ROOT, "scratchpad", "scrape_20260804", "elo_anchor.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
