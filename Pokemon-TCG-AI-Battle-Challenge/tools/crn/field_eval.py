"""Evaluate an agent against the REAL ladder field, weighted by observed share.

Why this exists: the anchored-Elo experiment showed local strength explains only
38% of ladder variance (R^2=0.387, Spearman 0.60) and even INVERTS for one pair --
fork_m3000 is the weakest agent in a local round-robin yet outscores router_v12 by
103 points on the ladder. The diagnosis is not "local testing is useless"; it is
that the opponent panel (our own meta-* agents and two public notebooks) does not
resemble what the ladder actually matches us against.

So build the panel FROM the ladder. `agent/field_decks.json` holds the 12 most
common 60-card lists among 2,776 games where BOTH players were rated >=1100,
with their observed frequency. Each is piloted by the strongest policy we own for
that archetype, and the score is the SHARE-WEIGHTED win rate.

    field_00  30.6%  Munkidori/Marnie's Impidimp   -> grim policy
    field_01  12.8%  Teal Mask Ogerpon             -> ogerpon_policy
    field_04   5.8%  Alakazam                      -> fork (m3000)
    others           Dunsparce / Crustle families  -> domain_policy

Pilots are imperfect for the archetypes we lack a tuned policy for, so an absolute
number here is still not a rating. The claim being tested is narrower and
checkable: does field-weighted win rate PREDICT ladder score better than a
round-robin against toy opponents? Run `--calibrate` to fit it against the agents
whose ladder scores we have observed and see the R^2 directly.
"""
from __future__ import annotations

import importlib.util as ilu
import json
import math
import multiprocessing as mp
import time
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

FIELD = json.load(open(os.path.join(ROOT, "agent", "field_decks.json")))
_ST = os.path.join(ROOT, "scratchpad", "scrape_20260804", "elo_stage")

# agents under test: staged copies of exactly what was submitted
SUBJECTS = {
    "family_v1":  (os.path.join(_ST, "family_v1"),  [866.8]),
    "family_v2":  (os.path.join(_ST, "family_v2"),  []),
    "grim_v2":    (os.path.join(_ST, "grim_v2"),    [863.3, 734.9]),
    "fork_m3000": (os.path.join(_ST, "fork_m3000"), [762.7, 697.6]),
    "router_v12": (os.path.join(_ST, "router_v12"), [688.4, 565.6]),
    # public "probablity-v2" notebook: Mega Lucario ex, the archetype my elite
    # mining scored at 75.9% among players rated >=1150 -- the best in the pool,
    # and the one we had no competent pilot for until now.
    "prob_v2":    (os.path.join(_ST, "prob_v2"),    []),
    # the PACKAGED Ogerpon candidate: prob_v2 policy + field_01 deck, with
    # the deck.csv-overwrite line disabled. Measured as it would ship.
    "ogerpon_pkg": (os.path.join(_ST, "ogerpon_pkg"), []),
    "fv2_m300": (os.path.join(_ST, "fv2_m300"), []),
    "fv2_m1000": (os.path.join(_ST, "fv2_m1000"), []),
    "fv2_m3000": (os.path.join(_ST, "fv2_m3000"), []),
    "fv2_off": (os.path.join(_ST, "fv2_off"), []),

    "c_sharp": (os.path.join(_ST, "c_sharp"), []),
    "c_half": (os.path.join(_ST, "c_half"), []),
    "c_2x": (os.path.join(_ST, "c_2x"), []),
    "c_8x": (os.path.join(_ST, "c_8x"), []),

    "m_all_half": (os.path.join(_ST, "m_all_half"), []),
    "m_all_2x": (os.path.join(_ST, "m_all_2x"), []),
    "m_all_4x": (os.path.join(_ST, "m_all_4x"), []),
    "m_garch_2x": (os.path.join(_ST, "m_garch_2x"), []),
    "m_arch_2x": (os.path.join(_ST, "m_arch_2x"), []),
    "m_router_2x": (os.path.join(_ST, "m_router_2x"), []),

    "g_mirror3k": (os.path.join(_ST, "g_mirror3k"), []),
    "g_mirror10k": (os.path.join(_ST, "g_mirror10k"), []),
    "g_mirror20k": (os.path.join(_ST, "g_mirror20k"), []),
    "g_mirror40k": (os.path.join(_ST, "g_mirror40k"), []),

    # family_v2 with an added Grass/Ogerpon search gate. Must be measured
    # against an OGERPON OPPONENT -- head-to-head vs family_v2 itself can
    # never fire the gate, which is why four margins returned identical 61-59.
    "family_v2_base": (os.path.join(_ST, "family_v2_base"), []),
    "g_ogerpon5k": (os.path.join(_ST, "g_ogerpon5k"), []),
    "g_ogerpon10k": (os.path.join(_ST, "g_ogerpon10k"), []),
    "g_ogerpon20k": (os.path.join(_ST, "g_ogerpon20k"), []),
    "g_ogerpon40k": (os.path.join(_ST, "g_ogerpon40k"), []),

    "s_off": (os.path.join(_ST, "s_off"), []),
    "s_race": (os.path.join(_ST, "s_race"), []),
    "s_hand": (os.path.join(_ST, "s_hand"), []),
    "s_heavy": (os.path.join(_ST, "s_heavy"), []),

    # SAME Ogerpon deck, different policy driving it. All prior Ogerpon
    # numbers used prob_v2's policy, which was written for LUCARIO;
    # ogerpon_policy targets card 96 directly and was never fairly measured
    # because the cardId bug made its card branches unreachable until today.
    "og_ogerpolicy": (os.path.join(_ST, "og_ogerpolicy"), []),
    "og_domain": (os.path.join(_ST, "og_domain"), []),

    "t_hammer2": (os.path.join(_ST, "t_hammer2"), []),
    "t_judge2": (os.path.join(_ST, "t_judge2"), []),
    "t_pokegear2": (os.path.join(_ST, "t_pokegear2"), []),
    "t_icecream2": (os.path.join(_ST, "t_icecream2"), []),
    "t_teraorb2": (os.path.join(_ST, "t_teraorb2"), []),
    "t_mixed": (os.path.join(_ST, "t_mixed"), []),

    # Ogerpon + a NON-EX basic attacker. The 6% Crustle cell is a prize-race
    # loss: an all-ex deck hands over 2 prizes per KO against a healing wall.
    # A 1-prize attacker on the same Grass energy should fix that cell without
    # touching the 94% we already get on the 30.6% pool.
    "tech_bulu2": (os.path.join(_ST, "tech_bulu2"), []),
    "tech_bulu3": (os.path.join(_ST, "tech_bulu3"), []),
    "tech_snor2": (os.path.join(_ST, "tech_snor2"), []),
    "tech_bouf2": (os.path.join(_ST, "tech_bouf2"), []),
    "tech_bulu2snor1": (os.path.join(_ST, "tech_bulu2snor1"), []),
    "tech_bulu1snor1": (os.path.join(_ST, "tech_bulu1snor1"), []),

    # every field deck under ONE fixed policy, measured on the FULL panel
    # (including field_05, which crashed out of every metagame_scan run and
    # turned out to be the Ogerpon deck's 4.2% catastrophe).
    "pkg_field_00": (os.path.join(_ST, "pkg_field_00"), []),
    "pkg_field_01": (os.path.join(_ST, "pkg_field_01"), []),
    "pkg_field_02": (os.path.join(_ST, "pkg_field_02"), []),
    "pkg_field_03": (os.path.join(_ST, "pkg_field_03"), []),
    "pkg_field_04": (os.path.join(_ST, "pkg_field_04"), []),
    "pkg_field_05": (os.path.join(_ST, "pkg_field_05"), []),
    "pkg_field_06": (os.path.join(_ST, "pkg_field_06"), []),
    "pkg_field_07": (os.path.join(_ST, "pkg_field_07"), []),

}

# which policy pilots which field deck. Matched by archetype where we have a
# tuned policy; domain_policy elsewhere.
# Pilots are now MEASURED, not assumed: pilot_select.py runs a mirror round-robin
# among every agent we own on each field deck and keeps the winner. The previous
# hand-assignment used domain_policy for ~22% of the field and was beaten 9-3 by
# grim on the Dunsparce deck alone, which inflated every agent's score.
_PF = os.path.join(ROOT, "agent", "field_pilots.json")
PILOT = json.load(open(_PF)) if os.path.exists(_PF) else {
    "field_00": "grim", "field_01": "ogerpon", "field_07": "ogerpon",
    "field_04": "fork",
}


def _load_pkg(d):
    main = os.path.join(d, "main.py")
    deck = [int(x) for x in open(os.path.join(d, "deck.csv")) if x.strip()]
    before = set(sys.modules)
    sys.path.insert(0, d)
    try:
        s = ilu.spec_from_file_location("subj_" + os.path.basename(d), main)
        m = ilu.module_from_spec(s)
        s.loader.exec_module(m)
    finally:
        try:
            sys.path.remove(d)
        except ValueError:
            pass
        for k in set(sys.modules) - before:
            sys.modules.pop(k, None)
    for attr in ("TIME_BUDGET_S", "TIME_BUDGET", "SEARCH_TIME_BUDGET_S"):
        if hasattr(m, attr):
            try:
                setattr(m, attr, 1e9)
            except Exception:
                pass

    def w(o, _f=m.agent, _d=deck):
        return list(_d) if o.get("select") is None else _f(o)
    return w, deck


def _pilot_for(key, deck):
    """Best policy we own for this field archetype."""
    kind = PILOT.get(key, "domain")
    if kind == "fork":
        f, _d = _load_pkg(os.path.join(_ST, "fork_m3000"))
        return (lambda o, _f=f, _d=deck: list(_d) if o.get("select") is None else _f(o))
    if kind == "grim":
        f, _d = _load_pkg(os.path.join(_ST, "grim_v2"))
        return (lambda o, _f=f, _d=deck: list(_d) if o.get("select") is None else _f(o))
    if kind == "prob_v2":
        f, _d = _load_pkg(os.path.join(_ST, "prob_v2"))
        return (lambda o, _f=f, _d=deck: list(_d) if o.get("select") is None else _f(o))
    if kind == "ogerpon":
        import ogerpon_policy
        return (lambda o, _d=deck: list(_d) if o.get("select") is None
                else ogerpon_policy.ogerpon_agent(o, _d))
    import domain_policy
    return (lambda o, _d=deck: list(_d) if o.get("select") is None
            else domain_policy.domain_agent(o, _d))


def _cell(args, q):
    try:
        q.put(job(args))
    except Exception:
        pass


def job(args):
    subj, fkey, seeds = args
    me, mydeck = _load_pkg(SUBJECTS[subj][0])
    fdeck = FIELD[fkey]["deck"]
    opp = _pilot_for(fkey, fdeck)
    w = l = 0
    for s in seeds:
        for seat in (0, 1):
            random.seed(s)
            if seat == 0:
                r = play(s, me, opp, mydeck, fdeck)
                w += int(r == 0); l += int(r == 1)
            else:
                r = play(s, opp, me, fdeck, mydeck)
                w += int(r == 1); l += int(r == 0)
    return subj, fkey, w, l


def main():
    n = int(os.environ.get("FE_SEEDS", "24"))
    workers = int(os.environ.get("FE_WORKERS", "8"))
    keys = [k for k in FIELD][:int(os.environ.get("FE_TOPK", "8"))]
    subs = [s for s in os.environ.get("FE_SUBJECTS", ",".join(SUBJECTS)).split(",") if s]
    seeds = [762000 + i for i in range(n)]
    jobs = [(s, k, seeds) for s in subs for k in keys]
    share = {k: FIELD[k]["share"] for k in keys}
    tot_share = sum(share.values())
    print(f"field eval: {len(subs)} agents x {len(keys)} field decks "
          f"({100*tot_share:.0f}% of the >=1100 field) x {n} seeds x 2 seats", flush=True)

    # Each cell runs in its OWN process with its own result pipe. The engine can
    # abort at C++ level ("buffer full. capacity:7", std::runtime_error), which
    # kills the worker outright -- with a shared Pool that destroys the whole run,
    # as it did on the first attempt (17/20 cells done, no summary). Isolation
    # means a crash costs one cell and is recorded rather than fatal.
    ctx = mp.get_context("fork")
    res = {}
    crashed = []
    pending = list(jobs)
    running = []
    while pending or running:
        while pending and len(running) < workers:
            a = pending.pop(0)
            q = ctx.Queue()
            pr = ctx.Process(target=_cell, args=(a, q), daemon=True)
            pr.start()
            running.append((pr, q, a, time.time()))
        time.sleep(0.5)
        still = []
        for pr, q, a, t0 in running:
            out = None
            try:
                out = q.get_nowait()
            except Exception:
                out = None
            if out is not None:
                subj, fkey, w, l = out
                res[(subj, fkey)] = (w, l)
                print(f"    {subj:11s} vs {fkey} {w}-{l}", flush=True)
                pr.join(timeout=1)
            elif not pr.is_alive():
                crashed.append((a[0], a[1]))
                print(f"    {a[0]:11s} vs {a[1]} ENGINE CRASH -- cell dropped", flush=True)
            elif time.time() - t0 > float(os.environ.get("FE_CELL_TIMEOUT", "1800")):
                pr.terminate()
                crashed.append((a[0], a[1]))
                print(f"    {a[0]:11s} vs {a[1]} TIMEOUT -- cell dropped", flush=True)
            else:
                still.append((pr, q, a, t0))
        running = still
    if crashed:
        print(f"\n  dropped {len(crashed)} cells: {crashed}", flush=True)

    print("\n===== FIELD-WEIGHTED WIN RATE =====")
    fw = {}
    for s in subs:
        num = den = 0.0
        for k in keys:
            w, l = res.get((s, k), (0, 0))
            if w + l == 0:
                continue
            num += share[k] * w / (w + l)
            den += share[k]
        fw[s] = 100.0 * num / max(den, 1e-9)
        obs = SUBJECTS[s][1]
        tag = f"observed {sum(obs)/len(obs):.1f}" if obs else "CANDIDATE"
        print(f"  {s:11s} {fw[s]:5.1f}%   {tag}")

    # calibration: does this metric predict ladder score?
    xs, ys = [], []
    for s in subs:
        obs = SUBJECTS[s][1]
        if obs:
            xs.append(fw[s]); ys.append(sum(obs) / len(obs))
    if len(xs) >= 3:
        nn = len(xs); mx = sum(xs)/nn; my = sum(ys)/nn
        sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        sxx = sum((x-mx)**2 for x in xs); syy = sum((y-my)**2 for y in ys)
        r = sxy/math.sqrt(sxx*syy) if sxx*syy > 0 else 0.0
        a = sxy/sxx if sxx > 1e-9 else 0.0
        b = my - a*mx
        resid = [y-(a*x+b) for x, y in zip(xs, ys)]
        sigma = math.sqrt(sum(e*e for e in resid)/max(1, nn-2))
        print(f"\n===== CALIBRATION vs ladder =====")
        print(f"  R^2 = {r*r:.3f}   (round-robin Elo panel gave 0.387)")
        print(f"  score = {a:.1f} * fieldwin + {b:.1f}   residual sigma = {sigma:.1f}")
        for s in subs:
            if not SUBJECTS[s][1]:
                pred = a*fw[s] + b
                print(f"  {s:11s} predicted {pred:6.1f}  [{pred-2*sigma:6.1f}, {pred+2*sigma:6.1f}]")
    json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()},
              open(os.path.join(ROOT, "scratchpad", "scrape_20260804", "field_eval.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
