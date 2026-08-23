"""Population evolutionary search over the fork's 69 weights, with racing.

beat_stock.py is a (1+lambda) hill-climber: one champion, lambda isotropic
mutants, best survivor promoted. Three limitations motivate this rewrite.

  no diversity      a single champion cannot escape a local optimum, and 69
                    correlated weights almost certainly have many. Here a
                    POPULATION is carried forward and recombined.
  fixed step size   beat_stock mutates every weight at rate 0.30 / scale 0.25
                    forever. This adapts the scale with the 1/5th success rule:
                    step up while progress is frequent, shrink when it stalls.
  uniform sampling  every candidate got the same 96 games, including hopeless
                    ones. SUCCESSIVE HALVING plays a cheap first round, keeps the
                    top half, and re-plays survivors with more games. Under a
                    noisy objective that buys far more resolution per CPU-hour
                    than spreading games evenly.

The objective is the paired mirror against the CURRENT champion, on common
random numbers -- verified reproducible, so a difference is signal rather than
shuffle luck.

Anti-overfitting is the lesson of champion B, which passed nine internal
promotions and then lost 54-66 on fresh seeds: 69 dimensions judged on 16 games
means a mutation looks better by chance roughly half the time. Every generation
here draws FRESH seeds, and dethroning requires a disjoint confirmation at 3x the
sample. A genome that survives all of that still has to clear validate_full.py on
both the mirror and a diverse field before it is allowed near a submission.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import sys
import time
import importlib.util as ilu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from paired_eval import play  # noqa: E402

FP = os.path.join(ROOT, "agent", "fork", "fork_main.py")
DECK = [int(x) for x in open(os.path.join(ROOT, "agent", "fork", "deck.csv")) if x.strip()]

POP = int(os.environ.get("EV_POP", "12"))
GENS = int(os.environ.get("EV_GENS", "300"))
BASE_SEEDS = int(os.environ.get("EV_SEEDS", "12"))     # round-1 seeds per candidate
WORKERS = int(os.environ.get("EV_WORKERS", "8"))
ELITE = int(os.environ.get("EV_ELITE", "4"))
CKPT = os.path.join(ROOT, "scratchpad", "scrape_20260804",
                    os.environ.get("EV_CKPT", "evo2.json"))
CAND = os.path.join(ROOT, "agent", "fork", os.environ.get("EV_CAND", "alak_w_evo2.json"))

_A = _B = None


def _init():
    global _A, _B
    _A, _B = _load(), _load()


def _load():
    s = ilu.spec_from_file_location("fm_evo", FP)
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)
    m.TIME_BUDGET_S = 1e9      # the 0.8s deadline binds under load and destroys
                               # reproducibility; unbounded keeps duels paired
    # Tune at the margin we actually SHIP. The submitted agent uses 3000, not the
    # stock 500 (validated: pooled 90/46, p=0.00023), and the weights interact
    # with how often the search is allowed to override -- optimising a genome at
    # 500 would tune a configuration we no longer run.
    mg = os.environ.get("EV_MARGIN")
    if mg:
        import deep_search
        deep_search.install(m, extra_turns=0, margin=float(mg))
    return m


def _duel(args):
    """Paired mirror: candidate vs reference on shared seeds. Returns (a_only, b_only)."""
    # Runs both inside pool workers (which call _init) and in the MAIN process for
    # the promotion-confirmation duel, where _A/_B were never set. Identical bug
    # to the one already fixed in evo_policy and not backported here -- it crashed
    # at generation 27, i.e. the first time a candidate cleared the racing gate,
    # so the failure only fires on success.
    global _A, _B
    if _A is None or _B is None:
        _init()
    cand, ref, seeds = args
    _A.WEIGHTS.clear(); _A.WEIGHTS.update(cand)
    _B.WEIGHTS.clear(); _B.WEIGHTS.update(ref)
    a = b = 0
    for s in seeds:
        # Counterbalance seat order by seed parity. The stock-vs-stock control
        # shows a consistent edge to whichever side is played second; without
        # this it lands on the champion every time and suppresses promotions.
        first_is_cand = (s % 2 == 0)
        x = y = 0
        for seat in (0, 1):
            random.seed(s)
            p0, p1 = (_A.agent, _B.agent) if first_is_cand else (_B.agent, _A.agent)
            r = play(s, p0, p1, DECK, DECK) if seat == 0 else \
                play(s, p1, p0, DECK, DECK)
            cand_won = (r == (0 if seat == 0 else 1)) == first_is_cand
            x += int(cand_won)
            y += int(not cand_won)
        if x > y: a += 1
        elif y > x: b += 1
    return a, b


# The public genome's values fall into two groups: irregular numbers the memetic
# search actually produced (604, 3923, 6993, 38066) and round hand-set defaults it
# never moved (20000, 30000, 18900). Only 21 of 69 were moved, so 48 coordinates
# are demonstrably unexplored by the run that reached ~950. EV_FOCUS=untouched
# concentrates mutation there -- searching where their optimiser did not go,
# rather than re-deriving the part it already converged.
def _focus_keys(base):
    return [k for k, v in base.items() if float(v) % 50 == 0.0]


def mutate(g, rng, rate, scale, focus=None, focus_boost=3.0):
    c = dict(g)
    for k in c:
        r = rate * focus_boost if (focus and k in focus) else rate
        if rng.random() < min(0.95, r):
            c[k] = max(1.0, c[k] * (1.0 + rng.gauss(0, scale)))
    return c


def crossover(p1, p2, rng):
    """Uniform crossover -- recombines whole coordinates, which is what a
    hill-climber can never do."""
    return {k: (p1[k] if rng.random() < 0.5 else p2[k]) for k in p1}


def race(pool, cands, champ, gen, rounds=(1, 3)):
    """Successive halving: cheap round over everyone, then more games on the top half."""
    alive = list(range(len(cands)))
    scores = {i: 0 for i in alive}
    for r, mult in enumerate(rounds):
        ns = BASE_SEEDS * mult
        seeds = [880000 + gen * 131 + r * 977 + i for i in range(ns)]
        res = pool.map(_duel, [(cands[i], champ, seeds) for i in alive])
        for i, (a, b) in zip(alive, res):
            scores[i] = a - b
        if r < len(rounds) - 1:
            alive.sort(key=lambda i: scores[i], reverse=True)
            alive = alive[:max(2, len(alive) // 2)]
    best = max(alive, key=lambda i: scores[i])
    return best, scores[best], len(alive)


def main():
    fm = _load()
    base = {k: float(v) for k, v in fm.WEIGHTS.items()}
    rng = random.Random(int(os.environ.get("EV_RNG", "20260808")))
    focus = _focus_keys(base) if os.environ.get("EV_FOCUS") == "untouched" else None
    if focus:
        print(f"focusing mutation on {len(focus)} coordinates their search never moved",
              flush=True)

    champ = dict(base)
    population = [dict(base) for _ in range(POP)]
    gen0 = 0
    wins = 0
    scale = float(os.environ.get("EV_SCALE", "0.25"))
    if os.path.exists(CKPT):
        try:
            d = json.load(open(CKPT))
            champ = d["champion"]
            population = d.get("population") or [dict(champ) for _ in range(POP)]
            gen0 = d.get("generation", 0)
            wins = d.get("promotions", 0)
            scale = d.get("scale", scale)
            print(f"resumed gen {gen0}, scale {scale:.3f}", flush=True)
        except Exception:
            pass

    print(f"{len(base)} weights | pop {POP} | elite {ELITE} | racing "
          f"{BASE_SEEDS}->{BASE_SEEDS*3} seeds | objective = beat champion",
          flush=True)
    t0 = time.time()
    recent = []
    stall = 0
    with mp.get_context("fork").Pool(WORKERS, initializer=_init) as pool:
        for gen in range(gen0 + 1, gen0 + GENS + 1):
            # offspring: mutations of elites plus recombinations of elite pairs
            elites = population[:ELITE] if len(population) >= ELITE else population
            cands = []
            while len(cands) < POP:
                if len(elites) >= 2 and rng.random() < 0.4:
                    p1, p2 = rng.sample(elites, 2)
                    child = crossover(p1, p2, rng)
                else:
                    child = dict(rng.choice(elites))
                cands.append(mutate(child, rng, 0.30, scale, focus))

            bi, bm, n_alive = race(pool, cands, champ, gen)
            best = cands[bi]

            st = "hold"
            # Screen loosely, confirm strictly. The old racing gate demanded a
            # margin of 9 in 36 seeds (~70/30); nine generations passed without a
            # single candidate reaching confirmation, so a real but modest edge
            # (55/45 scores ~+4) was being discarded unseen. The 9x disjoint
            # confirmation below is the real filter and is what champion B failed.
            if bm >= max(3, int(0.15 * BASE_SEEDS * 3)):
                cs = [455000 + gen * 67 + i for i in range(BASE_SEEDS * 9)]
                a, b = _duel((best, champ, cs))
                if a >= b + max(3, int(0.15 * BASE_SEEDS * 9)):
                    champ = best
                    wins += 1
                    st = f"PROMOTE (+{bm} then +{a-b})"
                    json.dump(champ, open(CAND, "w"), indent=1)
            recent.append(1 if st != "hold" else 0)
            recent = recent[-10:]

            # 1/5th success rule: grow the step while progress is frequent,
            # shrink it when the search stalls.
            if len(recent) == 10:
                # Same fix already applied to evo_policy but not backported here:
                # the 1/5th rule shrinks on failure, so a run with no promotions
                # anneals to the floor and becomes an ever-more-local hill
                # climber. That is what stalled the Kaggle Lucario run at 120
                # generations. Floor raised, plus a restart when stagnant.
                rate_ok = sum(recent) / 10.0
                scale = min(0.60, scale * 1.15) if rate_ok > 0.2 else max(0.12, scale * 0.92)
            stall = stall + 1 if st == "hold" else 0
            if stall >= int(os.environ.get("EV_RESTART", "35")):
                scale = 0.45
                population = [mutate(champ, rng, 0.60, 0.45, focus) for _ in range(POP)]
                stall = 0
                print(f"  gen {gen:4d} RESTART: {35} generations without a promotion, "
                      f"re-seeding population at scale {scale}", flush=True)

            population = [best] + [c for i, c in enumerate(cands) if i != bi]
            print(f"  gen {gen:4d} margin {bm:+3d} survivors {n_alive} "
                  f"scale {scale:.3f} {st} promo={wins} "
                  f"({(time.time()-t0)/60:.0f}m)", flush=True)
            json.dump({"champion": champ, "population": population[:POP],
                       "generation": gen, "promotions": wins, "scale": scale},
                      open(CKPT, "w"), indent=1)


if __name__ == "__main__":
    main()
