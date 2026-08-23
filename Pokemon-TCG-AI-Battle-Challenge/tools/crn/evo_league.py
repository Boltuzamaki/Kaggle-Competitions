"""Evolution against a LEAGUE of the field, not just the mirror.

beat_stock and evo2 both optimise "beat the stock fork head-to-head". That is a
single opponent, and ratings in this game are strongly non-transitive -- our own
agents measured 68.8% against Alakazam and 25% against Archaludon. A genome can
therefore win the mirror convincingly and still lose ground on a ladder made of
many different decks.

Fitness here is two-stage, which is what makes a league affordable:

  stage 1  MIRROR, cheap. Every candidate plays the champion on a small shared
           seed set. This is a filter, not a verdict -- it costs little and
           eliminates the clearly-bad half.
  stage 2  LEAGUE. Survivors play a panel of DIFFERENT opponents, paired against
           the champion on shared seeds, so the score is "does this genome beat
           the champion against the field" rather than "against itself".

Spending the expensive league games only on stage-1 survivors is what keeps the
cost near a mirror-only run while measuring the thing we actually care about.

The panel comes from clean_opponents.json (written by determinism_scan.py) and
contains only opponents verified to return the same result for the same seed;
td-td_08 and hybs-other fail that test and would inject pure noise into every
comparison.

Promotion still requires a disjoint-seed confirmation, and shipping still requires
validate_full.py -- a tuner's own gate is not validation. That is the lesson of
champion B, which passed nine internal promotions and then lost 54-66 on fresh
seeds.
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
    sys.path.insert(0, p)

from paired_eval import play, build  # noqa: E402

FP = os.path.join(ROOT, "agent", "fork", "fork_main.py")
DECK = [int(x) for x in open(os.path.join(ROOT, "agent", "fork", "deck.csv")) if x.strip()]

POP = int(os.environ.get("EL_POP", "10"))
GENS = int(os.environ.get("EL_GENS", "400"))
MIRROR_SEEDS = int(os.environ.get("EL_MSEEDS", "10"))
LEAGUE_SEEDS = int(os.environ.get("EL_LSEEDS", "8"))
WORKERS = int(os.environ.get("EL_WORKERS", "8"))
ELITE = int(os.environ.get("EL_ELITE", "4"))
MAX_OPPS = int(os.environ.get("EL_MAXOPPS", "6"))
CKPT = os.path.join(ROOT, "scratchpad", "scrape_20260804",
                    os.environ.get("EL_CKPT", "evo_league.json"))
CAND = os.path.join(ROOT, "agent", "fork", os.environ.get("EL_CAND", "alak_w_league.json"))

_A = _B = None


def _load():
    s = ilu.spec_from_file_location("fm_el", FP)
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)
    m.TIME_BUDGET_S = 1e9       # deadline binds under load and breaks reproducibility
    return m


def _init():
    global _A, _B
    _A, _B = _load(), _load()
    import arena  # noqa: F401


def _panel():
    """Deterministic, field-representative opponents, diversified by archetype."""
    f = os.path.join(HERE, "clean_opponents.json")
    if os.path.exists(f):
        names = json.load(open(f)).get("deterministic", [])
    else:
        names = ["public-archaludon", "public-alakazam", "meta-grimmsnarl"]
    # Exclude the hybs-* and td-* families STRUCTURALLY rather than trusting the
    # per-opponent scan. Those are backed by our search agent, and the scan at 6
    # seeds passed hybs-other, which had already failed at 10 seeds -- a low-rate
    # nondeterminism that a small sample cannot see. meta-* and public-* do not
    # use that agent and passed as whole families.
    names = [n for n in names if n.startswith(("meta-", "public-"))]
    # prefer the public agents (real competitors) then spread across archetypes
    pub = [n for n in names if n.startswith("public-")]
    rest = [n for n in names if not n.startswith("public-")]
    seen, out = set(), []
    for n in pub + rest:
        arch = n.split("-", 1)[-1]
        if arch in seen:
            continue
        seen.add(arch)
        out.append(n)
        if len(out) >= MAX_OPPS:
            break
    return out


PANEL = _panel()


def _mirror(args):
    cand, ref, seeds = args
    _A.WEIGHTS.clear(); _A.WEIGHTS.update(cand)
    _B.WEIGHTS.clear(); _B.WEIGHTS.update(ref)
    a = b = 0
    for s in seeds:
        x = y = 0
        for seat in (0, 1):
            random.seed(s)
            r = play(s, _A.agent, _B.agent, DECK, DECK) if seat == 0 else \
                play(s, _B.agent, _A.agent, DECK, DECK)
            x += int(r == (0 if seat == 0 else 1))
            y += int(r == (1 if seat == 0 else 0))
        if x > y: a += 1
        elif y > x: b += 1
    return a - b


def _league(args):
    """Paired against the champion across the panel: discordant-seed margin."""
    cand, ref, seeds = args
    _A.WEIGHTS.clear(); _A.WEIGHTS.update(cand)
    _B.WEIGHTS.clear(); _B.WEIGHTS.update(ref)
    a = b = 0
    for name in PANEL:
        fo, do = build(name)
        for s in seeds:
            x = y = 0
            for seat in (0, 1):
                random.seed(s)
                x += int(play(s, _A.agent, fo, DECK, do) == 0) if seat == 0 else \
                     int(play(s, fo, _A.agent, do, DECK) == 1)
            for seat in (0, 1):
                random.seed(s)
                y += int(play(s, _B.agent, fo, DECK, do) == 0) if seat == 0 else \
                     int(play(s, fo, _B.agent, do, DECK) == 1)
            if x > y: a += 1
            elif y > x: b += 1
    return a - b


def mutate(g, rng, rate, scale, focus=None, boost=3.0):
    c = dict(g)
    for k in c:
        r = rate * boost if (focus and k in focus) else rate
        if rng.random() < min(0.95, r):
            c[k] = max(1.0, c[k] * (1.0 + rng.gauss(0, scale)))
    return c


def crossover(p1, p2, rng):
    return {k: (p1[k] if rng.random() < 0.5 else p2[k]) for k in p1}


def main():
    fm = _load()
    base = {k: float(v) for k, v in fm.WEIGHTS.items()}
    # 48 of the public genome's 69 weights are still round hand-set defaults, i.e.
    # coordinates their memetic run never moved. Bias mutation there.
    focus = [k for k, v in base.items() if float(v) % 50 == 0.0] \
        if os.environ.get("EL_FOCUS", "1") == "1" else None
    rng = random.Random(int(os.environ.get("EL_RNG", "5150")))

    champ = dict(base)
    population = [dict(base) for _ in range(POP)]
    gen0 = wins = 0
    scale = float(os.environ.get("EL_SCALE", "0.25"))
    if os.path.exists(CKPT):
        try:
            d = json.load(open(CKPT))
            champ = d["champion"]; population = d.get("population") or population
            gen0 = d.get("generation", 0); wins = d.get("promotions", 0)
            scale = d.get("scale", scale)
            print(f"resumed gen {gen0}", flush=True)
        except Exception:
            pass

    print(f"league panel ({len(PANEL)}): {', '.join(PANEL)}", flush=True)
    print(f"{len(base)} weights | pop {POP} | focus {len(focus) if focus else 0} coords | "
          f"mirror {MIRROR_SEEDS} -> league {LEAGUE_SEEDS}x{len(PANEL)}", flush=True)

    t0 = time.time()
    recent = []
    with mp.get_context("fork").Pool(WORKERS, initializer=_init) as pool:
        for gen in range(gen0 + 1, gen0 + GENS + 1):
            elites = population[:ELITE] or [champ]
            cands = []
            while len(cands) < POP:
                if len(elites) >= 2 and rng.random() < 0.4:
                    p1, p2 = rng.sample(elites, 2)
                    child = crossover(p1, p2, rng)
                else:
                    child = dict(rng.choice(elites))
                cands.append(mutate(child, rng, 0.30, scale, focus))

            # stage 1: cheap mirror filter
            ms = [700000 + gen * 149 + i for i in range(MIRROR_SEEDS)]
            m_scores = pool.map(_mirror, [(c, champ, ms) for c in cands])
            order = sorted(range(len(cands)), key=lambda i: m_scores[i], reverse=True)
            survivors = order[:max(2, len(cands) // 2)]

            # stage 2: league, only for survivors
            ls = [810000 + gen * 211 + i for i in range(LEAGUE_SEEDS)]
            l_scores = pool.map(_league, [(cands[i], champ, ls) for i in survivors])
            bi = survivors[max(range(len(survivors)), key=lambda j: l_scores[j])]
            bm = max(l_scores)
            best = cands[bi]

            st = "hold"
            need = max(3, int(0.20 * LEAGUE_SEEDS * len(PANEL)))
            if bm >= need:
                cs = [366000 + gen * 83 + i for i in range(LEAGUE_SEEDS * 3)]
                conf = _league((best, champ, cs))
                if conf >= max(3, int(0.12 * LEAGUE_SEEDS * 3 * len(PANEL))):
                    champ = best; wins += 1
                    st = f"PROMOTE (league +{bm} then +{conf})"
                    json.dump(champ, open(CAND, "w"), indent=1)
            recent.append(1 if st != "hold" else 0)
            recent = recent[-10:]
            if len(recent) == 10:
                scale = min(0.60, scale * 1.15) if sum(recent) / 10.0 > 0.2 \
                    else max(0.03, scale * 0.92)

            population = [best] + [c for i, c in enumerate(cands) if i != bi]
            print(f"  gen {gen:4d} mirror {max(m_scores):+3d} league {bm:+3d} "
                  f"scale {scale:.3f} {st} promo={wins} "
                  f"({(time.time()-t0)/60:.0f}m)", flush=True)
            json.dump({"champion": champ, "population": population[:POP],
                       "generation": gen, "promotions": wins, "scale": scale,
                       "panel": PANEL}, open(CKPT, "w"), indent=1)


if __name__ == "__main__":
    main()
