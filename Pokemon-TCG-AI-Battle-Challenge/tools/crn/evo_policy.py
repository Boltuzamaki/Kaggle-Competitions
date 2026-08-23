"""Evolve any policy module's weight dict against a deterministic panel.

Generalises evo2/evo_league so a NEW archetype can be tuned the same way the fork's
69 weights are. First target is lucario_policy on the elite_07 Mega Lucario list,
which elite players (>=1150) won 75.9% of 83 games with, versus 48.8% for the
Alakazam line our fork plays.

Objective is self-relative: candidate versus the CURRENT champion of the same
policy, paired across the panel. Tuning against the fork directly would be a poor
signal at this stage -- the untuned Lucario policy wins about 8% of those games,
so almost every candidate scores zero and there is no gradient to climb. Beating
your own champion gives a usable gradient from the start, and absolute strength is
measured separately by elite_pilot.py.

Panel excludes hybs-* and td-*, whose agents are not reproducible run-to-run and
would inject pure noise into every paired comparison.
"""
from __future__ import annotations

import importlib
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

MODULE = os.environ.get("EP_MODULE", "lucario_policy")
AGENT_FN = os.environ.get("EP_AGENTFN", "lucario_agent")
DECK_KEY = os.environ.get("EP_DECK", "elite_07")
POP = int(os.environ.get("EP_POP", "10"))
GENS = int(os.environ.get("EP_GENS", "400"))
SEEDS = int(os.environ.get("EP_SEEDS", "8"))
WORKERS = int(os.environ.get("EP_WORKERS", "8"))
ELITE_N = int(os.environ.get("EP_ELITE", "4"))
CKPT = os.path.join(ROOT, "scratchpad", "scrape_20260804",
                    os.environ.get("EP_CKPT", "evo_policy.json"))
CAND = os.path.join(ROOT, "agent", os.environ.get("EP_CAND", "lucario_w.json"))
PANEL = [n for n in os.environ.get(
    "EP_PANEL", "public-archaludon,public-alakazam,meta-grimmsnarl,meta-garchomp"
).split(",") if n]

DECK = json.load(open(os.path.join(ROOT, "agent", "elite_decks.json")))[DECK_KEY]["deck"]
_MOD = None


def _init():
    global _MOD
    _MOD = importlib.import_module(MODULE)
    import arena  # noqa: F401


def _agent_with(weights):
    # _duel is called both inside pool workers (which run _init) and from the
    # MAIN process for the confirmation step, where _MOD was never set. That
    # crashed the first time any candidate cleared the promotion threshold.
    global _MOD
    if _MOD is None:
        _init()
    _MOD.W.clear()
    _MOD.W.update(weights)
    fn = getattr(_MOD, AGENT_FN)

    def w(o):
        return list(DECK) if o.get("select") is None else fn(o, DECK)
    return w


def _duel(args):
    """Paired across the panel: seeds this genome wins that the champion loses."""
    cand, champ, seeds = args
    a = b = 0
    for name in PANEL:
        fo, do = build(name)
        for s in seeds:
            def run(weights):
                ag = _agent_with(weights)
                g = 0
                for seat in (0, 1):
                    random.seed(s)
                    g += int(play(s, ag, fo, DECK, do) == 0) if seat == 0 else \
                         int(play(s, fo, ag, do, DECK) == 1)
                return g
            # Counterbalance: the stock-vs-stock control shows a consistent
            # advantage to whichever arm plays SECOND. Alternating by seed parity
            # splits that evenly instead of handing it to the champion every time,
            # which would bias the tuner against ever promoting.
            if s % 2 == 0:
                x = run(cand); y = run(champ)
            else:
                y = run(champ); x = run(cand)
            if x > y: a += 1
            elif y > x: b += 1
    return a - b


def mutate(g, rng, rate, scale):
    c = dict(g)
    for k in c:
        if rng.random() < rate:
            c[k] = max(1.0, c[k] * (1.0 + rng.gauss(0, scale)))
    return c


def crossover(p1, p2, rng):
    return {k: (p1[k] if rng.random() < 0.5 else p2[k]) for k in p1}


def main():
    mod = importlib.import_module(MODULE)
    base = {k: float(v) for k, v in mod.W.items()}
    rng = random.Random(int(os.environ.get("EP_RNG", "77113")))
    champ = dict(base)
    population = [dict(base) for _ in range(POP)]
    gen0 = wins = 0
    scale = float(os.environ.get("EP_SCALE", "0.30"))
    if os.path.exists(CKPT):
        try:
            d = json.load(open(CKPT))
            champ = d["champion"]; population = d.get("population") or population
            gen0 = d.get("generation", 0); wins = d.get("promotions", 0)
            scale = d.get("scale", scale)
            print(f"resumed gen {gen0}", flush=True)
        except Exception:
            pass

    print(f"{MODULE}: {len(base)} weights | deck {DECK_KEY} | pop {POP} | "
          f"panel {len(PANEL)} | {SEEDS} seeds", flush=True)
    t0 = time.time()
    recent = []
    stall = 0
    with mp.get_context("fork").Pool(WORKERS, initializer=_init) as pool:
        for gen in range(gen0 + 1, gen0 + GENS + 1):
            elites = population[:ELITE_N] or [champ]
            cands = []
            while len(cands) < POP:
                if len(elites) >= 2 and rng.random() < 0.4:
                    p1, p2 = rng.sample(elites, 2)
                    child = crossover(p1, p2, rng)
                else:
                    child = dict(rng.choice(elites))
                cands.append(mutate(child, rng, 0.30, scale))

            seeds = [430000 + gen * 173 + i for i in range(SEEDS)]
            scores = pool.map(_duel, [(c, champ, seeds) for c in cands])
            bi = max(range(len(cands)), key=lambda i: scores[i])
            bm = scores[bi]
            best = cands[bi]

            st = "hold"
            if bm >= max(3, int(0.20 * SEEDS * len(PANEL))):
                cs = [277000 + gen * 91 + i for i in range(SEEDS * 3)]
                conf = _duel((best, champ, cs))
                if conf >= max(3, int(0.12 * SEEDS * 3 * len(PANEL))):
                    champ = best; wins += 1
                    st = f"PROMOTE (+{bm} then +{conf})"
                    json.dump(champ, open(CAND, "w"), indent=1)
            recent.append(1 if st != "hold" else 0); recent = recent[-10:]
            if len(recent) == 10:
                # The remote run annealed to the 0.03 floor after 120 generations
                # with zero promotions and then stalled: the 1/5th rule shrinks on
                # failure, which is right near an optimum but exactly wrong for a
                # policy still at ~8% strength, where the need is exploration.
                # Floor raised, and a stagnant search RESTARTS from a heavily
                # perturbed champion instead of grinding a local point.
                scale = min(0.60, scale * 1.15) if sum(recent) / 10.0 > 0.2 \
                    else max(0.12, scale * 0.92)
            stall = stall + 1 if st == "hold" else 0
            if stall >= int(os.environ.get("EP_RESTART", "40")):
                scale = 0.45
                population = [mutate(champ, rng, 0.60, 0.45) for _ in range(POP)]
                stall = 0
                print(f"  gen {gen:4d} RESTART: 40 generations without a promotion, "
                      f"re-seeding population at scale {scale}", flush=True)

            population = [best] + [c for i, c in enumerate(cands) if i != bi]
            print(f"  gen {gen:4d} margin {bm:+3d} scale {scale:.3f} {st} "
                  f"promo={wins} ({(time.time()-t0)/60:.0f}m)", flush=True)
            json.dump({"champion": champ, "population": population[:POP],
                       "generation": gen, "promotions": wins, "scale": scale},
                      open(CKPT, "w"), indent=1)


if __name__ == "__main__":
    main()
