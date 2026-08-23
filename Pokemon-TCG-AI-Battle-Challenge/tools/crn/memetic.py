"""Memetic (population + local search) tuning of card-specific weights.

This replicates the MECHANISM behind the strongest public agent without using any
of its code. That agent's own comments describe it as "memetic-tuned ... (baked,
seed for wm4 evo)" over 74 card-specific weights -- i.e. successive generations of
population search over a per-card priority table, with the winning genome baked
in. It is LLM-written code plus automated tuning, not hand-crafted expertise,
which is why the approach is reproducible for our own deck.

Why plain SPSA was not enough here: our earlier runs did 60-90 iterations over 16
GENERIC weights and SPRT rejected the result. Two fixes:

  * population + elitism instead of a single point, so a bad gradient estimate
    cannot destroy progress (SPSA's noisy gradient was the likely failure mode);
  * PAIRED selection -- candidates are compared against the incumbent on shared
    seeds, which removes shuffle variance from the comparison and is far more
    sensitive than comparing raw win-rates.

Runs indefinitely with checkpointing, so it can be left for days and resumed.
Promotion still requires a disjoint-seed confirmation, and shipping still requires
an independent SPRT run -- a tuner's own gate is not validation (we learned that
when SPSA reported promote=True on weights SPRT later rejected at 200 seeds).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from paired_eval import play, build, mcnemar  # noqa: E402
import hybrid_agent  # noqa: E402
import meta_decks  # noqa: E402

MODULE = os.environ.get("MEM_MODULE", "ogerpon_policy")
DECKNAME = os.environ.get("MEM_DECK", "OTHER")
WFILE = os.environ.get("MEM_WFILE", "ogerpon_w.json")
CKPT = os.environ.get("MEM_CKPT", "memetic_ckpt.json")

policy = __import__(MODULE)
DECK = getattr(meta_decks, DECKNAME)

# Weighted opponent panel. Diversity matters more than count: tuning against two
# agents is how we overfit last time.
POOL = [("public-archaludon", 2.0), ("public-alakazam", 2.0),
        ("td-td_08", 1.5), ("td-td_00", 1.5),
        ("hybs-grimmsnarl", 1.0), ("hybs-dudunsparce", 1.0)]


def _agent_for(genome):
    def cand(obs):
        policy.W.clear()
        policy.W.update(genome)
        return hybrid_agent.hybrid_agent(obs, DECK, policy_module=policy)
    return cand


def duel(genome_a, genome_b, opps, seeds):
    """Paired: how many seeds does A win that B loses, and vice versa."""
    a_only = b_only = 0
    fa, fb = _agent_for(genome_a), _agent_for(genome_b)
    for fo, do, _w in opps:
        for s in seeds:
            ga = gb = 0
            for seat in (0, 1):
                random.seed(s)
                ga += int(play(s, fa, fo, DECK, do) == 0) if seat == 0 else \
                      int(play(s, fo, fa, do, DECK) == 1)
            for seat in (0, 1):
                random.seed(s)
                gb += int(play(s, fb, fo, DECK, do) == 0) if seat == 0 else \
                      int(play(s, fo, fb, do, DECK) == 1)
            if ga > gb:
                a_only += 1
            elif gb > ga:
                b_only += 1
    return a_only, b_only


def mutate(g, rng, rate, scale):
    child = dict(g)
    for k in child:
        if rng.random() < rate:
            child[k] = max(1.0, child[k] * (1.0 + rng.gauss(0, scale)))
    return child


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


def _duel_worker(args):
    """Run in a forked child: evaluate one candidate against the champion.

    Single-process generations cost ~22 minutes (864 games); forking one worker
    per candidate brings a generation to ~2 minutes, which is what makes a
    multi-hundred-generation run feasible at all. Each child gets its own copy of
    the ctypes engine handle via fork, so there is no shared-state hazard.
    """
    cand, champ, opp_names, seeds = args
    import arena  # noqa: F401
    opps = []
    for n in opp_names:
        fo, do = build(n)
        opps.append((fo, do, 1.0))
    return duel(cand, champ, opps, seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--pop", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--confirm-seeds", type=int, default=40)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    import arena  # noqa: F401
    opps = []
    built_names = []
    for name, w in POOL:
        try:
            fo, do = build(name)
            opps.append((fo, do, w))
            built_names.append(name)
        except SystemExit:
            print(f"  (skip {name})")
    print(f"panel: {len(opps)} opponents | module={MODULE} deck={DECKNAME}")

    base = dict(policy.W)
    rng = random.Random(20260807)
    ck_path = os.path.join(ROOT, "scratchpad", "scrape_20260804", CKPT)

    champion = dict(base)
    gen0 = 0
    if os.path.exists(ck_path):
        try:
            ck = json.load(open(ck_path))
            champion = ck["champion"]
            gen0 = ck.get("generation", 0)
            print(f"resumed from checkpoint at generation {gen0}")
        except Exception:
            pass

    print(f"{len(base)} tunable weights, population {a.pop}", flush=True)
    t0 = time.time()
    wins = 0
    for gen in range(gen0 + 1, gen0 + a.generations + 1):
        seeds = [900000 + gen * 97 + i for i in range(a.seeds)]  # fresh each gen
        # propose: mutations of the champion plus one crossover of two mutants
        cands = [mutate(champion, rng, 0.35, 0.30) for _ in range(a.pop - 1)]
        cands.append(crossover(cands[0], cands[-1], rng))

        opp_names = [n for n, _w in POOL if n in built_names]
        jobs = [(c, champion, opp_names, seeds) for c in cands]
        with mp.get_context("fork").Pool(processes=min(len(jobs), a.workers)) as pool:
            results = pool.map(_duel_worker, jobs)
        best_challenger, best_margin = None, 0
        for c, (aw, bw) in zip(cands, results):
            margin = aw - bw
            if margin > best_margin:
                best_challenger, best_margin = c, margin

        status = "hold"
        if best_challenger is not None and best_margin >= 2:
            # confirm on DISJOINT seeds before dethroning
            cs = [770000 + gen * 31 + i for i in range(a.seeds)]
            aw, bw = duel(best_challenger, champion, opps, cs)
            if aw > bw:
                champion = best_challenger
                wins += 1
                status = f"PROMOTE (+{best_margin} then +{aw-bw})"
        print(f"  gen {gen:4d}  best margin {best_margin:+3d}  {status}  "
              f"promotions={wins}  ({(time.time()-t0)/60:.0f}m)", flush=True)

        json.dump({"champion": champion, "generation": gen, "promotions": wins},
                  open(ck_path, "w"), indent=1)

    print("\nfinal confirmation vs baseline on disjoint seeds...")
    cs = [660000 + i for i in range(a.confirm_seeds)]
    aw, bw = duel(champion, base, opps, cs)
    p = mcnemar(aw, bw)
    print(f"  champion {aw} / baseline {bw} discordant seeds, p={p:.4f}")
    if aw > bw and p < 0.05:
        json.dump(champion, open(os.path.join(ROOT, "agent", WFILE), "w"), indent=1)
        print(f"  wrote agent/{WFILE} -- still needs an independent SPRT before shipping")
    else:
        print("  not promoted (needs p<0.05 on disjoint seeds)")


if __name__ == "__main__":
    main()
