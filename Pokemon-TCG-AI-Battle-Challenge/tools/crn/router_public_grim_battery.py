"""Compare guarded router v4 with exact router844 against public Grim policy.

The public policy is used only as a strong, stateful opponent.  It is reset
before every game, and both router arms receive the same seeds and seats.
"""
from __future__ import annotations

import importlib.util as ilu
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (HERE, os.path.join(ROOT, "agent"), os.path.join(ROOT, "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

from paired_eval import mcnemar, play  # noqa: E402

ROUTER_MAIN = os.path.join(
    ROOT, "references", "top_rankers", "router_844", "extracted", "main.py"
)
ROUTER_DECK = [
    int(x)
    for x in open(
        os.path.join(
            ROOT, "references", "top_rankers", "router_844", "extracted", "deck.csv"
        )
    )
    if x.strip()
]
GRIM_DIR = os.path.join(
    ROOT, "references", "top_rankers", "grim_control", "extracted"
)
GRIM_MAIN = os.path.join(GRIM_DIR, "main.py")
GRIM_DECK = [int(x) for x in open(os.path.join(GRIM_DIR, "deck.csv")) if x.strip()]


def load(name: str, path: str):
    spec = ilu.spec_from_file_location(name, path)
    module = ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk(seeds):
    candidate = load(f"router_v4_{os.getpid()}", ROUTER_MAIN)
    control = load(f"router_control_{os.getpid()}", ROUTER_MAIN)
    control.facing_terminal_mill_favorable_lineage = lambda opponent: False

    # The Grim package uses sibling absolute imports.
    if GRIM_DIR not in sys.path:
        sys.path.insert(0, GRIM_DIR)
    grim = load(f"public_grim_{os.getpid()}", GRIM_MAIN)

    def cand(obs):
        return ROUTER_DECK if obs.get("select") is None else candidate.agent(obs)

    def base(obs):
        return ROUTER_DECK if obs.get("select") is None else control.agent(obs)

    def opponent(obs):
        return GRIM_DECK if obs.get("select") is None else grim.agent(obs)

    cw = bw = co = bo = 0
    per_seat = [[0, 0], [0, 0]]
    for seed in seeds:
        counts = []
        for policy_index, policy in enumerate((cand, base)):
            wins = 0
            for seat in (0, 1):
                grim._reset()
                random.seed(seed)
                if seat == 0:
                    result = play(seed, policy, opponent, ROUTER_DECK, GRIM_DECK)
                    won = int(result == 0)
                else:
                    result = play(seed, opponent, policy, GRIM_DECK, ROUTER_DECK)
                    won = int(result == 1)
                wins += won
                per_seat[policy_index][seat] += won
            counts.append(wins)
        cw += counts[0]
        bw += counts[1]
        co += int(counts[0] > counts[1])
        bo += int(counts[1] > counts[0])
    return cw, bw, co, bo, per_seat


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    seed0 = int(os.environ.get("RPG_SEED0", "202608080"))
    seeds = [seed0 + i for i in range(n)]
    jobs = [seeds[i::workers] for i in range(workers)]
    total = [0, 0, 0, 0]
    seats = [[0, 0], [0, 0]]
    with mp.get_context("fork").Pool(workers) as pool:
        for cw, bw, co, bo, per_seat in pool.imap_unordered(chunk, jobs):
            total = [a + b for a, b in zip(total, (cw, bw, co, bo))]
            for policy in (0, 1):
                for seat in (0, 1):
                    seats[policy][seat] += per_seat[policy][seat]
    cw, bw, co, bo = total
    print(f"guarded {cw}/{2*n} control {bw}/{2*n}")
    print(f"discordant {co}/{bo} p={mcnemar(co, bo):.6f}")
    print(f"seat0 guarded/control {seats[0][0]}/{seats[1][0]}")
    print(f"seat1 guarded/control {seats[0][1]}/{seats[1][1]}")


if __name__ == "__main__":
    main()
