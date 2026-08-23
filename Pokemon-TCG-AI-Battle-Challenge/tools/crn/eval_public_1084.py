"""Local deterministic evaluation of the public 1084.5 Lucario baseline.

No Kaggle submission is created or uploaded. The public notebook output contains
an obvious stray ``hi`` syntax token; its local copy is repaired before use.
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
import deep_search  # noqa: E402

PUBLIC_MAIN = os.path.join(
    ROOT, "references", "top_rankers", "baseline_1084", "output", "main.py"
)
PUBLIC_DECK = [
    int(x)
    for x in open(
        os.path.join(ROOT, "references", "top_rankers", "baseline_1084", "output", "deck.csv")
    )
    if x.strip()
]
FORK_MAIN = os.path.join(ROOT, "agent", "fork", "fork_main.py")
FORK_DECK = [int(x) for x in open(os.path.join(ROOT, "agent", "fork", "deck.csv")) if x.strip()]


def load_module(path: str, name: str):
    spec = ilu.spec_from_file_location(name, path)
    module = ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk(seeds: list[int]):
    os.chdir(os.path.dirname(PUBLIC_MAIN))
    public = load_module(PUBLIC_MAIN, f"public1084_{os.getpid()}")
    fork = load_module(FORK_MAIN, f"fork3000_{os.getpid()}")
    fork.TIME_BUDGET_S = 1e9
    deep_search.install(fork, extra_turns=0, margin=3000.0)

    def public_agent(obs):
        return PUBLIC_DECK if obs.get("select") is None else public.agent(obs)

    def fork_agent(obs):
        return FORK_DECK if obs.get("select") is None else fork.agent(obs)

    public_wins = fork_wins = public_only = fork_only = both = neither = 0
    for seed in seeds:
        counts = []
        for candidate, deck in ((public_agent, PUBLIC_DECK), (fork_agent, FORK_DECK)):
            wins = 0
            for seat in (0, 1):
                random.seed(seed)
                if seat == 0:
                    result = play(seed, candidate, fork_agent, deck, FORK_DECK)
                    wins += int(result == 0)
                else:
                    result = play(seed, fork_agent, candidate, FORK_DECK, deck)
                    wins += int(result == 1)
            counts.append(wins)
        public_wins += counts[0]
        fork_wins += counts[1]
        if counts[0] > counts[1]:
            public_only += 1
        elif counts[1] > counts[0]:
            fork_only += 1
        elif counts[0] == 2:
            both += 1
        else:
            neither += 1
    return public_wins, fork_wins, public_only, fork_only, both, neither


def main():
    seeds_n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seed0 = int(os.environ.get("P1084_SEED0", "1084000"))
    seeds = [seed0 + i for i in range(seeds_n)]
    jobs = [seeds[i::workers] for i in range(workers)]
    totals = [0] * 6
    with mp.get_context("fork").Pool(workers) as pool:
        for row in pool.imap_unordered(chunk, jobs):
            totals = [a + b for a, b in zip(totals, row)]
    pw, fw, po, fo, both, neither = totals
    games = seeds_n * 2
    print(f"public1084 {pw}/{games} vs fork-m3000 {fw}/{games}")
    print(f"seed-level public better {po}, fork better {fo}, both {both}, neither {neither}")
    print(f"McNemar p={mcnemar(po, fo):.6f} on {po + fo} discordant seeds")


if __name__ == "__main__":
    main()
