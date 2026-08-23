"""CRN battery for the guarded successor to the owned 844.4 router.

Candidate and control use the same deck and source. The control disables only
the visible terminal-mill favorable-family guard, recovering router844 behavior.
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
import domain_policy  # noqa: E402
import top_decks  # noqa: E402

MAIN = os.path.join(ROOT, "references", "top_rankers", "router_844", "extracted", "main.py")
DECK = [
    int(x)
    for x in open(
        os.path.join(ROOT, "references", "top_rankers", "router_844", "extracted", "deck.csv")
    )
    if x.strip()
]
PANELS = {
    "upper": {
        "grimmsnarl": top_decks.TD_01,
        "alakazam": top_decks.TD_02,
        "froslass_lopunny": top_decks.TD_04,
        "festival": top_decks.TD_10,
    },
    "confirm": {
        "grimmsnarl": top_decks.TD_01,
        "dragapult": top_decks.TD_03,
        "froslass_lopunny": top_decks.TD_09,
        "ogerpon": top_decks.TD_00,
    },
}


def load(name: str):
    spec = ilu.spec_from_file_location(name, MAIN)
    module = ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk(args):
    panel_name, seeds = args
    candidate = load(f"router_v4_{os.getpid()}")
    control = load(f"router_control_{os.getpid()}")
    control.facing_terminal_mill_favorable_lineage = lambda opponent: False

    def cand(obs):
        return DECK if obs.get("select") is None else candidate.agent(obs)

    def base(obs):
        return DECK if obs.get("select") is None else control.agent(obs)

    rows = {}
    for opponent_name, opponent_deck in PANELS[panel_name].items():
        if os.environ.get("RV4_OPP", "domain") == "v3":
            import arena
            import search_agent
            opponent = arena.bind_deck(search_agent.agent, opponent_deck, search_agent)
        else:
            def opponent(obs, deck=opponent_deck):
                return list(deck) if obs.get("select") is None else domain_policy.domain_agent(obs, deck)

        cw = bw = co = bo = 0
        for seed in seeds:
            counts = []
            for policy in (cand, base):
                wins = 0
                for seat in (0, 1):
                    random.seed(seed)
                    if seat == 0:
                        result = play(seed, policy, opponent, DECK, opponent_deck)
                        wins += int(result == 0)
                    else:
                        result = play(seed, opponent, policy, opponent_deck, DECK)
                        wins += int(result == 1)
                counts.append(wins)
            cw += counts[0]
            bw += counts[1]
            co += int(counts[0] > counts[1])
            bo += int(counts[1] > counts[0])
        rows[opponent_name] = (cw, bw, co, bo)
    return rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    panel = os.environ.get("RV4_PANEL", "upper")
    seed0 = int(os.environ.get("RV4_SEED0", "8444000"))
    seeds = [seed0 + i for i in range(n)]
    jobs = [(panel, seeds[i::workers]) for i in range(workers)]
    aggregate = {name: [0, 0, 0, 0] for name in PANELS[panel]}
    with mp.get_context("fork").Pool(workers) as pool:
        for result in pool.imap_unordered(chunk, jobs):
            for name, row in result.items():
                aggregate[name] = [a + b for a, b in zip(aggregate[name], row)]
    pooled = [0, 0, 0, 0]
    for name, (cw, bw, co, bo) in aggregate.items():
        pooled = [a + b for a, b in zip(pooled, (cw, bw, co, bo))]
        print(f"{name:18s} v4 {cw:4d} control {bw:4d} disc {co:3d}/{bo:3d} p={mcnemar(co, bo):.5f}")
    cw, bw, co, bo = pooled
    print(f"POOLED             v4 {cw:4d} control {bw:4d} disc {co:3d}/{bo:3d} p={mcnemar(co, bo):.6f}")


if __name__ == "__main__":
    main()
