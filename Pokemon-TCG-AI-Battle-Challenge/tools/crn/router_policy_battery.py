"""Policy-only variants on top of the confirmed guarded router successor."""
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
DECK = [int(x) for x in open(os.path.join(os.path.dirname(MAIN), "deck.csv")) if x.strip()]
PANEL = {
    "grimmsnarl": top_decks.TD_01,
    "alakazam": top_decks.TD_02,
    "froslass_lopunny": top_decks.TD_04,
    "festival": top_decks.TD_10,
}
ARMS = [a for a in os.environ.get("RPB_ARMS", "control,nw_damage,nw_favorable,nw_ready").split(",") if a]


def load(name: str, arm: str):
    spec = ilu.spec_from_file_location(name, MAIN)
    module = ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.should_wall_mode

    def damage_family(opponent) -> bool:
        return bool(module.opponent_visible_ids(opponent) & {104, 112, 646, 647, 648, 860, 861})

    if arm == "nw_damage":
        module.should_wall_mode = lambda me, opponent, state: (
            False if damage_family(opponent) else original(me, opponent, state)
        )
    elif arm == "nw_favorable":
        module.should_wall_mode = lambda me, opponent, state: (
            False if module.facing_terminal_mill_favorable_lineage(opponent)
            else original(me, opponent, state)
        )
    elif arm == "nw_ready":
        module.should_wall_mode = lambda me, opponent, state: (
            False
            if module.facing_terminal_mill_favorable_lineage(opponent) and module.has_ready_tusk(me)
            else original(me, opponent, state)
        )
    return module


def chunk(args):
    arm, seeds = args
    candidate = load(f"rpb_{arm}_{os.getpid()}", arm)
    control = load(f"rpb_control_{arm}_{os.getpid()}", "control")

    def cand(obs):
        return DECK if obs.get("select") is None else candidate.agent(obs)

    def base(obs):
        return DECK if obs.get("select") is None else control.agent(obs)

    result = {}
    for name, opponent_deck in PANEL.items():
        def opponent(obs, deck=opponent_deck):
            return list(deck) if obs.get("select") is None else domain_policy.domain_agent(obs, deck)
        row = [0, 0, 0, 0]
        for seed in seeds:
            counts = []
            for policy in (cand, base):
                wins = 0
                for seat in (0, 1):
                    random.seed(seed)
                    if seat == 0:
                        winner = play(seed, policy, opponent, DECK, opponent_deck)
                        wins += int(winner == 0)
                    else:
                        winner = play(seed, opponent, policy, opponent_deck, DECK)
                        wins += int(winner == 1)
                counts.append(wins)
            row[0] += counts[0]; row[1] += counts[1]
            row[2] += int(counts[0] > counts[1]); row[3] += int(counts[1] > counts[0])
        result[name] = row
    return arm, result


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seed0 = int(os.environ.get("RPB_SEED0", "15000000"))
    chunks = max(1, workers // len(ARMS))
    seeds = [seed0 + i for i in range(n)]
    jobs = [(arm, seeds[i::chunks]) for arm in ARMS for i in range(chunks)]
    totals = {arm: {name: [0, 0, 0, 0] for name in PANEL} for arm in ARMS}
    with mp.get_context("fork").Pool(workers) as pool:
        for arm, rows in pool.imap_unordered(chunk, jobs):
            for name, row in rows.items():
                totals[arm][name] = [a + b for a, b in zip(totals[arm][name], row)]
    for arm in ARMS:
        pooled = [sum(totals[arm][name][i] for name in PANEL) for i in range(4)]
        print(f"{arm:12s} {pooled[0]:4d}/{pooled[1]:4d} disc {pooled[2]:3d}/{pooled[3]:3d} "
              f"p={mcnemar(pooled[2], pooled[3]):.5f}")
        print(" " * 13 + " ".join(
            f"{name}={row[0]-row[1]:+d}({row[2]}/{row[3]})"
            for name, row in totals[arm].items()
        ))


if __name__ == "__main__":
    main()
