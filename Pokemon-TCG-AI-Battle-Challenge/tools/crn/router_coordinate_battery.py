"""Independent causal coordinate screens for guarded router constants."""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (HERE, os.path.join(ROOT, "agent"), os.path.join(ROOT, "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

from paired_eval import mcnemar, play  # noqa: E402
import domain_policy  # noqa: E402
import top_decks  # noqa: E402

MAIN = os.path.join(ROOT, "references", "top_rankers", "router_844", "extracted", "main.py")
SOURCE = open(MAIN, encoding="utf-8").read()
DECK = [int(x) for x in open(os.path.join(os.path.dirname(MAIN), "deck.csv")) if x.strip()]
PANEL = {
    "grimmsnarl": top_decks.TD_01,
    "alakazam": top_decks.TD_02,
    "froslass_lopunny": top_decks.TD_04,
    "festival": top_decks.TD_10,
}
ARMS = [a for a in os.environ.get(
    "RCB_ARMS", "control,wall12,wall16,wall24,wall28,floor2,xero6,xero_aggr"
).split(",") if a]


def transformed(arm: str) -> str:
    source = SOURCE
    if arm.startswith("wall"):
        value = int(arm[4:])
        source = source.replace("if opponent.deckCount <= 20:\n        return False", f"if opponent.deckCount <= {value}:\n        return False")
    elif arm == "floor2":
        source = source.replace("if opponent_bench_counter_pressure(opponent):\n        return 3", "if opponent_bench_counter_pressure(opponent):\n        return 2")
    elif arm == "xero6":
        source = source.replace("opponent.handCount >= 8:\n            score = 265000 + 2500 * (opponent.handCount - 8)", "opponent.handCount >= 6:\n            score = 265000 + 2500 * (opponent.handCount - 6)")
    elif arm == "xero_aggr":
        source = source.replace("opponent.handCount >= 5 and not active_ready_tusk:\n            score = 12000", "opponent.handCount >= 5 and not active_ready_tusk:\n            score = 180000")
    return source


def load(name: str, arm: str):
    module = types.ModuleType(name)
    module.__file__ = MAIN
    exec(compile(transformed(arm), MAIN, "exec"), module.__dict__)
    return module


def chunk(args):
    arm, seeds = args
    candidate = load(f"rcb_{arm}_{os.getpid()}", arm)
    control = load(f"rcb_control_{arm}_{os.getpid()}", "control")

    def cand(obs):
        return DECK if obs.get("select") is None else candidate.agent(obs)
    def base(obs):
        return DECK if obs.get("select") is None else control.agent(obs)

    rows = {}
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
        rows[name] = row
    return arm, rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seed0 = int(os.environ.get("RCB_SEED0", "17000000"))
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
        print(f"{arm:10s} {pooled[0]:4d}/{pooled[1]:4d} disc {pooled[2]:3d}/{pooled[3]:3d} p={mcnemar(pooled[2], pooled[3]):.5f}")
        print(" " * 11 + " ".join(f"{name}={row[0]-row[1]:+d}({row[2]}/{row[3]})" for name, row in totals[arm].items()))


if __name__ == "__main__":
    main()
