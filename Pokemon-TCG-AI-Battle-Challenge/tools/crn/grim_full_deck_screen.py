"""Screen complete replay-mined Grim decks under frozen family-v2 policy."""
from __future__ import annotations

import importlib.util as ilu
import json
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (HERE, os.path.join(ROOT, "agent"), os.path.join(ROOT, "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)
from paired_eval import build, mcnemar, play

BASE_DIR = os.path.join(ROOT, "agent", "grim_search_family_v2_package")
MAIN = os.path.join(BASE_DIR, "main.py")
BASE = [int(x) for x in open(os.path.join(BASE_DIR, "deck.csv")) if x.strip()]
TEMPLATES = os.path.join(
    ROOT, "references", "competitor_refresh", "crustle_v29", "selector_templates_v29.json"
)
OPPONENTS = tuple(os.environ.get(
    "GFDS_OPPS",
    "meta-grimmsnarl,public-alakazam,public-archaludon,public-garchomp-v28,router-v12,meta-crustle",
).split(","))


def variants():
    result = {}
    for row in json.load(open(TEMPLATES)):
        deck = [int(x) for x in row.get("deck", [])]
        if len(deck) == 60 and 648 in deck and deck != BASE:
            result[str(row.get("participantId"))] = deck
    return result


def load(name):
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
    spec = ilu.spec_from_file_location(name, MAIN)
    module = ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(job):
    name, deck, seeds = job
    candidate = load(f"gfds_candidate_{os.getpid()}")
    control = load(f"gfds_control_{os.getpid()}")
    aw = bw = ao = bo = 0
    by_opp = {}
    for opponent_name in OPPONENTS:
        opponent, opponent_deck = build(opponent_name)
        ca = cb = 0
        for seed in seeds:
            counts = []
            for module, own_deck in ((candidate, deck), (control, BASE)):
                wins = 0
                for seat in (0, 1):
                    if hasattr(module, "_reset"):
                        module._reset()
                    random.seed(seed)
                    if seat == 0:
                        wins += int(play(seed, module.agent, opponent, own_deck, opponent_deck) == 0)
                    else:
                        wins += int(play(seed, opponent, module.agent, opponent_deck, own_deck) == 1)
                counts.append(wins)
            aw += counts[0]; bw += counts[1]; ca += counts[0]; cb += counts[1]
            ao += int(counts[0] > counts[1]); bo += int(counts[1] > counts[0])
        by_opp[opponent_name] = (ca, cb)
    return name, aw, bw, ao, bo, by_opp


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    seed0 = int(os.environ.get("GFDS_SEED0", "8800000"))
    decks = variants()
    selected = os.environ.get("GFDS_ARMS")
    if selected:
        wanted = set(selected.split(",")); decks = {k: v for k, v in decks.items() if k in wanted}
    jobs = [(name, deck, range(seed0, seed0 + n)) for name, deck in decks.items()]
    with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
        results = list(pool.imap_unordered(evaluate, jobs))
    for name, aw, bw, ao, bo, by in sorted(results, key=lambda x: x[1] - x[2], reverse=True):
        detail = " ".join(f"{k}:{a}/{b}" for k, (a, b) in by.items())
        print(f"{name:36s} {aw:3d}/{bw:3d} delta={aw-bw:+3d} disc={ao}/{bo} "
              f"p={mcnemar(ao, bo):.4f} {detail}", flush=True)


if __name__ == "__main__":
    main()
