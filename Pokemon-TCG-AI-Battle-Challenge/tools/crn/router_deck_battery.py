"""Screen Hand Trimmer counts on the confirmed guarded router policy."""
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
BASE = [int(x) for x in open(os.path.join(os.path.dirname(MAIN), "deck.csv")) if x.strip()]
HAND_TRIMMER = 1087
AMARYS = 1207
ROTO_STICK = 1077
REMOVALS = [1147, 607, 1123, 1182]
PANELS = {
    "upper": {
        "grimmsnarl": top_decks.TD_01,
        "alakazam": top_decks.TD_02,
        "froslass_lopunny": top_decks.TD_04,
        "festival": top_decks.TD_10,
    },
    "confirm": {
        "ogerpon": top_decks.TD_00,
        "dragapult": top_decks.TD_03,
        "froslass_lopunny": top_decks.TD_09,
        "crustle": top_decks.TD_12,
    },
}
PANEL = PANELS[os.environ.get("RDB_PANEL", "upper")]
ARMS = [a for a in os.environ.get("RDB_ARMS", "control,ht1,ht2,ht3,ht4").split(",") if a]


def deck_for(arm: str) -> list[int]:
    deck = list(BASE)
    if arm.startswith("ht"):
        count = int(arm[2:])
        for card_id in REMOVALS[:count]:
            deck.remove(card_id)
            deck.append(HAND_TRIMMER)
    elif arm.startswith("am"):
        count = int(arm[2:])
        for _ in range(count):
            deck.remove(1197)  # Xerosic: dead weight into Froslass and same supporter slot.
            deck.append(AMARYS)
    elif arm.startswith("rs"):
        count = int(arm[2:])
        for _ in range(count):
            deck.remove(1197)
            deck.append(ROTO_STICK)
    elif arm.startswith("ro"):
        count = int(arm[2:])
        for card_id in REMOVALS[:count]:
            deck.remove(card_id)
            deck.append(ROTO_STICK)
    assert len(deck) == 60
    return deck


def load(name: str, deck: list[int], arm: str = "control"):
    spec = ilu.spec_from_file_location(name, MAIN)
    module = ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._ARCHIVE_DECK = tuple(deck)
    if arm.startswith("am"):
        original_play_score = module.play_score

        def play_score(card_id, me, opponent, state, wall_mode, ko_mode):
            if card_id == AMARYS:
                visible = module.opponent_visible_ids(opponent)
                if visible & {860, 861} and not state.supporterPlayed and me.handCount >= 5:
                    # Draw four, attack, then discard our hand at end of turn;
                    # this directly suppresses Resentful Refrain's 50x hand damage.
                    return 450000
                return -10000
            return original_play_score(card_id, me, opponent, state, wall_mode, ko_mode)

        module.play_score = play_score
    return module


def chunk(args):
    arm, seeds = args
    candidate_deck = deck_for(arm)
    candidate = load(f"rdb_{arm}_{os.getpid()}", candidate_deck, arm)
    control = load(f"rdb_base_{arm}_{os.getpid()}", BASE)

    def cand(obs):
        return candidate_deck if obs.get("select") is None else candidate.agent(obs)

    def base(obs):
        return BASE if obs.get("select") is None else control.agent(obs)

    aw = bw = ao = bo = 0
    matchup = {}
    for name, opponent_deck in PANEL.items():
        def opponent(obs, deck=opponent_deck):
            return list(deck) if obs.get("select") is None else domain_policy.domain_agent(obs, deck)
        local = [0, 0, 0, 0]
        for seed in seeds:
            counts = []
            for policy, deck in ((cand, candidate_deck), (base, BASE)):
                wins = 0
                for seat in (0, 1):
                    random.seed(seed)
                    if seat == 0:
                        result = play(seed, policy, opponent, deck, opponent_deck)
                        wins += int(result == 0)
                    else:
                        result = play(seed, opponent, policy, opponent_deck, deck)
                        wins += int(result == 1)
                counts.append(wins)
            local[0] += counts[0]
            local[1] += counts[1]
            local[2] += int(counts[0] > counts[1])
            local[3] += int(counts[1] > counts[0])
        matchup[name] = local
        aw += local[0]; bw += local[1]; ao += local[2]; bo += local[3]
    return arm, (aw, bw, ao, bo), matchup


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seed0 = int(os.environ.get("RDB_SEED0", "10870000"))
    chunks = max(1, workers // len(ARMS))
    seeds = [seed0 + i for i in range(n)]
    jobs = [(arm, seeds[i::chunks]) for arm in ARMS for i in range(chunks)]
    totals = {arm: [0, 0, 0, 0] for arm in ARMS}
    details = {arm: {name: [0, 0, 0, 0] for name in PANEL} for arm in ARMS}
    with mp.get_context("fork").Pool(workers) as pool:
        for arm, row, matchup in pool.imap_unordered(chunk, jobs):
            totals[arm] = [a + b for a, b in zip(totals[arm], row)]
            for name, values in matchup.items():
                details[arm][name] = [a + b for a, b in zip(details[arm][name], values)]
    for arm in ARMS:
        aw, bw, ao, bo = totals[arm]
        f = details[arm]["froslass_lopunny"]
        print(f"{arm:8s} all {aw:4d}/{bw:4d} disc {ao:3d}/{bo:3d} p={mcnemar(ao, bo):.5f} | "
              f"fros {f[0]:3d}/{f[1]:3d} disc {f[2]:2d}/{f[3]:2d}")
        print(" " * 9 + " ".join(
            f"{name}={values[0]-values[1]:+d}({values[2]}/{values[3]})"
            for name, values in details[arm].items()
        ))


if __name__ == "__main__":
    main()
