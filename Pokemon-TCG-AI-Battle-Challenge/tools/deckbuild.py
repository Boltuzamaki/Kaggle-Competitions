"""
Deck builder + head-to-head tester for the PTCG AI Battle (cabt).

Decks enter a match ONLY via each agent's deck-selection return (cabt.py:122
`decks = [state[0].action, state[1].action]` -> `battle_start`). So to compare
deck A vs deck B we play our *same* heuristic policy on both sides, differing
only in the deck it returns during deck-selection. Whatever wins the arena with
significance is the better deck (PLAN.md, Phase 2).

Deck legality (checked by `battle_start`, errorType):
    1 invalid card id | 2 >4 of a name (except Basic Energy)
    3 no Basic Pokémon | 4 >1 Ace Spec

Usage
-----
    python tools/deckbuild.py --list                 # validate all candidates
    python tools/deckbuild.py --a v1_default --b v2_search --games 60
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "agent"))

from agent.main import _score_option  # reuse the exact in-game policy  # noqa: E402


# ---------------------------------------------------------------------------
# Candidate decks. Card IDs decoded from data/EN_Card_Data.csv.
# Archetype: Mega Abomasnow ex (Water) — 722 Snover -> 723 Mega Abomasnow ex
# (350 HP; Frost Barrier {W}{W}{W} -> 200), 721 Kyogre secondary. No energy
# acceleration, so attackers are energy-hungry (hence lots of Basic {W} Energy).
# ---------------------------------------------------------------------------
def _deck(spec: dict[int, int]) -> list[int]:
    out: list[int] = []
    for cid, n in spec.items():
        out += [cid] * n
    return out


DECKS: dict[str, list[int]] = {
    # The engine's tuned sample deck: 10 Pokémon / 17 Trainers / 33 Energy.
    "v1_default": _deck({
        722: 4, 723: 4, 721: 2,                       # Pokémon (10)
        1219: 4, 1227: 4, 1121: 2, 1145: 2,           # Trainers (17)
        1163: 2, 1262: 2, 1092: 1,
        3: 33,                                         # Basic {W} Energy
    }),
    # Consistency-tuned: trade dead energy for search (+Ultra Ball to 4, +Poffin).
    # 10 Pokémon / 23 Trainers / 27 Energy.
    "v2_search": _deck({
        722: 4, 723: 4, 721: 2,                       # Pokémon (10)
        1219: 4, 1227: 4, 1121: 4, 1145: 2,           # Trainers (23)
        1163: 2, 1262: 2, 1092: 1, 1086: 4,
        3: 27,
    }),
    # Lean: minimal safe tweak — only +2 Ultra Ball (search), -2 energy. No new
    # cards. (Secret Box 1092 is an Ace Spec, so it stays at 1.)
    # 10 Pokémon / 19 Trainers / 31 Energy.
    "v3_lean": _deck({
        722: 4, 723: 4, 721: 2,                       # Pokémon (10)
        1219: 4, 1227: 4, 1121: 4, 1145: 2,           # Trainers (19)
        1163: 2, 1262: 2, 1092: 1,
        3: 31,
    }),
}


def make_agent(deck: list[int]):
    """Our heuristic policy, returning `deck` during the deck-selection phase."""
    def agent(obs_dict):
        try:
            sel = obs_dict.get("select")
            if sel is None:
                return list(deck)
            opts = sel.get("option") or []
            if not opts:
                return []
            mc = sel.get("maxCount", 1) or 1
            order = sorted(range(len(opts)), key=lambda i: _score_option(opts[i], obs_dict), reverse=True)
            return order[:mc]
        except Exception:
            sel = obs_dict.get("select") if isinstance(obs_dict, dict) else None
            return list(deck) if sel is None else [0]
    return agent


def validate(deck: list[int]) -> tuple[bool, str]:
    from kaggle_environments.envs.cabt.cg.game import battle_start, battle_finish
    if len(deck) != 60:
        return False, f"{len(deck)} cards (need 60)"
    try:
        _, sd = battle_start(list(deck), list(deck))
        ep, et = getattr(sd, "errorPlayer", -1), getattr(sd, "errorType", 0)
        battle_finish()
    except Exception as e:
        return False, str(e)
    if ep is not None and ep >= 0:
        msg = {1: "invalid card id", 2: ">4 of a name", 3: "no Basic Pokémon", 4: ">1 Ace Spec"}.get(et, f"errorType {et}")
        return False, msg
    return True, "legal"


def wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - m), min(1.0, c + m))


def play(deck_a: list[int], deck_b: list[int], games: int) -> None:
    from kaggle_environments import make
    a, b = make_agent(deck_a), make_agent(deck_b)
    w = l = d = 0
    for g in range(games):
        players = [a, b] if g % 2 == 0 else [b, a]
        seat = 0 if g % 2 == 0 else 1
        env = make("cabt")
        env.run(players)
        f = env.steps[-1]
        ar, br = f[seat].get("reward"), f[1 - seat].get("reward")
        if ar is None or br is None or ar == br:
            d += 1
        elif ar > br:
            w += 1
        else:
            l += 1
        print(f"  game {g + 1}/{games}: A {w}W-{l}L-{d}D", end="\r")
    lo, hi = wilson(w, games)
    print("\n" + "=" * 50)
    print(f"A vs B: {w}W - {l}L - {d}D   A win-rate {w / games:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")
    if lo > 0.5:
        print("[OK] A is significantly better -- adopt A.")
    elif hi < 0.5:
        print("[BAD] A is significantly worse -- keep B.")
    else:
        print("[--] No significant difference -- keep the incumbent (B).")
    print("=" * 50)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="validate all candidate decks")
    ap.add_argument("--a", default="v2_search")
    ap.add_argument("--b", default="v1_default")
    ap.add_argument("--games", type=int, default=60)
    args = ap.parse_args()

    if args.list:
        for name, deck in DECKS.items():
            ok, msg = validate(deck)
            print(f"  {name:14} {len(deck)} cards  ->  {'OK ' if ok else 'BAD'}  {msg}")
        return

    for name in (args.a, args.b):
        ok, msg = validate(DECKS[name])
        if not ok:
            sys.exit(f"Deck {name} is illegal: {msg}")
    print(f"A = {args.a}   B = {args.b}   games = {args.games}")
    play(DECKS[args.a], DECKS[args.b], args.games)


if __name__ == "__main__":
    main()
