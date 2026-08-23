"""
Starter heuristic agent for the Pokémon TCG AI Battle Challenge (cabt engine).

Contract
--------
    agent(obs_dict: dict) -> list[int]

The engine hands you `obs_dict["select"]` = a list of *legal* options and a
`maxCount` (how many to pick). You return the index/indices you choose. You can
never make an illegal move — only pick the best legal one.

Design goals (in priority order):
    1. NEVER crash and NEVER time out  -> both are an automatic loss.
    2. Return a valid 60-card deck during the deck-selection phase.
    3. Beat the random baseline with cheap, fast heuristics.

This is intentionally simple and fast. Layer smarter policy/search on top only
after A/B-testing it in tools/run_match.py (see PLAN.md, Phase 4).

IMPORTANT (submission): Kaggle loads this file with `exec()`, where `__file__`
is NOT defined and the working dir is unknown. So the deck is EMBEDDED here as a
constant — never read from a sibling file at import time, or the agent dies on
load ("Validation Episode failed"). Keep DECK and agent/deck.csv in sync.
"""

from __future__ import annotations

import random

# ---------------------------------------------------------------------------
# Deck: EMBEDDED as a constant (must be exactly 60 valid card IDs).
# Mirror of agent/deck.csv — the cabt sample deck. Replace both together when
# you build a real meta deck (PLAN.md, Phase 2). Deck rules: 60 cards, <=4 of
# the same name (except Basic Energy), >=1 Basic Pokemon, <=1 Ace Spec.
# ---------------------------------------------------------------------------
DECK: list[int] = [
    721, 721, 722, 722, 722, 722, 723, 723, 723, 723,
    1092, 1121, 1121, 1145, 1145, 1163, 1163, 1219, 1219, 1219,
    1219, 1227, 1227, 1227, 1227, 1262, 1262, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
]


# ---------------------------------------------------------------------------
# Heuristic scoring of a single option.
#
# Real cabt options are STRUCTURED dicts keyed by an integer `type` = OptionType
# (from the cabt API docs), e.g. {"type": 13, "attackId": 5} or
# {"type": 8, "area": 2, "index": 1, ...}. There is no free text to match, so we
# score on the action *type*.
#
# OptionType enum (cabt api.html):
#   0 NUMBER  1 YES  2 NO  3 CARD  4 TOOL_CARD  5 ENERGY_CARD  6 ENERGY
#   7 PLAY    8 ATTACH  9 EVOLVE  10 ABILITY  11 DISCARD  12 RETREAT
#   13 ATTACK 14 END  15 SKILL  16 SPECIAL_CONDITION
#
# Guiding idea for the MAIN menu: do free/setup actions first (ability, evolve,
# attach, play), THEN attack, and only END the turn when nothing better remains
# — because ATTACK (and END) terminate your turn. Picking END/NO at random is
# exactly why a uniform-random agent throws away games.
# ---------------------------------------------------------------------------
_TYPE_SCORE = {
    10: 9.0,   # ABILITY  — free value, always worth using
    9: 8.0,    # EVOLVE   — strengthens the board
    8: 7.0,    # ATTACH   — power up attackers (once per turn)
    7: 6.0,    # PLAY     — develop board / draw / search
    1: 5.5,    # YES      — effects offered are usually beneficial
    13: 5.0,   # ATTACK   — take prizes, but AFTER setup (ends turn)
    15: 4.5,   # SKILL
    16: 4.0,   # SPECIAL_CONDITION
    3: 2.0,    # CARD          } neutral sub-selections: pick *a* legal one
    4: 2.0,    # TOOL_CARD     }
    5: 2.0,    # ENERGY_CARD   }
    6: 2.0,    # ENERGY        }
    0: 2.0,    # NUMBER        }
    12: -1.0,  # RETREAT  — costs energy + tempo; avoid unless best
    11: -2.0,  # DISCARD  — losing cards is usually bad
    2: -3.0,   # NO       — declining an offered effect
    14: -100.0,  # END    — only when it's the sole remaining option
}


def _score_option(option, obs_dict) -> float:
    """Higher = more desirable. Cheap: runs for every legal option every turn."""
    if not isinstance(option, dict):
        return 0.0
    score = _TYPE_SCORE.get(option.get("type"), 1.0)

    # For a NUMBER choice (draw/search counts are common), prefer bigger numbers.
    if option.get("type") == 0 and isinstance(option.get("number"), (int, float)):
        score += 0.3 * float(option["number"])

    return score


def agent(obs_dict: dict) -> list[int]:
    """Main entry point the cabt engine calls each decision.

    Engine contract (cabt / kaggle-environments 1.30.1):
      * obs["select"] is None  -> DECK-SELECTION phase: return the 60 card IDs.
      * otherwise              -> return index/indices into obs["select"]["option"]
                                  (exactly obs["select"]["maxCount"] of them).
    """
    try:
        select = obs_dict.get("select")

        # --- Deck-selection phase: return raw card IDs, not indices ----------
        if select is None:
            return list(DECK)

        options = select.get("option") or []
        max_count = select.get("maxCount", 1) or 1
        n = len(options)
        if n == 0:
            return []

        # --- In-game decision: greedy over the heuristic score ---------------
        scored = sorted(
            range(n), key=lambda i: _score_option(options[i], obs_dict), reverse=True
        )
        return scored[:max_count]

    except Exception:
        # Absolute safety net: never crash / forfeit -> return something legal.
        try:
            select = obs_dict.get("select")
            if select is None:
                return list(DECK)
            n = len(select.get("option") or [])
            mc = select.get("maxCount", 1) or 1
            return list(range(min(mc, n))) if n else []
        except Exception:
            return list(DECK) if DECK else [0]


# Convenience for local testing: `python agent/main.py`
if __name__ == "__main__":
    print(f"Embedded deck: {len(DECK)} cards")
    # Deck-selection phase (select is None) -> returns the deck.
    print("Deck-select returns", len(agent({"select": None})), "card IDs")
    # In-game decision -> returns the index of the best option.
    demo = {
        "current": {"turn": 3},
        "select": {
            "option": ["Attack: Thunderbolt (120 damage)", "Retreat", "Pass"],
            "maxCount": 1,
        },
    }
    print("Demo choice:", agent(demo), "->", demo["select"]["option"][agent(demo)[0]])
