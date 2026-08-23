"""
CANDIDATE entry point — hybrid agent (our domain policy + determinized search)
piloting the Mega Lucario ex deck.

This is a SEPARATE file from `main.py` (the safe, proven default = v3 search on
our Abomasnow deck). Only bundle/submit this once it has beaten v3 with
statistical significance in the arena (tools/arena.py) — see PLAN.md Part I,
"never replace the safety submission until a candidate is proven locally and on
the leaderboard."

Falls back through: hybrid_agent -> domain_policy -> legal-first-option, so it
never crashes even if something in the search path fails.
"""

from __future__ import annotations

# Mega Lucario ex (Fighting) — verified 60-card meta deck (references/top_rankers/decks.py).
# Embedded directly (Kaggle loads this file with exec(): no __file__, unknown cwd).
DECK: list[int] = [
    673, 673, 674, 674, 675, 675, 676, 676, 676, 677,
    677, 677, 678, 678, 678, 678, 1102, 1102, 1102, 1102,
    1123, 1123, 1141, 1141, 1141, 1141, 1142, 1142, 1142, 1142,
    1152, 1152, 1152, 1152, 1159, 1182, 1182, 1192, 1192, 1192,
    1192, 1227, 1227, 1227, 1227, 1252, 1252, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
]

import hybrid_agent  # noqa: E402
import domain_policy  # noqa: E402


def agent(obs_dict: dict) -> list[int]:
    try:
        return hybrid_agent.hybrid_agent(obs_dict, DECK)
    except Exception:
        try:
            return domain_policy.domain_agent(obs_dict, DECK)
        except Exception:
            try:
                select = obs_dict.get("select")
                if select is None:
                    return list(DECK)
                n = len(select.get("option") or [])
                mc = select.get("maxCount", 1) or 1
                return list(range(min(mc, n))) if n else []
            except Exception:
                return list(DECK) if DECK else [0]


if __name__ == "__main__":
    print(f"Embedded deck (Mega Lucario ex): {len(DECK)} cards")
    print("Deck-select returns", len(agent({"select": None})), "card IDs")
