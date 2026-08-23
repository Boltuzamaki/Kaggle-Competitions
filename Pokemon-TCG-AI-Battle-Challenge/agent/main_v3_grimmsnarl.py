"""Submission entry point: our v3 determinized-search policy on the Grimmsnarl deck.

This is the controlled upgrade of the 531.2 submission. The policy is byte-for-byte
the same v3 shallow determinized search that scored 531.2; only the deck changes,
from the cabt engine's Mega Abomasnow sample list (33 basic Energy, zero presence
in the live field) to the Marnie's Grimmsnarl ex list mined from the official
2026-08-03 top-episode dump.

Local field gauntlet, same policy on both decks: 77.3% rating on Grimmsnarl vs
69.0% on Abomasnow across 11 competitors.
"""

from __future__ import annotations

import search_agent
from meta_decks import GRIMMSNARL


DECK: list[int] = list(GRIMMSNARL)

# The search agent determinizes hidden information by drawing from its embedded
# deck, so it must point at the deck we are actually piloting.
search_agent.DECK = DECK


def agent(obs_dict: dict) -> list[int]:
    try:
        search_agent.DECK = DECK
        return search_agent.agent(obs_dict)
    except Exception:
        select = obs_dict.get("select")
        if select is None:
            return list(DECK)
        options = select.get("option") or []
        count = select.get("maxCount", 1) or 1
        return list(range(min(count, len(options))))


if __name__ == "__main__":
    print(f"v3 search on Grimmsnarl: {len(DECK)} cards")
