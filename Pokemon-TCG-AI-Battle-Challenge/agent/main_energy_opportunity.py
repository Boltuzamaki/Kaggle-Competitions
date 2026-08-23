"""Original Energy-opportunity Archaludon entry point for EXP-29."""

from __future__ import annotations

from archaludon_policy import (
    ARCHALUDON_DECK,
    archaludon_energy_opportunity_agent,
)


DECK = list(ARCHALUDON_DECK)


def agent(obs_dict: dict) -> list[int]:
    return archaludon_energy_opportunity_agent(obs_dict, DECK)


if __name__ == "__main__":
    print(f"Energy-opportunity Archaludon deck: {len(DECK)} cards")
    print(f"Deck-select: {len(agent({'select': None}))} cards")
