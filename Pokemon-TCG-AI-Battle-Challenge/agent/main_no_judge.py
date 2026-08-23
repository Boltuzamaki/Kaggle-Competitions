"""Balanced clean-room Archaludon entry point for EXP-22 validation."""

from __future__ import annotations

from archaludon_policy import (
    ARCHALUDON_DECK,
    JUDGE,
    METAL_ENERGY,
    archaludon_sequenced_agent,
)


DECK = list(ARCHALUDON_DECK)
DECK.remove(JUDGE)
DECK.append(METAL_ENERGY)


def agent(obs_dict: dict) -> list[int]:
    return archaludon_sequenced_agent(obs_dict, DECK)


if __name__ == "__main__":
    print(f"No-Judge Archaludon deck: {len(DECK)} cards")
    print(f"Deck-select: {len(agent({'select': None}))} cards")

