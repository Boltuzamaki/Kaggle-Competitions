"""Candidate entry point for our clean-room Archaludon policy."""

from __future__ import annotations

from archaludon_policy import ARCHALUDON_DECK, archaludon_agent

DECK = list(ARCHALUDON_DECK)


def agent(obs_dict):
    return archaludon_agent(obs_dict, DECK)


if __name__ == "__main__":
    print(f"Our Archaludon deck: {len(DECK)} cards")
    print(f"Deck-select: {len(agent({'select': None}))} cards")
