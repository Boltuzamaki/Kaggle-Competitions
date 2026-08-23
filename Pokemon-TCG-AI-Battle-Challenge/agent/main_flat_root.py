"""Submission entry point for the clean-room flat-root w8/i40 candidate."""

from __future__ import annotations

import archaludon_policy
import information_search


DECK: list[int] = list(archaludon_policy.ARCHALUDON_DECK)


def agent(obs_dict: dict) -> list[int]:
    try:
        return information_search.flat_root_agent_config(
            obs_dict,
            DECK,
            iterations=40,
            root_width=8,
        )
    except Exception:
        try:
            return archaludon_policy.archaludon_sequenced_agent(obs_dict, DECK)
        except Exception:
            select = obs_dict.get("select")
            if select is None:
                return list(DECK)
            options = select.get("option") or []
            count = select.get("maxCount", 1) or 1
            return list(range(min(count, len(options))))


if __name__ == "__main__":
    print(f"Flat-root w8/i40 deck: {len(DECK)} cards")
