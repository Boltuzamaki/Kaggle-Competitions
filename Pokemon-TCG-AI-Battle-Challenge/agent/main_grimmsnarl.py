"""Submission entry point: our original Marnie's Grimmsnarl ex policy.

Layers, strongest first, each falling back to the next so the agent always
returns a legal move (a crash or a timeout is a loss):

  1. grimmsnarl_policy -- our Grimmsnarl-aware policy (Punk Up energy routing,
     Shadow Bullet snipe targeting, Adrena-Brain damage-counter movement).
  2. domain_policy     -- our deck-agnostic card-data policy.
  3. index fallback    -- first legal option(s).
"""

from __future__ import annotations

import grimmsnarl_policy


DECK: list[int] = list(grimmsnarl_policy.GRIMMSNARL_DECK)


def agent(obs_dict: dict) -> list[int]:
    try:
        return grimmsnarl_policy.grimmsnarl_agent(obs_dict, DECK)
    except Exception:
        try:
            import domain_policy
            return domain_policy.domain_agent(obs_dict, DECK)
        except Exception:
            select = obs_dict.get("select")
            if select is None:
                return list(DECK)
            options = select.get("option") or []
            count = select.get("maxCount", 1) or 1
            return list(range(min(count, len(options))))


if __name__ == "__main__":
    print(f"Grimmsnarl deck: {len(DECK)} cards")
