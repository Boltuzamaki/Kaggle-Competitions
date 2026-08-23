"""Submission entry point: domain-driven search (hybrid v5) on the Garchomp deck.

Chosen on the first benchmark this project has had that can actually discriminate:
paired common-random-number evaluation against a frozen public ~950-rated agent.

    vs public-archaludon, 120 games each, shared seeds
      hybrid  29.2%   |  v3 (previously submitted)  18.3%
      discordant 22/9, McNemar p = 0.031

The same comparison against our own weak local opponents reported p = 0.77, "no
detectable difference" -- the benchmark was saturated. Public agents are used
only as frozen opponents for measurement; no public policy code is included.
"""

from __future__ import annotations

import hybrid_agent
from meta_decks import GARCHOMP

DECK: list[int] = list(GARCHOMP)


def agent(obs_dict: dict) -> list[int]:
    try:
        return hybrid_agent.hybrid_agent(obs_dict, DECK)
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
    print(f"hybrid Garchomp: {len(DECK)} cards")
