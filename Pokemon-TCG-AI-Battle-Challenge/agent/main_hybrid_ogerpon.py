"""Submission entry point: domain-driven search (hybrid v5) on the Ogerpon deck.

Chosen on the first benchmark this project has had that can actually discriminate:
paired common-random-number evaluation against a frozen public ~950-rated agent.

    Deck chosen by paired cross-deck CRN vs BOTH frozen public agents, 240 games
    per side. Two independent policies agree:
      hybrid: Ogerpon 111/240 (46.3%) vs Garchomp 81/240 (33.8%), 48/26, p=0.0146
      v3    : Ogerpon  96/240 (40.0%) vs Garchomp 63/240 (26.3%), 48/23, p=0.0044
    The edge is concentrated in the Alakazam matchup (66.7% vs 33.3%); Garchomp
    stays better into Archaludon, so both are kept on the ladder.

The same comparison against our own weak local opponents reported p = 0.77, "no
detectable difference" -- the benchmark was saturated. Public agents are used
only as frozen opponents for measurement; no public policy code is included.
"""

from __future__ import annotations

import hybrid_agent
from meta_decks import OTHER

DECK: list[int] = list(OTHER)


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
    print(f"hybrid Ogerpon: {len(DECK)} cards")
