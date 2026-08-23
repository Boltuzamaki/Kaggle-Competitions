"""Matchup-gated policy selection between hybrid search and v3 search.

Benchmarked against frozen public ~950-rated agents (100 games each), our two
policies are complementary rather than ordered:

                        vs public-archaludon    vs public-alakazam
    hybrid (v5)                 40%                    25%
    v3 (determinized search)    21%                    30%

Neither dominates, so a router that picks per opponent is worth more than either
alone. This mirrors the strongest published result on this problem (discussion
713608): their best agent gated on opponent identity for +5.7pp, and the
DIRECTION mattered -- "default search, switch to pure" scored 57.0% while
"default pure, switch to search" scored 59.7% with the same parts. So the default
here is the policy with the better overall rating (hybrid, 51.8% vs 47.8%), and
we only switch when the opponent is confidently identified as a matchup the other
policy handles better.

Identification reuses opponent_deck_model, which resolves the opponent's
archetype from revealed cards in ~90% of decisions.

Fallback chain: gate -> hybrid -> domain policy -> first legal index. A crash or
timeout is an instant loss, so no failure mode is allowed to escape.
"""

from __future__ import annotations

import os

import domain_policy
import hybrid_agent
import search_agent

try:
    from opponent_deck_model import identify
except Exception:  # pragma: no cover
    identify = None

# Archetypes where v3's plainer search measured better than hybrid.
V3_FAVOURED = {a.strip() for a in os.environ.get(
    "GATE_V3_ARCHETYPES", "alakazam").split(",") if a.strip()}
MIN_EVIDENCE = int(os.environ.get("GATE_MIN_EVIDENCE", "3"))

STATS = {"v3": 0, "hybrid": 0, "unidentified": 0}


def _route(obs_dict):
    """Return 'v3' or 'hybrid' for this decision."""
    if identify is None:
        return "hybrid"
    try:
        cur = obs_dict.get("current")
        if not cur:
            return "hybrid"
        me = cur.get("yourIndex")
        if me is None:
            return "hybrid"
        opp = cur["players"][1 - me]

        class _P:  # opponent_deck_model expects attribute access
            active = opp.get("active") or []
            bench = opp.get("bench") or []
            discard = opp.get("discard") or []
            deckCount = opp.get("deckCount", 0) or 0

        # dicts -> light shim so .id / .energyCards lookups work
        class _M:
            def __init__(self, d):
                self.id = d.get("id")
                self.energyCards = d.get("energyCards") or []
                self.preEvolution = d.get("preEvolution") or []

        _P.active = [_M(m) for m in _P.active if m]
        _P.bench = [_M(m) for m in _P.bench if m]
        _P.discard = [_M(c) for c in _P.discard if c]

        name, score = identify(_P)
        if name is None or score < MIN_EVIDENCE:
            STATS["unidentified"] += 1
            return "hybrid"
        if name in V3_FAVOURED:
            STATS["v3"] += 1
            return "v3"
        STATS["hybrid"] += 1
        return "hybrid"
    except Exception:
        return "hybrid"


def gated_agent(obs_dict: dict, deck) -> list[int]:
    deck = list(deck)
    try:
        if obs_dict.get("select") is None:
            return deck
        if _route(obs_dict) == "v3":
            try:
                search_agent.DECK = deck
                return search_agent.agent(obs_dict)
            except Exception:
                pass
        return hybrid_agent.hybrid_agent(obs_dict, deck)
    except Exception:
        try:
            return domain_policy.domain_agent(obs_dict, deck)
        except Exception:
            sel = obs_dict.get("select")
            if sel is None:
                return deck
            n = len(sel.get("option") or [])
            mc = sel.get("maxCount", 1) or 1
            return list(range(min(mc, n))) if n else []


def make_agent(deck):
    frozen = list(deck)

    def run(obs_dict):
        return gated_agent(obs_dict, frozen)
    return run
