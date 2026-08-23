"""Submission entry point for the validated Grim family-search v2 composite.

The underlying Grim Candy v2 policy and deck are frozen. Search is enabled
only after a public Archaludon, Garchomp, or Router/Crustle family signature is
visible; all other observations are delegated directly to the frozen policy.
"""
from __future__ import annotations

from grim_base import DECK, agent as grim_agent
from grim_search_hybrid import make_agent, visible_family_gate


_GARCHOMP_IDS = {341, 342, 379, 380, 381, 387}
_ARCHALUDON_IDS = {57, 169, 190, 666}
_ROUTER_CRUSTLE_IDS = {344, 345, 607}

_garch_search = make_agent(grim_agent, DECK, margin=10000.0)
_arch_search = make_agent(grim_agent, DECK, margin=20000.0)
_router_search = make_agent(grim_agent, DECK, margin=20000.0)
_arch_gate = visible_family_gate(_arch_search, grim_agent, _ARCHALUDON_IDS)
_family_v1 = visible_family_gate(_garch_search, _arch_gate, _GARCHOMP_IDS)
agent = visible_family_gate(
    _router_search, _family_v1, _ROUTER_CRUSTLE_IDS)
