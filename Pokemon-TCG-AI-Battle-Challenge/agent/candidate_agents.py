"""Small wrappers for isolated hybrid-policy A/B experiments."""

from __future__ import annotations

import os

import domain_policy_active_threat
import domain_policy_no_retreat
import hybrid_agent
import archaludon_policy
import beam_archaludon
import context_ranker
import information_search
import search_agent

try:
    from cg.api import to_observation_class
except Exception:
    to_observation_class = None


def hybrid_active_threat(obs_dict, deck):
    return hybrid_agent.hybrid_agent(
        obs_dict, deck, policy_module=domain_policy_active_threat
    )


def hybrid_no_retreat(obs_dict, deck):
    return hybrid_agent.hybrid_agent(
        obs_dict, deck, policy_module=domain_policy_no_retreat
    )


def archaludon_search(obs_dict, deck):
    hybrid_agent._TOP_K = 6
    hybrid_agent._DET = 3
    hybrid_agent._MAXROLL = 24
    hybrid_agent._CORRECT_OWN_DET = False
    hybrid_agent._OPPONENT_BELIEF = False
    return hybrid_agent.hybrid_agent(
        obs_dict, deck, policy_module=archaludon_policy
    )


def _archaludon_search_config(
    obs_dict,
    deck,
    top_k,
    determinizations,
    max_roll,
    correct_own=False,
    opponent_belief=False,
):
    hybrid_agent._TOP_K = top_k
    hybrid_agent._DET = determinizations
    hybrid_agent._MAXROLL = max_roll
    hybrid_agent._CORRECT_OWN_DET = correct_own
    hybrid_agent._OPPONENT_BELIEF = opponent_belief
    return hybrid_agent.hybrid_agent(
        obs_dict, deck, policy_module=archaludon_policy
    )


def arch_search_k3_d6(obs_dict, deck):
    return _archaludon_search_config(obs_dict, deck, 3, 6, 24)


def arch_search_k6_d3(obs_dict, deck):
    return _archaludon_search_config(obs_dict, deck, 6, 3, 24)


def arch_search_k9_d2(obs_dict, deck):
    return _archaludon_search_config(obs_dict, deck, 9, 2, 24)


def arch_search_k6_d3_correct(obs_dict, deck):
    return _archaludon_search_config(obs_dict, deck, 6, 3, 24, correct_own=True)


def arch_search_k9_d2_correct(obs_dict, deck):
    return _archaludon_search_config(obs_dict, deck, 9, 2, 24, correct_own=True)


def arch_belief_d1_r48(obs_dict, deck):
    return _archaludon_search_config(
        obs_dict, deck, 6, 1, 48, opponent_belief=True
    )


def arch_belief_d3_r16(obs_dict, deck):
    return _archaludon_search_config(
        obs_dict, deck, 6, 3, 16, opponent_belief=True
    )


def arch_belief_d6_r8(obs_dict, deck):
    return _archaludon_search_config(
        obs_dict, deck, 6, 6, 8, opponent_belief=True
    )


def arch_belief_d12_r4(obs_dict, deck):
    return _archaludon_search_config(
        obs_dict, deck, 6, 12, 4, opponent_belief=True
    )


def arch_beam_2(obs_dict, deck):
    return beam_archaludon.beam_agent(obs_dict, deck, 2)


def arch_beam_4(obs_dict, deck):
    return beam_archaludon.beam_agent(obs_dict, deck, 4)


def arch_beam_8(obs_dict, deck):
    return beam_archaludon.beam_agent(obs_dict, deck, 8)


def arch_beam_2_response(obs_dict, deck):
    return beam_archaludon.beam_agent(obs_dict, deck, 2, opponent_response=True)


def arch_flat_root(obs_dict, deck):
    return information_search.flat_root_agent(obs_dict, deck)


def arch_cached_flat_root(obs_dict, deck):
    return information_search.cached_flat_root_agent(obs_dict, deck)


def arch_gated_flat_root(obs_dict, deck):
    return information_search.gated_flat_root_agent(obs_dict, deck)


def arch_progressive_widening(obs_dict, deck):
    return information_search.progressive_widening_agent(obs_dict, deck)


def arch_risk_sensitive(obs_dict, deck):
    return information_search.risk_sensitive_agent(obs_dict, deck)


def arch_chance_coupled(obs_dict, deck):
    return information_search.chance_coupled_agent(obs_dict, deck)


def arch_adaptive_budget(obs_dict, deck):
    return information_search.adaptive_budget_agent(obs_dict, deck)


def arch_disagreement_fallback(obs_dict, deck):
    return information_search.disagreement_fallback_agent(obs_dict, deck)


def arch_ismcts(obs_dict, deck):
    return information_search.ismcts_agent(obs_dict, deck)


def arch_gated_ismcts(obs_dict, deck):
    return information_search.gated_ismcts_agent(obs_dict, deck)


def _visible_archetype(obs_dict):
    try:
        observation = to_observation_class(obs_dict)
        me = observation.current.yourIndex
        opponent = observation.current.players[1 - me]
        ids = {
            pokemon.id
            for pokemon in list(opponent.active) + list(opponent.bench)
            if pokemon is not None
        }
        ids.update(card.id for card in (opponent.discard or []))
        signatures = (
            ("alakazam", {741, 742, 743, 305, 66, 858, 343}),
            ("lucario", {673, 674, 675, 676, 677, 678}),
            ("dragapult", {119, 120, 121, 184, 235}),
            ("abomasnow", {721, 722, 723}),
            ("archaludon", {169, 190, 666, 57}),
        )
        for name, signature in signatures:
            if ids & signature:
                return name
    except Exception:
        pass
    return "unknown"


def arch_adaptive(obs_dict, deck):
    archetype = _visible_archetype(obs_dict) if obs_dict.get("select") is not None else "unknown"
    if archetype == "lucario":
        return arch_beam_2_response(obs_dict, deck)
    if archetype in {"dragapult", "abomasnow", "archaludon"}:
        return arch_beam_2(obs_dict, deck)
    return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)


_RL_CACHE = {}
_RL_CHECKPOINT_CACHE = {}
_CONTEXT_MODEL = None


def _arch_rl(obs_dict, deck, searches):
    if obs_dict.get("select") is None:
        return list(deck)
    try:
        if searches not in _RL_CACHE:
            import rlnet

            model_path = os.path.join(os.path.dirname(__file__), "model.pth")
            _RL_CACHE[searches] = rlnet.RLAgent(
                model_path=model_path,
                search_count=searches,
                time_budget_s=1.5,
                deck=deck,
            )
        action = _RL_CACHE[searches].act(obs_dict)
        if action is not None:
            return action
    except Exception:
        pass
    return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)


def arch_rl_16(obs_dict, deck):
    return _arch_rl(obs_dict, deck, 16)


def arch_rl_24(obs_dict, deck):
    return _arch_rl(obs_dict, deck, 24)


def arch_rl_32(obs_dict, deck):
    return _arch_rl(obs_dict, deck, 32)


def arch_rl_48(obs_dict, deck):
    return _arch_rl(obs_dict, deck, 48)


def rl_exp18_8(obs_dict, deck):
    """Research-only wrapper for the mixed-opponent EXP-18 checkpoint."""
    if obs_dict.get("select") is None:
        return list(deck)
    try:
        key = ("exp18", 8)
        if key not in _RL_CHECKPOINT_CACHE:
            import rlnet

            model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "scratchpad",
                "rl_exp18_league_smoke.pth",
            )
            _RL_CHECKPOINT_CACHE[key] = rlnet.RLAgent(
                model_path=model_path,
                search_count=8,
                time_budget_s=1.5,
                deck=deck,
            )
        action = _RL_CHECKPOINT_CACHE[key].act(obs_dict)
        if action is not None:
            return action
    except Exception:
        pass
    return search_agent._heuristic(obs_dict)


def _context_blend(obs_dict, deck, alpha):
    global _CONTEXT_MODEL
    if _CONTEXT_MODEL is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "scratchpad",
            "context_ranker.json",
        )
        _CONTEXT_MODEL = context_ranker.ContextModel(model_path)
    return context_ranker.blended_agent(obs_dict, deck, _CONTEXT_MODEL, alpha)


def arch_context_015(obs_dict, deck):
    return _context_blend(obs_dict, deck, 0.15)


def arch_context_035(obs_dict, deck):
    return _context_blend(obs_dict, deck, 0.35)


def arch_context_070(obs_dict, deck):
    return _context_blend(obs_dict, deck, 0.70)


def arch_learned_adaptive(obs_dict, deck):
    archetype = _visible_archetype(obs_dict) if obs_dict.get("select") is not None else "unknown"
    if archetype in {"lucario", "dragapult", "archaludon"}:
        return arch_context_035(obs_dict, deck)
    if archetype == "abomasnow":
        return arch_beam_2(obs_dict, deck)
    return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)
