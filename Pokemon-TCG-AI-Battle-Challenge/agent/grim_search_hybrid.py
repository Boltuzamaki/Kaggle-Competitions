"""Conservative forward-search wrapper around a frozen baseline policy.

The baseline chooses the default real action. Search may replace it only at a
single-choice MAIN decision and only when deterministic rollouts give another
candidate a configurable value margin. Simulated continuation uses our clean
domain policy, so the frozen baseline's stateful memory is never mutated by
counterfactual branches.
"""
from __future__ import annotations

import domain_policy
import hybrid_agent
from cg.api import search_begin, search_end, search_step, to_observation_class


def make_agent(base_agent, deck, margin=1000.0, top_k=4, determinizations=2,
               max_roll=20, opponent_roll=0, margin_late=None,
               late_prizes=2):
    frozen_deck = list(deck)

    def run(obs_dict):
        baseline = base_agent(obs_dict)
        try:
            select = obs_dict.get("select")
            if select is None:
                return frozen_deck
            options = select.get("option") or []
            if (int(select.get("context", -1)) != 0
                    or int(select.get("maxCount", 1) or 1) != 1
                    or len(options) < 2 or len(baseline) != 1):
                return baseline

            observation = to_observation_class(obs_dict)
            me = observation.current.yourIndex
            base_index = int(baseline[0])
            ranked = [i for i in domain_policy.rank_options(observation)
                      if 0 <= i < len(options)]
            candidates = [base_index]
            candidates.extend(i for i in ranked if i != base_index)
            candidates = candidates[:top_k]
            if len(candidates) < 2:
                return baseline

            totals = {i: 0.0 for i in candidates}
            visits = {i: 0 for i in candidates}
            for _ in range(determinizations):
                root = search_begin(
                    observation,
                    **hybrid_agent._determinize(observation.current, me, frozen_deck),
                )
                try:
                    for candidate in candidates:
                        try:
                            state = search_step(root.searchId, [candidate])
                            steps = 0
                            while (state.observation.current.result < 0
                                   and state.observation.select is not None
                                   and state.observation.current.yourIndex == me
                                   and steps < max_roll):
                                ranked_next = domain_policy.rank_options(state.observation)
                                nested = state.observation.select
                                count = nested.maxCount or 1
                                action = [i for i in ranked_next
                                          if 0 <= i < len(nested.option)][:count] or [0]
                                state = search_step(state.searchId, action)
                                steps += 1
                            # Optional second ply: let the opponent greedily
                            # complete one turn before valuing our root choice.
                            # This directly addresses the own-turn myopia of the
                            # original wrapper without mutating frozen Grim state.
                            opponent_steps = 0
                            while (opponent_roll > 0
                                   and state.observation.current.result < 0
                                   and state.observation.select is not None
                                   and state.observation.current.yourIndex != me
                                   and opponent_steps < opponent_roll):
                                opponent_ranked = domain_policy.rank_options(state.observation)
                                nested = state.observation.select
                                count = nested.maxCount or 1
                                action = [i for i in opponent_ranked
                                          if 0 <= i < len(nested.option)][:count] or [0]
                                state = search_step(state.searchId, action)
                                opponent_steps += 1
                            totals[candidate] += domain_policy.evaluate_board(
                                state.observation.current, me)
                            visits[candidate] += 1
                        except Exception:
                            continue
                finally:
                    search_end()

            if visits[base_index] == 0:
                return baseline
            values = {i: totals[i] / visits[i] for i in candidates if visits[i]}
            best = max(values, key=values.get)
            effective_margin = margin
            if margin_late is not None:
                try:
                    if len(observation.current.players[me].prize) <= late_prizes:
                        effective_margin = margin_late
                except Exception:
                    pass
            if (best != base_index
                    and values[best] >= values[base_index] + effective_margin):
                return [best]
        except Exception:
            try:
                search_end()
            except Exception:
                pass
        return baseline

    return run


def visible_family_gate(search_agent, base_agent, opponent_card_ids):
    """Enable a search wrapper only after a public archetype signature appears."""
    signatures = set(int(value) for value in opponent_card_ids)

    def run(obs_dict):
        try:
            current = obs_dict.get("current") or {}
            me = int(current.get("yourIndex", 0) or 0)
            players = current.get("players") or []
            if len(players) == 2:
                opponent = players[1 - me]
                visible = list(opponent.get("active") or [])
                visible += list(opponent.get("bench") or [])
                visible += list(opponent.get("discard") or [])
                if any(int((card or {}).get("id", 0) or 0) in signatures
                       for card in visible):
                    return search_agent(obs_dict)
        except Exception:
            pass
        return base_agent(obs_dict)

    return run
