"""Clean-room imperfect-information search variants for Archaludon.

The implementations here deliberately share statistics across fresh hidden-state
samples.  Public agents are not imported and public actions are never targets.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import time

import archaludon_policy
import hybrid_agent

try:
    from cg.api import SelectContext, search_begin, search_end, search_step, to_observation_class
    _HAS_SEARCH = True
except Exception:
    _HAS_SEARCH = False


def _scalar(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


_OPTION_FIELDS = (
    "type", "number", "area", "index", "playerIndex", "cardId", "serial",
    "attackId", "inPlayArea", "inPlayIndex", "toolIndex", "energyIndex", "count",
)


def option_signature(option):
    """Stable action identity usable across determinizations."""
    return tuple(_scalar(getattr(option, name, None)) for name in _OPTION_FIELDS)


def _pokemon_key(pokemon):
    if pokemon is None:
        return None
    return (
        pokemon.id,
        pokemon.hp,
        tuple(sorted(getattr(energy, "id", 0) for energy in (pokemon.energies or []))),
        tuple(sorted(getattr(tool, "id", 0) for tool in (pokemon.tools or []))),
    )


def public_information_key(observation):
    """Observation key with only information visible to the acting player."""
    current = observation.current
    me = current.yourIndex
    players = []
    for seat, player in enumerate(current.players):
        hand = tuple(
            sorted(card.id for card in (player.hand or []) if card is not None)
        ) if seat == me else player.handCount
        players.append(
            (
                tuple(_pokemon_key(pokemon) for pokemon in player.active),
                tuple(_pokemon_key(pokemon) for pokemon in player.bench),
                hand,
                tuple(sorted(card.id for card in (player.discard or []))),
                player.deckCount,
                len(player.prize),
            )
        )
    select = observation.select
    legal = tuple(option_signature(option) for option in (select.option if select else []))
    return (
        me,
        current.turn,
        current.result,
        tuple(players),
        _scalar(getattr(select, "context", None)),
        legal,
    )


def _ranked_actions(observation, width):
    select = observation.select
    if select is None or not select.option:
        return []
    ranked = [
        index for index in archaludon_policy.rank_options(observation)
        if 0 <= index < len(select.option)
    ]
    if not ranked:
        ranked = list(range(len(select.option)))
    seen = Counter()
    actions = []
    for index in ranked[:width]:
        base = option_signature(select.option[index])
        signature = base + (seen[base],)
        seen[base] += 1
        actions.append((signature, index))
    return actions


def _rollout(state, me, max_steps=18):
    current = state.observation.current
    steps = 0
    while (
        current.result < 0
        and state.observation.select is not None
        and current.yourIndex == me
        and steps < max_steps
    ):
        select = state.observation.select
        ranked = archaludon_policy.rank_options(state.observation)
        maximum = min(len(select.option), select.maxCount or 1)
        minimum = max(0, select.minCount or 0)
        count = max(minimum, maximum)
        action = [index for index in ranked if 0 <= index < len(select.option)][:count]
        if len(action) < minimum:
            action = list(range(minimum))
        if not action and maximum:
            action = [0]
        state = search_step(state.searchId, action)
        current = state.observation.current
        steps += 1
    return archaludon_policy.evaluate_board(current, me)


def _simulation_state_key(observation):
    """Hash the determinized state available inside a search simulation."""
    current = observation.current
    players = []
    for player in current.players:
        players.append(
            (
                tuple(_pokemon_key(pokemon) for pokemon in player.active),
                tuple(_pokemon_key(pokemon) for pokemon in player.bench),
                tuple(sorted(getattr(card, "id", 0) for card in (player.hand or []))),
                tuple(sorted(getattr(card, "id", 0) for card in (player.discard or []))),
                player.deckCount,
                tuple(sorted(getattr(card, "id", 0) for card in (player.prize or []))),
            )
        )
    select = observation.select
    legal = tuple(option_signature(option) for option in (select.option if select else []))
    return (
        current.yourIndex,
        current.turn,
        current.result,
        tuple(players),
        _scalar(getattr(select, "context", None)),
        legal,
    )


def _cached_rollout(state, me, cache, max_steps=18):
    key = (_simulation_state_key(state.observation), me, max_steps)
    if key in cache:
        return cache[key], True
    value = _rollout(state, me, max_steps=max_steps)
    cache[key] = value
    return value, False


def _fresh_root(observation, me, deck):
    hybrid_agent._CORRECT_OWN_DET = True
    hybrid_agent._OPPONENT_BELIEF = True
    return search_begin(
        observation,
        **hybrid_agent._determinize(observation.current, me, deck),
    )


def flat_root_choice(obs_dict, deck, iterations=24, root_width=6):
    """Root-sampling flat Monte Carlo with one fresh sample per iteration."""
    observation = to_observation_class(obs_dict)
    me = observation.current.yourIndex
    override = archaludon_policy.hard_override(observation)
    if override is not None:
        return override
    actions = _ranked_actions(observation, root_width)
    if len(actions) <= 1:
        return actions[0][1] if actions else 0
    totals = {signature: 0.0 for signature, _ in actions}
    visits = {signature: 0 for signature, _ in actions}
    for iteration in range(max(iterations, len(actions))):
        signature, index = actions[iteration % len(actions)]
        try:
            root = _fresh_root(observation, me, deck)
            child = search_step(root.searchId, [index])
            totals[signature] += _rollout(child, me)
            visits[signature] += 1
        except Exception:
            totals[signature] -= 1e12
            visits[signature] += 1
        finally:
            try:
                search_end()
            except Exception:
                pass
    return max(actions, key=lambda item: totals[item[0]] / max(visits[item[0]], 1))[1]


def cached_flat_root_choice(obs_dict, deck, iterations=24, root_width=6):
    """Flat root search with a per-decision determinized-state value cache."""
    observation = to_observation_class(obs_dict)
    me = observation.current.yourIndex
    override = archaludon_policy.hard_override(observation)
    if override is not None:
        return override
    actions = _ranked_actions(observation, root_width)
    if len(actions) <= 1:
        return actions[0][1] if actions else 0
    totals = {signature: 0.0 for signature, _ in actions}
    visits = {signature: 0 for signature, _ in actions}
    cache = {}
    for iteration in range(max(iterations, len(actions))):
        signature, index = actions[iteration % len(actions)]
        try:
            root = _fresh_root(observation, me, deck)
            child = search_step(root.searchId, [index])
            value, _hit = _cached_rollout(child, me, cache)
            totals[signature] += value
            visits[signature] += 1
        except Exception:
            totals[signature] -= 1e12
            visits[signature] += 1
        finally:
            try:
                search_end()
            except Exception:
                pass
    return max(actions, key=lambda item: totals[item[0]] / max(visits[item[0]], 1))[1]


def progressive_widening_choice(obs_dict, deck, iterations=36, root_width=10):
    """Grow the searched root set as evidence accumulates."""
    observation = to_observation_class(obs_dict)
    me = observation.current.yourIndex
    override = archaludon_policy.hard_override(observation)
    if override is not None:
        return override
    actions = _ranked_actions(observation, root_width)
    if len(actions) <= 1:
        return actions[0][1] if actions else 0
    totals = {signature: 0.0 for signature, _ in actions}
    visits = {signature: 0 for signature, _ in actions}
    for iteration in range(max(iterations, len(actions))):
        active_count = min(
            len(actions),
            max(2, int(math.ceil(1.4 * math.sqrt(iteration + 1)))),
        )
        active = actions[:active_count]
        signature, index = min(
            active,
            key=lambda item: (
                visits[item[0]],
                -(totals[item[0]] / max(visits[item[0]], 1)),
            ),
        )
        try:
            root = _fresh_root(observation, me, deck)
            child = search_step(root.searchId, [index])
            totals[signature] += _rollout(child, me)
            visits[signature] += 1
        except Exception:
            totals[signature] -= 1e12
            visits[signature] += 1
        finally:
            try:
                search_end()
            except Exception:
                pass
    viable = [item for item in actions if visits[item[0]]]
    return max(
        viable,
        key=lambda item: totals[item[0]] / visits[item[0]],
    )[1]


def risk_sensitive_choice(
    obs_dict,
    deck,
    iterations=36,
    root_width=6,
    downside_weight=0.45,
):
    """Prefer high mean value while penalizing poor hidden-state outcomes."""
    observation = to_observation_class(obs_dict)
    me = observation.current.yourIndex
    override = archaludon_policy.hard_override(observation)
    if override is not None:
        return override
    actions = _ranked_actions(observation, root_width)
    if len(actions) <= 1:
        return actions[0][1] if actions else 0
    samples = {signature: [] for signature, _ in actions}
    for iteration in range(max(iterations, len(actions))):
        signature, index = actions[iteration % len(actions)]
        try:
            root = _fresh_root(observation, me, deck)
            child = search_step(root.searchId, [index])
            samples[signature].append(_rollout(child, me))
        except Exception:
            samples[signature].append(-1e12)
        finally:
            try:
                search_end()
            except Exception:
                pass

    def robust_value(signature):
        values = samples[signature]
        if not values:
            return -1e12
        mean = sum(values) / len(values)
        downside = [
            (mean - value) ** 2
            for value in values
            if value < mean
        ]
        downside_deviation = math.sqrt(sum(downside) / max(len(downside), 1))
        return mean - downside_weight * downside_deviation

    return max(actions, key=lambda item: robust_value(item[0]))[1]


def chance_coupled_choice(obs_dict, deck, iterations=12, root_width=6):
    """Compare every root action under each shared hidden-state sample."""
    observation = to_observation_class(obs_dict)
    me = observation.current.yourIndex
    override = archaludon_policy.hard_override(observation)
    if override is not None:
        return override
    actions = _ranked_actions(observation, root_width)
    if len(actions) <= 1:
        return actions[0][1] if actions else 0
    totals = {signature: 0.0 for signature, _ in actions}
    visits = {signature: 0 for signature, _ in actions}
    for _ in range(max(1, iterations)):
        try:
            root = _fresh_root(observation, me, deck)
            for signature, index in actions:
                try:
                    child = search_step(root.searchId, [index])
                    totals[signature] += _rollout(child, me)
                except Exception:
                    totals[signature] -= 1e12
                visits[signature] += 1
        finally:
            try:
                search_end()
            except Exception:
                pass
    return max(
        actions,
        key=lambda item: totals[item[0]] / max(visits[item[0]], 1),
    )[1]


@dataclass
class ActionStat:
    visits: int = 0
    available: int = 0
    total: float = 0.0

    @property
    def mean(self):
        return self.total / max(self.visits, 1)


@dataclass
class InfoNode:
    visits: int = 0
    actions: dict[tuple, ActionStat] = field(default_factory=dict)


def ismcts_choice(obs_dict, deck, iterations=32, root_width=6, child_width=3, max_steps=16):
    """Single-observer ISMCTS over one shared availability-aware tree."""
    root_observation = to_observation_class(obs_dict)
    me = root_observation.current.yourIndex
    override = archaludon_policy.hard_override(root_observation)
    if override is not None:
        return override
    root_actions = _ranked_actions(root_observation, root_width)
    if len(root_actions) <= 1:
        return root_actions[0][1] if root_actions else 0

    tree: dict[tuple, InfoNode] = {}
    root_key = (public_information_key(root_observation), ())
    exploration = 1.15
    for _ in range(max(iterations, len(root_actions))):
        traversed = []
        path = []
        try:
            state = _fresh_root(root_observation, me, deck)
            for depth in range(max_steps):
                observation = state.observation
                current = observation.current
                if current.result >= 0 or observation.select is None or current.yourIndex != me:
                    break
                available = _ranked_actions(
                    observation,
                    root_width if depth == 0 else child_width,
                )
                if not available:
                    break
                key = (public_information_key(observation), tuple(path))
                node = tree.setdefault(key, InfoNode())
                for signature, _index in available:
                    stat = node.actions.setdefault(signature, ActionStat())
                    stat.available += 1

                unvisited = [item for item in available if node.actions[item[0]].visits == 0]
                if unvisited:
                    signature, index = unvisited[0]
                else:
                    log_available = math.log(max(node.visits, 1) + 1.0)
                    signature, index = max(
                        available,
                        key=lambda item: (
                            node.actions[item[0]].mean
                            + exploration
                            * math.sqrt(log_available / node.actions[item[0]].visits)
                        ),
                    )
                stat = node.actions[signature]
                traversed.append((node, stat))
                path.append(signature)
                state = search_step(state.searchId, [index])

            value = archaludon_policy.evaluate_board(state.observation.current, me)
            for node, stat in traversed:
                node.visits += 1
                stat.visits += 1
                stat.total += value
        except Exception:
            for node, stat in traversed:
                node.visits += 1
                stat.visits += 1
                stat.total -= 1e12
        finally:
            try:
                search_end()
            except Exception:
                pass

    root = tree.get(root_key)
    if root is None:
        return root_actions[0][1]
    viable = [item for item in root_actions if root.actions.get(item[0], ActionStat()).visits]
    if not viable:
        return root_actions[0][1]
    # Robust-child is the standard final ISMCTS decision; mean breaks visit ties.
    return max(
        viable,
        key=lambda item: (root.actions[item[0]].visits, root.actions[item[0]].mean),
    )[1]


def strategy_fusion_probe(obs_dict, deck, samples=8, root_width=6):
    """Measure per-determinization preferred-root disagreement for diagnostics."""
    observation = to_observation_class(obs_dict)
    me = observation.current.yourIndex
    actions = _ranked_actions(observation, root_width)
    winners = []
    for _ in range(samples):
        values = {}
        try:
            root = _fresh_root(observation, me, deck)
            for signature, index in actions:
                try:
                    values[signature] = _rollout(search_step(root.searchId, [index]), me)
                except Exception:
                    values[signature] = -1e12
            if values:
                winners.append(max(values, key=values.get))
        finally:
            try:
                search_end()
            except Exception:
                pass
    counts = Counter(winners)
    agreement = max(counts.values(), default=0) / max(len(winners), 1)
    return {
        "samples": len(winners),
        "distinct_preferred_actions": len(counts),
        "agreement": agreement,
        "disagreement": 1.0 - agreement,
        "preferred_counts": {str(key): value for key, value in counts.items()},
    }


def _eligible(obs_dict):
    if not _HAS_SEARCH or obs_dict.get("select") is None:
        return False
    observation = to_observation_class(obs_dict)
    return bool(
        observation.select
        and observation.select.context == SelectContext.MAIN
        and (observation.select.maxCount or 1) == 1
        and len(observation.select.option) > 1
    )


def flat_root_agent_config(obs_dict, deck, iterations=24, root_width=6):
    if obs_dict.get("select") is None:
        return list(deck)
    try:
        if _eligible(obs_dict):
            return [
                flat_root_choice(
                    obs_dict,
                    deck,
                    iterations=iterations,
                    root_width=root_width,
                )
            ]
    except Exception:
        pass
    return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)


def flat_root_agent(obs_dict, deck):
    return flat_root_agent_config(obs_dict, deck)


def cached_flat_root_agent(obs_dict, deck):
    if obs_dict.get("select") is None:
        return list(deck)
    try:
        if _eligible(obs_dict):
            return [cached_flat_root_choice(obs_dict, deck)]
    except Exception:
        pass
    return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)


def progressive_widening_agent(obs_dict, deck):
    if obs_dict.get("select") is None:
        return list(deck)
    try:
        if _eligible(obs_dict):
            return [progressive_widening_choice(obs_dict, deck)]
    except Exception:
        pass
    return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)


def risk_sensitive_agent(obs_dict, deck):
    if obs_dict.get("select") is None:
        return list(deck)
    try:
        if _eligible(obs_dict):
            return [risk_sensitive_choice(obs_dict, deck)]
    except Exception:
        pass
    return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)


def chance_coupled_agent(obs_dict, deck):
    if obs_dict.get("select") is None:
        return list(deck)
    try:
        if _eligible(obs_dict):
            return [chance_coupled_choice(obs_dict, deck)]
    except Exception:
        pass
    return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)


def adaptive_budget_agent(obs_dict, deck):
    """Scale root-search work by branch count and public decision criticality."""
    if obs_dict.get("select") is None:
        return list(deck)
    try:
        if not _eligible(obs_dict):
            return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)
        observation = to_observation_class(obs_dict)
        override = archaludon_policy.hard_override(observation)
        if override is not None:
            return [override]
        policy = archaludon_policy.SequencedArchaludonPolicy(observation)
        scores = sorted(
            (policy._main_score(option) for option in observation.select.option),
            reverse=True,
        )
        margin = scores[0] - scores[1] if len(scores) > 1 else 1e12
        branches = len(observation.select.option)
        prize_critical = (
            len(policy.mine.prize) <= 2
            or len(policy.theirs.prize) <= 2
        )
        if margin <= 30000.0 or prize_critical:
            iterations = 32
            width = min(8, branches)
        elif branches <= 3 or margin >= 120000.0:
            iterations = 8
            width = min(3, branches)
        else:
            iterations = 16
            width = min(6, branches)
        return [
            flat_root_choice(
                obs_dict,
                deck,
                iterations=iterations,
                root_width=width,
            )
        ]
    except Exception:
        return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)


def disagreement_fallback_agent(obs_dict, deck):
    """Escalate only when the domain policy and a small search disagree."""
    policy_action = archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)
    if obs_dict.get("select") is None:
        return policy_action
    try:
        if not _eligible(obs_dict) or not policy_action:
            return policy_action
        search_action = flat_root_choice(
            obs_dict,
            deck,
            iterations=12,
            root_width=6,
        )
        if search_action == policy_action[0]:
            return policy_action
        tie_break = risk_sensitive_choice(
            obs_dict,
            deck,
            iterations=18,
            root_width=6,
            downside_weight=0.35,
        )
        if tie_break == search_action:
            return [search_action]
    except Exception:
        pass
    return policy_action


def gated_flat_root_agent(obs_dict, deck):
    """Use the stronger flat search only for close or prize-critical choices."""
    if obs_dict.get("select") is None:
        return list(deck)
    try:
        if not _eligible(obs_dict):
            return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)
        observation = to_observation_class(obs_dict)
        override = archaludon_policy.hard_override(observation)
        if override is not None:
            return [override]
        policy = archaludon_policy.SequencedArchaludonPolicy(observation)
        scores = sorted(
            (policy._main_score(option) for option in observation.select.option),
            reverse=True,
        )
        close = len(scores) >= 2 and scores[0] - scores[1] <= 30000.0
        active = policy.active()
        target = policy.opponent_active()
        legal_knockout = any(
            option.type == archaludon_policy.OptionType.ATTACK
            and archaludon_policy._damage(
                getattr(option, "attackId", None), active, target
            ) >= target.hp
            for option in observation.select.option
        ) if active is not None and target is not None else False
        prize_critical = (
            len(policy.mine.prize) <= 2
            or len(policy.theirs.prize) <= 2
            or legal_knockout
        )
        if not close and not prize_critical:
            return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)
        return [flat_root_choice(obs_dict, deck, iterations=16, root_width=6)]
    except Exception:
        return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)


def ismcts_agent(obs_dict, deck):
    if obs_dict.get("select") is None:
        return list(deck)
    try:
        if _eligible(obs_dict):
            return [ismcts_choice(obs_dict, deck)]
    except Exception:
        pass
    return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)


def gated_ismcts_agent(obs_dict, deck):
    """Spend ISMCTS only when our heuristic has a genuinely close decision."""
    if obs_dict.get("select") is None:
        return list(deck)
    try:
        if not _eligible(obs_dict):
            return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)
        observation = to_observation_class(obs_dict)
        override = archaludon_policy.hard_override(observation)
        if override is not None:
            return [override]
        policy = archaludon_policy.SequencedArchaludonPolicy(observation)
        scores = sorted(
            (policy._main_score(option) for option in observation.select.option),
            reverse=True,
        )
        if len(scores) < 2 or scores[0] - scores[1] > 50000.0:
            return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)
        started = time.perf_counter()
        choice = ismcts_choice(obs_dict, deck, iterations=20)
        _ = time.perf_counter() - started
        return [choice]
    except Exception:
        return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)
