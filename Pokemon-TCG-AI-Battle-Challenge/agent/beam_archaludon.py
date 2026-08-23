"""Full-turn beam search for the clean-room Archaludon policy."""

from __future__ import annotations

from dataclasses import dataclass

import archaludon_policy
import hybrid_agent

try:
    from cg.api import (
        OptionType,
        SelectContext,
        search_begin,
        search_end,
        search_step,
        to_observation_class,
    )
    _HAS_SEARCH = True
except Exception:
    _HAS_SEARCH = False


@dataclass
class _Node:
    search_id: int
    observation: object
    root: int | None
    score: float
    terminal: bool = False


def _pokemon_key(pokemon):
    if pokemon is None:
        return None
    return (
        pokemon.id,
        pokemon.hp,
        tuple(sorted(energy.id for energy in (pokemon.energies or []))),
        tuple(sorted(tool.id for tool in (pokemon.tools or []))),
    )


def _state_key(observation):
    current = observation.current
    players = []
    for player in current.players:
        players.append(
            (
                tuple(_pokemon_key(pokemon) for pokemon in player.active),
                tuple(_pokemon_key(pokemon) for pokemon in player.bench),
                tuple(sorted(card.id for card in (player.hand or []) if card is not None)),
                tuple(sorted(card.id for card in (player.discard or []))),
                player.deckCount,
                len(player.prize),
            )
        )
    context = getattr(observation.select, "context", None) if observation.select else None
    return (current.yourIndex, current.turn, current.result, tuple(players), context)


def _actions(observation, root, beam_width):
    select = observation.select
    if select is None or not select.option:
        return []
    ranked = archaludon_policy.rank_options(observation)
    ranked = [index for index in ranked if 0 <= index < len(select.option)]
    minimum = max(0, select.minCount or 0)
    maximum = min(len(select.option), select.maxCount or 1)
    if maximum != 1:
        count = max(minimum, maximum)
        return [ranked[:count] or list(range(count))]
    branch = beam_width if root is None else 2
    return [[index] for index in ranked[:branch]]


def _opponent_action(observation):
    select = observation.select
    if select is None or not select.option:
        return []
    maximum = min(len(select.option), select.maxCount or 1)
    minimum = max(0, select.minCount or 0)
    priorities = {
        OptionType.ABILITY: 90.0,
        OptionType.EVOLVE: 80.0,
        OptionType.ATTACH: 70.0,
        OptionType.PLAY: 60.0,
        OptionType.ATTACK: 50.0,
        OptionType.RETREAT: 10.0,
        OptionType.YES: 5.0,
        OptionType.NO: 4.0,
        OptionType.END: -100.0,
    }
    scores = []
    for option in select.option:
        score = priorities.get(option.type, 1.0)
        if option.type == OptionType.NUMBER:
            score += getattr(option, "number", 0)
        scores.append(score)
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    count = max(minimum, maximum)
    return ranked[:count]


def _opponent_response(node, me, max_steps=10):
    current = node.observation.current
    for _ in range(max_steps):
        if (
            current.result >= 0
            or node.observation.select is None
            or current.yourIndex == me
        ):
            break
        action = _opponent_action(node.observation)
        if not action:
            break
        try:
            step = search_step(node.search_id, action)
        except Exception:
            break
        node = _Node(
            step.searchId,
            step.observation,
            node.root,
            archaludon_policy.evaluate_board(step.observation.current, me),
            step.observation.current.result >= 0,
        )
        current = step.observation.current
    return node


def _beam_choice(obs_dict, deck, beam_width, max_steps=12, opponent_response=False):
    observation = to_observation_class(obs_dict)
    me = observation.current.yourIndex
    override = archaludon_policy.hard_override(observation)
    if override is not None:
        return override

    hybrid_agent._CORRECT_OWN_DET = False
    hybrid_agent._OPPONENT_BELIEF = opponent_response
    seed = search_begin(
        observation,
        **hybrid_agent._determinize(observation.current, me, deck),
    )
    beam = [_Node(seed.searchId, observation, None, archaludon_policy.evaluate_board(observation.current, me))]

    for _ in range(max_steps):
        children = []
        for node in beam:
            current = node.observation.current
            finished = (
                node.terminal
                or current.result >= 0
                or node.observation.select is None
                or current.yourIndex != me
            )
            if finished:
                node.terminal = True
                children.append(node)
                continue
            actions = _actions(node.observation, node.root, beam_width)
            if not actions:
                node.terminal = True
                children.append(node)
                continue
            for action in actions:
                try:
                    step = search_step(node.search_id, action)
                    root = action[0] if node.root is None else node.root
                    terminal = (
                        step.observation.current.result >= 0
                        or step.observation.select is None
                        or step.observation.current.yourIndex != me
                    )
                    children.append(
                        _Node(
                            step.searchId,
                            step.observation,
                            root,
                            archaludon_policy.evaluate_board(step.observation.current, me),
                            terminal,
                        )
                    )
                except Exception:
                    continue
        if not children:
            break
        deduplicated = {}
        for child in children:
            key = _state_key(child.observation)
            previous = deduplicated.get(key)
            if previous is None or child.score > previous.score:
                deduplicated[key] = child
        beam = sorted(deduplicated.values(), key=lambda node: node.score, reverse=True)[:beam_width]
        if all(node.terminal for node in beam):
            break

    roots = [node for node in beam if node.root is not None]
    if opponent_response:
        roots = [_opponent_response(node, me) for node in roots]
    if not roots:
        raise RuntimeError("beam produced no root action")
    return max(roots, key=lambda node: node.score).root


def beam_agent(obs_dict, deck, beam_width, opponent_response=False):
    try:
        if obs_dict.get("select") is None:
            return list(deck)
        observation = to_observation_class(obs_dict)
        if (
            not _HAS_SEARCH
            or observation.select is None
            or observation.select.context != SelectContext.MAIN
            or (observation.select.maxCount or 1) != 1
            or len(observation.select.option) <= 1
        ):
            return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)
        return [_beam_choice(obs_dict, deck, beam_width, opponent_response=opponent_response)]
    except Exception:
        try:
            search_end()
        except Exception:
            pass
        return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)
    finally:
        try:
            search_end()
        except Exception:
            pass
