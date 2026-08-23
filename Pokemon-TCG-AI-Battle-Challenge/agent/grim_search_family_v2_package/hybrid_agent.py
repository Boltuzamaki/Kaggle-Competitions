"""
Hybrid agent (v5): our own domain-knowledge policy (domain_policy.py) DRIVING a
determinized search — the combination the meta-analysis (PLAN.md Phase 2 +
top-ranker study) points to: "search should refine a strong heuristic, not
replace it."

Differs from search_agent.py (v3) in one key way: v3's rollout/candidate-ranking
uses generic OptionType priority; this uses our domain_policy (real prize math,
weakness x2, KO detection, threat prediction) for BOTH the candidate shortlist
at the root AND the policy used to finish the simulated turn, and scores the
resulting board with domain_policy.evaluate_board (also domain-aware) instead of
a generic prize-diff+HP value.

Entry point: hybrid_agent(obs_dict, deck) -> list[int]. Always falls back to the
pure domain_policy heuristic (no search) on any failure -- never crashes.
"""

from __future__ import annotations

import random
from collections import Counter

import domain_policy
from opponent_beliefs import META_PROFILES

try:
    from cg.api import (
        all_card_data,
        to_observation_class,
        search_begin,
        search_step,
        search_end,
    )
    _HAS_SEARCH = True
except Exception:
    _HAS_SEARCH = False

_SNORLAX = 1072  # a Basic Pokemon, for legal opponent determinization
_TOP_K = 6        # only search the domain policy's top-K candidates at the root
_DET = 3          # determinizations averaged per decision
_MAXROLL = 24     # cap on simulated rollout steps (finishing my turn)
_CORRECT_OWN_DET = False
_OPPONENT_BELIEF = False

if _HAS_SEARCH:
    _CARD_DATA = {card.cardId: card for card in all_card_data()}
    _NAME_TO_IDS = {}
    for _card_data in _CARD_DATA.values():
        _NAME_TO_IDS.setdefault(_card_data.name, []).append(_card_data.cardId)
else:
    _CARD_DATA = {}
    _NAME_TO_IDS = {}


def _card_id(value):
    if isinstance(value, int):
        return value
    return getattr(value, "id", None)


def _remove_visible(counts, card_id, deck_ids, include_evolution=False):
    if card_id is None:
        return
    if counts[card_id] > 0:
        counts[card_id] -= 1
    if not include_evolution:
        return
    data = _CARD_DATA.get(card_id)
    evolves_from = getattr(data, "evolvesFrom", None) if data else None
    if not evolves_from:
        return
    for previous_id in _NAME_TO_IDS.get(evolves_from, []):
        if previous_id in deck_ids and counts[previous_id] > 0:
            _remove_visible(counts, previous_id, deck_ids, include_evolution=True)
            break


def _joint_own_hidden(mp, deck):
    counts = Counter(deck)
    deck_ids = set(counts)
    for card in list(mp.hand or []) + list(mp.discard or []):
        _remove_visible(counts, _card_id(card), deck_ids)
    for pokemon in list(mp.active or []) + list(mp.bench or []):
        if pokemon is None:
            continue
        _remove_visible(counts, getattr(pokemon, "id", None), deck_ids, include_evolution=True)
        for energy in getattr(pokemon, "energies", None) or []:
            _remove_visible(counts, _card_id(energy), deck_ids)
        for tool in getattr(pokemon, "tools", None) or []:
            _remove_visible(counts, _card_id(tool), deck_ids)

    unknown = list(counts.elements())
    needed = mp.deckCount + len(mp.prize)
    random.shuffle(unknown)
    if len(unknown) < needed:
        unknown.extend(random.choices(deck, k=needed - len(unknown)))
    unknown = unknown[:needed]
    return unknown[: mp.deckCount], unknown[mp.deckCount :]


def _joint_opponent_hidden(op):
    observed = []
    for card in list(op.hand or []) + list(op.discard or []):
        card_id = _card_id(card)
        if card_id is not None:
            observed.append(card_id)
    for pokemon in list(op.active or []) + list(op.bench or []):
        if pokemon is None:
            continue
        observed.append(pokemon.id)
        observed.extend(
            card_id for energy in (getattr(pokemon, "energies", None) or [])
            if (card_id := _card_id(energy)) is not None
        )
        observed.extend(
            card_id for tool in (getattr(pokemon, "tools", None) or [])
            if (card_id := _card_id(tool)) is not None
        )

    def profile_score(profile):
        counts = Counter(profile)
        score = 0.0
        for card_id in observed:
            data = _CARD_DATA.get(card_id)
            weight = 4.0 if data is not None and getattr(data, "cardType", None) == 0 else 1.0
            score += weight if counts[card_id] > 0 else -weight * 2.0
        return score

    profile = max(META_PROFILES.values(), key=profile_score)
    if not observed or profile_score(profile) <= 0:
        return None

    counts = Counter(profile)
    deck_ids = set(counts)
    for card_id in observed:
        include_evolution = bool(
            _CARD_DATA.get(card_id) and getattr(_CARD_DATA[card_id], "cardType", None) == 0
        )
        _remove_visible(counts, card_id, deck_ids, include_evolution=include_evolution)

    hidden_active = bool(op.active and op.active[0] is None)
    needed = op.deckCount + len(op.prize) + op.handCount + int(hidden_active)
    unknown = list(counts.elements())
    random.shuffle(unknown)
    if len(unknown) < needed:
        unknown.extend(random.choices(profile, k=needed - len(unknown)))

    active_cards = []
    if hidden_active:
        basic_index = next(
            (
                index
                for index, card_id in enumerate(unknown)
                if _CARD_DATA.get(card_id) is not None
                and getattr(_CARD_DATA[card_id], "basic", False)
            ),
            None,
        )
        active_cards = [unknown.pop(basic_index)] if basic_index is not None else [_SNORLAX]

    unknown = unknown[: op.deckCount + len(op.prize) + op.handCount]
    deck_end = op.deckCount
    prize_end = deck_end + len(op.prize)
    return (
        unknown[:deck_end],
        unknown[deck_end:prize_end],
        unknown[prize_end:],
        active_cards,
    )


def _determinize(cur, me, deck):
    mp, op = cur.players[me], cur.players[1 - me]
    if _CORRECT_OWN_DET:
        your_deck, your_prize = _joint_own_hidden(mp, deck)
    else:
        your_deck = [random.choice(deck) for _ in range(mp.deckCount)]
        your_prize = [random.choice(deck) for _ in range(len(mp.prize))]
    opponent_hidden = _joint_opponent_hidden(op) if _OPPONENT_BELIEF else None
    if opponent_hidden is None:
        opponent_deck = [_SNORLAX] * op.deckCount
        opponent_prize = [1] * len(op.prize)
        opponent_hand = [1] * op.handCount
        opponent_active = [_SNORLAX] if (op.active and op.active[0] is None) else []
    else:
        opponent_deck, opponent_prize, opponent_hand, opponent_active = opponent_hidden
    return dict(
        your_deck=your_deck,
        your_prize=your_prize,
        opponent_deck=opponent_deck,
        opponent_prize=opponent_prize,
        opponent_hand=opponent_hand,
        opponent_active=opponent_active,
    )


def _hard_override(O, policy_module=domain_policy):
    """A guaranteed win (take a lethal attack) or a guaranteed escape (retreat
    out of confirmed lethal threat) at the ROOT decision. Bypasses search
    entirely: loss-pattern analysis showed the search could rate a setup move
    higher than a *scored* lethal/escape and pick that instead. These two
    situations are unambiguous enough to never leave to a ranking."""
    custom = getattr(policy_module, "hard_override", None)
    if callable(custom):
        return custom(O)
    sel = O.select
    if sel is None or getattr(sel, "context", None) != policy_module.SelectContext.MAIN:
        return None
    dp = policy_module.DomainPolicy(O)
    if dp.plan and dp.plan[3]:
        for i, o in enumerate(sel.option):
            if o.type == policy_module.OptionType.ATTACK and getattr(o, "attackId", None) == dp.plan[1]:
                return i
    if dp._lethal_threat():
        for i, o in enumerate(sel.option):
            if o.type == policy_module.OptionType.RETREAT:
                return i
    return None


def _search_choice(obs_dict, deck, policy_module=domain_policy):
    O = to_observation_class(obs_dict)
    me = O.current.yourIndex
    n = len(O.select.option)

    override = _hard_override(O, policy_module)
    if override is not None:
        return override

    # Root candidate shortlist, ranked by domain knowledge (not brute force).
    root_ranked = policy_module.rank_options(O)
    candidates = [i for i in root_ranked if 0 <= i < n][:_TOP_K]
    if not candidates:
        candidates = list(range(min(n, _TOP_K)))
    if len(candidates) == 1:
        return candidates[0]

    vals = {i: 0.0 for i in candidates}
    for _ in range(_DET):
        ss = search_begin(O, **_determinize(O.current, me, deck))
        for i in candidates:
            try:
                st = search_step(ss.searchId, [i])
                cur = st.observation.current
                g = 0
                while (cur.result < 0 and st.observation.select is not None
                       and cur.yourIndex == me and g < _MAXROLL):
                    pick = policy_module.rank_options(st.observation)
                    sel = st.observation.select
                    mc = sel.maxCount or 1
                    pick = [p for p in pick if 0 <= p < len(sel.option)][:mc] or [0]
                    st = search_step(st.searchId, pick)
                    cur = st.observation.current
                    g += 1
                vals[i] += policy_module.evaluate_board(cur, me)
            except Exception:
                vals[i] += -1e12
        search_end()
    return max(candidates, key=lambda i: vals[i])


def hybrid_agent(obs_dict: dict, deck: list, policy_module=domain_policy) -> list:
    try:
        select = obs_dict.get("select")
        if select is None:
            return list(deck)
        options = select.get("option") or []
        max_count = select.get("maxCount", 1) or 1
        n = len(options)
        if n == 0:
            return []

        if _HAS_SEARCH and max_count == 1 and n > 1:
            try:
                return [_search_choice(obs_dict, deck, policy_module)]
            except Exception:
                try:
                    search_end()
                except Exception:
                    pass
        return policy_module.domain_agent(obs_dict, deck)

    except Exception:
        try:
            select = obs_dict.get("select")
            if select is None:
                return list(deck)
            n = len(select.get("option") or [])
            mc = select.get("maxCount", 1) or 1
            return list(range(min(mc, n))) if n else []
        except Exception:
            return list(deck) if deck else [0]
