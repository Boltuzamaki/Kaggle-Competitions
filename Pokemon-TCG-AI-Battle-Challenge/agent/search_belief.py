"""v3 determinized search with a real opponent-deck model.

Identical to search_agent except for one line of the determinization: instead of
filling the opponent's deck with placeholder Basics, we roll out against the meta
list their revealed cards actually match (see opponent_deck_model).

Keeping everything else fixed makes this a controlled test of whether
determinization bias -- not sample count -- is what limits the search.
"""

from __future__ import annotations

import random

import search_agent
from search_agent import _rollout_pick, _value, _heuristic, _MAXROLL, _HAS_SEARCH, _DET
from opponent_deck_model import opponent_deck_sample, identify

try:
    from cg.api import to_observation_class, search_begin, search_step, search_end
except Exception:  # pragma: no cover
    _HAS_SEARCH = False

_RNG = random.Random(0xC0FFEE)

# Diagnostics: how often the model commits to an archetype.
STATS = {"identified": 0, "fallback": 0}


def _determinize_belief(cur, me):
    mp, op = cur.players[me], cur.players[1 - me]
    deck = list(search_agent.DECK)
    opp_deck = opponent_deck_sample(op, _RNG)
    name, score = identify(op)
    STATS["identified" if (name and score >= 2) else "fallback"] += 1
    return dict(
        your_deck=[_RNG.choice(deck) for _ in range(mp.deckCount)],
        your_prize=[_RNG.choice(deck) for _ in range(len(mp.prize))],
        opponent_deck=opp_deck,
        opponent_prize=[opp_deck[0] if opp_deck else 1] * len(op.prize),
        opponent_hand=[opp_deck[0] if opp_deck else 1] * op.handCount,
        opponent_active=[opp_deck[0] if opp_deck else 1]
        if (op.active and op.active[0] is None) else [],
    )


def _search_choice(obs):
    O = to_observation_class(obs)
    me = O.current.yourIndex
    n = len(O.select.option)
    vals = [0.0] * n
    for _ in range(_DET):
        ss = search_begin(O, **_determinize_belief(O.current, me))
        try:
            for i in range(n):
                try:
                    st = search_step(ss.searchId, [i])
                    cur = st.observation.current
                    g = 0
                    while (cur.result < 0 and st.observation.select is not None
                           and cur.yourIndex == me and g < _MAXROLL):
                        st = search_step(st.searchId, _rollout_pick(st))
                        cur = st.observation.current
                        g += 1
                    vals[i] += _value(cur, me)
                except Exception:
                    vals[i] += -1e9
        finally:
            try:
                search_end()
            except Exception:
                pass
    return max(range(n), key=lambda i: vals[i])


def agent(obs_dict: dict) -> list[int]:
    try:
        select = obs_dict.get("select")
        if select is None:
            return list(search_agent.DECK)
        options = select.get("option") or []
        max_count = select.get("maxCount", 1) or 1
        n = len(options)
        if n == 0:
            return []
        if _HAS_SEARCH and max_count == 1 and n > 1:
            try:
                return [_search_choice(obs_dict)]
            except Exception:
                try:
                    search_end()
                except Exception:
                    pass
        return _heuristic(obs_dict)
    except Exception:
        try:
            select = obs_dict.get("select")
            if select is None:
                return list(search_agent.DECK)
            n = len(select.get("option") or [])
            mc = select.get("maxCount", 1) or 1
            return list(range(min(mc, n))) if n else []
        except Exception:
            return list(search_agent.DECK) or [0]


def make_agent(deck):
    frozen = list(deck)

    def run(obs_dict):
        search_agent.DECK = frozen
        return agent(obs_dict)
    return run
