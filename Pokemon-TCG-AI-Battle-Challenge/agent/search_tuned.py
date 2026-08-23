"""v3 determinized search with a parameterised leaf evaluation.

search_agent._value hardcodes the weights that decide every rollout's worth:
prize differential, opponent board HP, own board HP, energy readiness and board
presence. Those numbers were never tuned, because until common random numbers
existed no local A/B could resolve a few points of win-rate.

Defaults here reproduce search_agent exactly, so `PARAMS` unchanged == stock v3.
Overrides load from value_w.json next to this module (or the Kaggle agent dir).
"""

from __future__ import annotations

import json
import os
import random

import search_agent
from search_agent import _rollout_pick, _heuristic, _board_hp, _determinize, _HAS_SEARCH

try:
    from cg.api import to_observation_class, search_begin, search_step, search_end
except Exception:  # pragma: no cover
    _HAS_SEARCH = False

# Defaults == stock v3 (search_agent._value and its search constants).
PARAMS = {
    "prize": 1000.0,
    "opp_hp": 1.0,
    "my_hp": 0.3,
    "energy": 3.0,
    "presence": 5.0,
    "det": 3.0,        # determinizations per decision
    "maxroll": 18.0,   # rollout micro-steps
}

for _p in ("value_w.json",
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "value_w.json"),
           "/kaggle_simulations/agent/value_w.json"):
    try:
        if os.path.exists(_p):
            PARAMS.update(json.load(open(_p)))
            break
    except Exception:
        pass

P = PARAMS


def _value(cur, me):
    r = cur.result
    if r >= 0:
        return 1e6 if r == me else (-1e6 if r == (1 - me) else 0.0)
    mp, op = cur.players[me], cur.players[1 - me]
    v = (len(op.prize) - len(mp.prize)) * P["prize"]
    v -= _board_hp(op) * P["opp_hp"]
    v += _board_hp(mp) * P["my_hp"]
    for a in mp.active:
        if a:
            v += P["energy"] * sum(a.energies or [])
    v += P["presence"] * (len(mp.active) + len(mp.bench))
    return v


def _search_choice(obs):
    O = to_observation_class(obs)
    me = O.current.yourIndex
    n = len(O.select.option)
    vals = [0.0] * n
    maxroll = int(P["maxroll"])
    for _ in range(max(1, int(P["det"]))):
        ss = search_begin(O, **_determinize(O.current, me))
        try:
            for i in range(n):
                try:
                    st = search_step(ss.searchId, [i])
                    cur = st.observation.current
                    g = 0
                    while (cur.result < 0 and st.observation.select is not None
                           and cur.yourIndex == me and g < maxroll):
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


def make_agent(deck, params=None):
    """Bind a deck (and optionally a parameter set) to an agent callable."""
    frozen = list(deck)
    genome = dict(params) if params else None

    def run(obs_dict):
        search_agent.DECK = frozen
        if genome:
            P.clear()
            P.update(genome)
        return agent(obs_dict)
    return run
