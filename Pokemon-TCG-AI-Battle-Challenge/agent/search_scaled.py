"""Time-budgeted scaling of our v3 determinized search.

The competition gives each player 600 seconds per game and our v3 agent spends
about 0.4 of them -- roughly 0.07% of the budget. The public timing analysis of
the top teams ("Top players' methods, revealed by 30,000 games") found the
leading agent is the one that actually consumes its clock. This module spends
that budget on more determinizations per decision, which is the one search knob
that reduces variance in a game this stochastic.

Safety first: running out of clock is an instant loss, so every guard here is
conservative. We target only a fraction of the true budget, keep a hard reserve,
re-check the deadline between determinizations, and always return a legal move.
"""

from __future__ import annotations

import os
import random
import time

import search_agent
from search_agent import (
    _determinize, _rollout_pick, _value, _heuristic, _MAXROLL, _HAS_SEARCH,
)

try:
    from cg.api import to_observation_class, search_begin, search_step, search_end
except Exception:  # pragma: no cover
    _HAS_SEARCH = False

# --- budget policy ---------------------------------------------------------
# The real limit is 600 s/game. We aim at a fraction of it: even a 10x overrun
# of our own estimate must not approach the wall.
TOTAL_BUDGET_S = float(os.environ.get("PTCG_BUDGET_S", "180.0"))
RESERVE_S = 30.0        # never spend below this much remaining
EXPECTED_DECISIONS = 220  # rough decisions per game; only sets the opening slice
MAX_DET = int(os.environ.get("PTCG_MAX_DET", "64"))
MIN_DET = 3             # never search less than stock v3

_spent = 0.0
_decisions = 0
_last_turn = -1


def _reset_game():
    global _spent, _decisions, _last_turn
    _spent = 0.0
    _decisions = 0
    _last_turn = -1


def _slice_for_now():
    """Seconds this decision may use."""
    remaining = TOTAL_BUDGET_S - _spent - RESERVE_S
    if remaining <= 0:
        return 0.0
    # Assume at least a few decisions are still to come, so we never burn the
    # whole remainder on one node.
    left = max(EXPECTED_DECISIONS - _decisions, 40)
    return max(0.0, min(remaining / left, remaining * 0.05))


def _scaled_choice(obs, deadline):
    """v3's determinized lookahead, but averaging as many samples as fit."""
    O = to_observation_class(obs)
    me = O.current.yourIndex
    n = len(O.select.option)
    vals = [0.0] * n
    runs = 0
    while runs < MAX_DET:
        ss = search_begin(O, **_determinize(O.current, me))
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
        runs += 1
        # Always complete MIN_DET samples, then stop as soon as the clock says so.
        if runs >= MIN_DET and time.time() >= deadline:
            break
    return max(range(n), key=lambda i: vals[i])


def agent(obs_dict: dict) -> list[int]:
    global _spent, _decisions, _last_turn
    try:
        select = obs_dict.get("select")
        if select is None:                 # deck selection == a new game
            _reset_game()
            return list(search_agent.DECK)

        # A turn counter that went backwards means the engine started a new game
        # in the same process; reset the clock so budgets never leak between games.
        try:
            turn = (obs_dict.get("current") or {}).get("turn", -1)
            if turn is not None and turn < _last_turn:
                _reset_game()
            _last_turn = turn if turn is not None else _last_turn
        except Exception:
            pass

        options = select.get("option") or []
        max_count = select.get("maxCount", 1) or 1
        n = len(options)
        if n == 0:
            return []

        if _HAS_SEARCH and max_count == 1 and n > 1:
            budget = _slice_for_now()
            if budget <= 0:                # out of clock: cheap heuristic only
                return _heuristic(obs_dict)
            t0 = time.time()
            try:
                pick = _scaled_choice(obs_dict, t0 + budget)
                return [pick]
            except Exception:
                try:
                    search_end()
                except Exception:
                    pass
            finally:
                _spent += time.time() - t0
                _decisions += 1
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
    """Bind a deck: returns agent(obs) -> list[int]."""
    frozen = list(deck)

    def run(obs_dict):
        search_agent.DECK = frozen
        return agent(obs_dict)
    return run
