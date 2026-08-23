"""2-ply minimax: score our move by the opponent's BEST reply, not by our own board.

Our hybrid search stops after completing our own turn and evaluates the resulting
board. It never models what the opponent does next. That myopia is the single
constraint eight experiments failed to work around, and it was exposed most
clearly when correcting our damage model made the agent MUCH worse (SPRT W27 L93):
with 1-ply search the agent cannot see that attacking early hands the opponent a
strong reply, so an accurate "90 damage now" beats a patient 270-damage line.

Structure (ply 1: our action + greedy turn completion; ply 2: minimise over the
opponent's top-K first actions) mirrors the search layer the strongest public
agent uses. This is our own implementation against the engine's search API -- no
public policy code is used.

Cost is ~K_OPP times the 1-ply search, so K_OPP is deliberately small and the
whole thing degrades to the 1-ply value if anything fails.
"""
from __future__ import annotations

import os
import random

import domain_policy
import hybrid_agent
from hybrid_agent import _determinize, _hard_override, _TOP_K, _DET, _MAXROLL

try:
    from cg.api import to_observation_class, search_begin, search_step, search_end
    _OK = True
except Exception:
    _OK = False

K_OPP = int(os.environ.get("MM_K_OPP", "3"))     # opponent branching at ply 2
OPP_ROLL = int(os.environ.get("MM_OPP_ROLL", "14"))  # steps to finish their turn


def _complete_turn(st, actor, policy_module, cap):
    """Greedily play out `actor`'s turn. Returns the final search state."""
    cur = st.observation.current
    g = 0
    while (cur.result < 0 and st.observation.select is not None
           and cur.yourIndex == actor and g < cap):
        sel = st.observation.select
        pick = policy_module.rank_options(st.observation)
        mc = sel.maxCount or 1
        pick = [p for p in pick if 0 <= p < len(sel.option)][:mc] or [0]
        st = search_step(st.searchId, pick)
        cur = st.observation.current
        g += 1
    return st


def _search_choice(obs, deck, policy_module):
    O = to_observation_class(obs)
    me = O.current.yourIndex
    n = len(O.select.option)
    if n <= 1:
        return 0

    override = _hard_override(O, policy_module)
    if override is not None:
        return override

    ranked = policy_module.rank_options(O)
    cands = [i for i in ranked if 0 <= i < n][:_TOP_K] or list(range(min(n, _TOP_K)))
    if len(cands) == 1:
        return cands[0]

    vals = {i: 0.0 for i in cands}
    for _ in range(_DET):
        ss = search_begin(O, **_determinize(O.current, me, deck))
        for i in cands:
            try:
                st = search_step(ss.searchId, [i])
                st = _complete_turn(st, me, policy_module, _MAXROLL)
                cur = st.observation.current

                # ply 2: the opponent replies. Score our move by their BEST
                # answer (worst case for us), not by our own board.
                if (cur.result < 0 and st.observation.select is not None
                        and cur.yourIndex != me):
                    opp_ranked = policy_module.rank_options(st.observation)
                    osel = st.observation.select
                    opts = [p for p in opp_ranked
                            if 0 <= p < len(osel.option)][:K_OPP] or [0]
                    worst = None
                    for oi in opts:
                        try:
                            st2 = search_step(st.searchId, [oi])
                            st2 = _complete_turn(st2, 1 - me, policy_module, OPP_ROLL)
                            v = policy_module.evaluate_board(st2.observation.current, me)
                            worst = v if worst is None else min(worst, v)
                        except Exception:
                            continue
                    vals[i] += worst if worst is not None \
                        else policy_module.evaluate_board(cur, me)
                else:
                    vals[i] += policy_module.evaluate_board(cur, me)
            except Exception:
                vals[i] += -1e12
        search_end()
    return max(cands, key=lambda i: vals[i])


def minimax_agent(obs_dict: dict, deck: list, policy_module=domain_policy) -> list:
    """Never raises: degrades to the 1-ply hybrid, then to the flat policy."""
    deck = list(deck)
    try:
        sel = obs_dict.get("select")
        if sel is None:
            return deck
        opts = sel.get("option") or []
        n = len(opts)
        if n == 0:
            return []
        mc = sel.get("maxCount", 1) or 1
        if _OK and mc == 1 and n > 1:
            try:
                return [_search_choice(obs_dict, deck, policy_module)]
            except Exception:
                try:
                    search_end()
                except Exception:
                    pass
        return hybrid_agent.hybrid_agent(obs_dict, deck, policy_module)
    except Exception:
        try:
            return hybrid_agent.hybrid_agent(obs_dict, deck, policy_module)
        except Exception:
            return domain_policy.domain_agent(obs_dict, deck)


def make_agent(deck, policy_module=domain_policy):
    frozen = list(deck)

    def run(obs):
        return minimax_agent(obs, frozen, policy_module)
    return run
