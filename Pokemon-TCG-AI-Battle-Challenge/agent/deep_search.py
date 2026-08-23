"""Multi-turn lookahead built on top of the forked agent's search primitives.

The fork's `_search_decide` is exactly 2 plies deep:

    ply 1  take candidate move, `_greedy_complete_turn(me)`   -> our turn ends
    ply 2  min over opponent's top-K first actions,
           `_greedy_complete_turn(opp)`                       -> opponent turn ends
           `_leaf_eval(state)`                                <- leaf HERE

`_greedy_complete_turn` returns at the turn hand-off, so the horizon is one full
turn each. That is why every "more search" knob came back flat: N_DET and K_OPP
widen the tree without deepening it, and re-sampling a 2-turn horizon more times
does not tell you whether a setup play pays off on turn 3 or 4.

This module keeps the fork's ply-1 and ply-2 structure and then CONTINUES the
line: after the opponent's reply it alternates greedy turns for `EXTRA_TURNS`
more, evaluating the leaf at the deeper horizon. Greedy continuation is the
standard cheap deepening -- it is a rollout, not a full minimax, so cost grows
linearly in depth rather than exponentially.

Rationale for expecting a gain here specifically, where width did not:
this deck's core decisions (invest in Kadabra/Alakazam via Rare Candy, hold Boss's
Orders, spend Poffin now or later) pay off two-to-four turns out. A 2-turn horizon
scores the investment turn as a tempo LOSS, because the payoff lands past the leaf.

Everything here calls the fork's own primitives; no policy logic is reimplemented.
Depth is a runtime knob so the same module serves every arm of the experiment.

Honest caveat: our earlier hand-rolled 2-ply minimax looked strong at n=7 and then
lost at n=54 (W26 L28), and the scaled-damage fix made the agent worse because its
shallow search had come to rely on a compensating bias. Deeper is not automatically
better when the leaf evaluator is imperfect -- a deeper rollout can amplify
`_leaf_eval`'s errors instead of averaging them out. This must be measured.
"""
from __future__ import annotations

import time

EXTRA_TURNS = 2          # additional turn-PAIRS explored past the fork's leaf
DEEP_SUBSTEP_CAP = 40    # per continued turn


def install(m, extra_turns=None, k_opp=None, n_det=None, budget=None, margin=500.0,
            consensus=0.0, margin_late=None, late_prizes=2):
    """Replace `m._search_decide` with a deeper variant. Returns the module.

    `m` is an already-imported fork_main module object, so several independently
    configured copies can coexist in one process for paired testing.
    """
    depth = EXTRA_TURNS if extra_turns is None else extra_turns
    if k_opp is not None:
        m.K_OPP = k_opp
    if n_det is not None:
        m.N_DET = n_det
    if budget is not None:
        m.TIME_BUDGET_S = budget

    SelectContext = m.SelectContext
    OptionType = m.OptionType
    search_begin, search_step, search_end = m.search_begin, m.search_step, m.search_end

    def _continue_line(sid, cur, me_i, deadline, turns):
        """Alternate greedy turns past the fork's leaf. Stops early on terminal."""
        for _ in range(turns):
            cs = cur.current
            if cs is None or (cs.result is not None and cs.result >= 0):
                return cur
            if time.monotonic() > deadline:
                return cur
            owner = cs.yourIndex
            sid, cur = m._advance_forced(sid, cur, owner, deadline, limit=6)
            cs = cur.current
            if cs is None or (cs.result is not None and cs.result >= 0) or cur.select is None:
                return cur
            sid, cur = m._greedy_complete_turn(sid, cur, cs.yourIndex, deadline)
        return cur

    def _deep_decide(obs, base_order, base_scores):
        if not (m.USE_SEARCH and m._search_ok):
            return None
        st, sel = obs.current, obs.select
        if st is None or sel is None or sel.context != SelectContext.MAIN:
            return None
        n = len(sel.option)
        if n < 3 or n > m.SEARCH_MAX_OPTS or st.turn < 2:
            return None
        if getattr(obs, "search_begin_input", None) is None:
            m._search_ok = False
            return None

        me_i = st.yourIndex
        heur_top = base_order[0]
        cand = [heur_top]
        for i in base_order[1:]:
            if sel.option[i].type in (OptionType.ATTACK, OptionType.END):
                continue
            if base_scores[i] < 0:
                continue
            cand.append(i)
            if len(cand) >= 8:
                break
        if len(cand) < 2:
            return None

        t0 = time.monotonic()
        deadline = t0 + m.TIME_BUDGET_S
        acc = {i: 0.0 for i in cand}
        n_eval = {i: 0 for i in cand}
        # per-determinization votes: how many independent samples rank each
        # candidate above the heuristic's top choice. A margin filter only
        # looks at the AVERAGE, so one lucky determinization can carry a bad
        # move; requiring agreement across samples filters that directly.
        votes = {i: 0 for i in cand}
        prev_acc = {i: 0.0 for i in cand}
        n_dets_done = 0
        began = False
        try:
            for _det in range(m.N_DET):
                if time.monotonic() > deadline:
                    break
                hidden = m._sample_hidden(st, me_i)
                try:
                    ss0 = search_begin(obs, **hidden)
                    began = True
                except Exception:
                    m._search_ok = False
                    return None
                root_sid = ss0.searchId

                for idx in cand:
                    if time.monotonic() > deadline:
                        break
                    try:
                        ss = search_step(root_sid, [idx])
                    except Exception:
                        continue
                    sid1, cur = ss.searchId, ss.observation
                    sid1, cur = m._greedy_complete_turn(sid1, cur, me_i, deadline)
                    cs = cur.current
                    if (cs is None or (cs.result is not None and cs.result >= 0)
                            or cs.yourIndex == me_i or cur.select is None):
                        acc[idx] += m._leaf_eval(cs, me_i)
                        n_eval[idx] += 1
                        continue
                    sid1, cur = m._advance_forced(sid1, cur, 1 - me_i, deadline)
                    cs = cur.current
                    if (cs is None or cur.select is None
                            or cur.select.context != SelectContext.MAIN
                            or cs.yourIndex == me_i):
                        acc[idx] += m._leaf_eval(cs, me_i)
                        n_eval[idx] += 1
                        continue

                    _, op_order = m._greedy_pick(cur)
                    worst = None
                    for k in range(min(m.K_OPP, len(op_order))):
                        if time.monotonic() > deadline:
                            break
                        try:
                            ss2 = search_step(sid1, [op_order[k]])
                        except Exception:
                            continue
                        sid2, cur2 = ss2.searchId, ss2.observation
                        sid2, cur2 = m._greedy_complete_turn(sid2, cur2, 1 - me_i, deadline)
                        sid2, cur2 = m._advance_forced(sid2, cur2, me_i, deadline, limit=6)
                        # ---- the extension: keep playing past the fork's leaf ----
                        if depth > 0:
                            cur2 = _continue_line(sid2, cur2, me_i, deadline, depth * 2)
                        v = m._leaf_eval(cur2.current, me_i)
                        worst = v if worst is None else min(worst, v)
                    if worst is None:
                        worst = m._leaf_eval(cs, me_i)
                    acc[idx] += worst
                    n_eval[idx] += 1

                # Tally THIS determinization's own preference. acc is cumulative,
                # so the per-sample value is the delta since the previous round --
                # comparing raw acc would compare running totals and let early
                # rounds dominate the vote.
                delta = {i: acc[i] - prev_acc.get(i, 0.0) for i in cand}
                top_d = delta.get(heur_top)
                if top_d is not None:
                    for i in cand:
                        if i != heur_top and n_eval[i] == n_eval.get(heur_top, 0) \
                           and delta[i] > top_d:
                            votes[i] += 1
                prev_acc = dict(acc)
                n_dets_done += 1
                try:
                    search_end()
                except Exception:
                    pass
                began = False

            n_top = n_eval.get(heur_top, 0)
            if n_top == 0:
                return None
            evaluated = [i for i in cand if n_eval[i] == n_top]
            avg = {i: acc[i] / n_eval[i] + 1e-6 * base_scores[i] for i in evaluated}
            best = max(evaluated, key=lambda i: avg[i])
            if best == heur_top:
                return None
            # The stock threshold is 500.0 -- half a prize. Measured on a real
            # game it fires once in 43 decisions, so the shipped agent is ~98%
            # pure heuristic and almost nothing inside the search can influence
            # play. This is the knob that decides how much the search is used
            # at all, which is why depth and evaluator changes both read flat.
            # Phase-dependent threshold. The search looks exactly two turns
            # ahead, which is a poor model of a long game but a good one near the
            # end -- with few prizes left, two turns can actually reach the
            # finish. So its overrides deserve more trust late than early.
            eff_margin = margin
            if margin_late is not None:
                try:
                    opp_prizes = len(st.players[1 - me_i].prize or [])
                    my_prizes = len(st.players[me_i].prize or [])
                    if min(opp_prizes, my_prizes) <= late_prizes:
                        eff_margin = margin_late
                except Exception:
                    pass
            if avg[best] < avg[heur_top] + eff_margin:
                return None
            if consensus > 0.0 and n_dets_done > 0:
                if votes.get(best, 0) < consensus * n_dets_done:
                    return None
            return best
        except Exception:
            return None
        finally:
            if began:
                try:
                    search_end()
                except Exception:
                    pass

    m._search_decide = _deep_decide
    return m
