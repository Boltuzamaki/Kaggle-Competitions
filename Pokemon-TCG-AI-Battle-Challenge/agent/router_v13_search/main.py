"""Router v12 with conservative, policy-guided mill-objective forward search."""
from __future__ import annotations

import dataclasses
import os
import random

import base_main
from cg.api import search_begin, search_end, search_step, to_observation_class

DECK = list(base_main.read_deck_csv())
DET = int(os.environ.get("R13_DET", "3"))
MARGIN = float(os.environ.get("R13_MARGIN", "1800"))
MAX_OPTIONS = int(os.environ.get("R13_MAX_OPTIONS", "8"))
MAX_ROLL = int(os.environ.get("R13_MAX_ROLL", "36"))


def _dict(obs):
    return dataclasses.asdict(obs)


def _hidden(cur, me):
    mp, op = cur.players[me], cur.players[1-me]
    # Independent determinizations prevent a single lucky draw order dominating.
    return dict(
        your_deck=[random.choice(DECK) for _ in range(mp.deckCount)],
        your_prize=[random.choice(DECK) for _ in range(len(mp.prize))],
        opponent_deck=[1072] * op.deckCount,
        opponent_prize=[1] * len(op.prize),
        opponent_hand=[1] * op.handCount,
        opponent_active=[1072] if (op.active and op.active[0] is None) else [],
    )


def _value(cur, me, root_opp_deck):
    if cur.result >= 0:
        if cur.result == me:
            return 1_000_000.0
        if cur.result == 1-me:
            return -1_000_000.0
        return 0.0
    mp, op = cur.players[me], cur.players[1-me]
    my_board = [p for p in list(mp.active)+list(mp.bench) if p]
    op_board = [p for p in list(op.active)+list(op.bench) if p]
    milled = root_opp_deck - op.deckCount
    v = milled * 2500.0 - op.deckCount * 35.0
    v += (len(op.prize)-len(mp.prize)) * 1300.0
    v += sum(p.hp for p in my_board) * 0.8 - sum(p.hp for p in op_board) * 0.08
    v += sum(1 for p in my_board if p.id == 58) * 500.0
    v += sum(1 for p in my_board if p.id == 345) * 650.0
    # Avoid rollout lines that consume our own deck without milling enough.
    v += mp.deckCount * 18.0
    return v


def _rollout(st, me, root_opp_deck):
    cur = st.observation.current
    steps = 0
    while (cur.result < 0 and st.observation.select is not None
           and cur.yourIndex == me and steps < MAX_ROLL):
        od = _dict(st.observation)
        pick = base_main.agent(od)
        if not pick:
            break
        st = search_step(st.searchId, pick)
        cur = st.observation.current
        steps += 1
    return _value(cur, me, root_opp_deck)


def _search(obs_dict, baseline):
    sel = obs_dict.get("select") or {}
    opts = sel.get("option") or []
    if (int(sel.get("context", -1) or 0) != 0 or
            (sel.get("maxCount", 1) or 1) != 1 or
            not 2 <= len(opts) <= MAX_OPTIONS or baseline[0] >= len(opts)):
        return baseline
    O = to_observation_class(obs_dict)
    me = O.current.yourIndex
    root_opp_deck = O.current.players[1-me].deckCount
    vals = [0.0] * len(opts)
    try:
        for _ in range(DET):
            root = search_begin(O, **_hidden(O.current, me))
            for i in range(len(opts)):
                try:
                    vals[i] += _rollout(search_step(root.searchId, [i]), me, root_opp_deck)
                except Exception:
                    vals[i] -= 1_000_000.0
            search_end()
    except Exception:
        try: search_end()
        except Exception: pass
        return baseline
    vals = [v / DET for v in vals]
    best = max(range(len(vals)), key=vals.__getitem__)
    return [best] if best != baseline[0] and vals[best] - vals[baseline[0]] >= MARGIN else baseline


def agent(obs_dict, configuration=None):
    baseline = base_main.agent(obs_dict, configuration)
    if obs_dict.get("select") is None:
        return baseline
    try:
        return _search(obs_dict, baseline)
    except Exception:
        return baseline
