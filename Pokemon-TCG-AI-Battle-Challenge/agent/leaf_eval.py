"""Alternative leaf evaluators for the forked agent's search.

Tonight's result: deepening the search from 2 turns to 4 changed 51 of 600 paired
games and won exactly as many as it lost. The horizon is not the constraint --
the evaluator at the leaf is. The fork's is:

    1000*(prize_diff) + (my_hp - op_hp) + 5*(energy_diff) - 4000*(no_active)

Three things that decide Pokemon TCG positions are missing entirely:

  card advantage  hand size and deck count do not appear at all, yet this deck's
                  whole engine (Poffin, Poke Pad, Hilda, Dawn) exists to generate
                  cards, and Alakazam's Powerful Hand scales damage with HAND SIZE.
  evolution       Abra, Kadabra and Alakazam differ only through their raw HP
                  number, so the search cannot see that a Rare Candy line is
                  nearly complete -- exactly the multi-turn investment that a
                  2-turn horizon already struggles with.
  board width     bench development is invisible, so trading away a benched
                  attacker looks free.

Each term is a separate installable variant so a battery can attribute any effect
to one change rather than to a bundle -- the mistake that made the wholesale deck
swap uninterpretable.

Raw HP is also questionable as a term: it sums absolute HP, so a healthy 60 HP
Dunsparce and a badly damaged 210 HP Alakazam can score alike. `frachp` replaces
it with fraction-of-max, which is what "how healthy is my board" actually means.
"""
from __future__ import annotations

ABRA, KADABRA, ALAKAZAM = 741, 742, 743
STAGE_VALUE = {ABRA: 0.0, KADABRA: 120.0, ALAKAZAM: 400.0}


def _fields(state, me_i):
    me = state.players[me_i]
    op = state.players[1 - me_i]
    mf = [p for p in (me.active + me.bench) if p]
    of = [p for p in (op.active + op.bench) if p]
    return me, op, mf, of


def _terminal(state, me_i):
    if state.result is not None and state.result >= 0:
        if state.result == me_i:
            return 1e7
        if state.result == 2:
            return 0.0
        return -1e7
    return None


def _count(pl, attr):
    v = getattr(pl, attr, None)
    if isinstance(v, int):
        return v
    try:
        return len(v or [])
    except Exception:
        return 0


def make(variant, w_hand=60.0, w_stage=1.0, w_bench=80.0, w_deck=2.0):
    """Return a `_leaf_eval(state, me_i)` implementing one variant."""

    def ev(state, me_i):
        if state is None:
            return 0.0
        t = _terminal(state, me_i)
        if t is not None:
            return t
        me, op, mf, of = _fields(state, me_i)
        my_en = sum(len(p.energies) for p in mf)
        op_en = sum(len(p.energies) for p in of)
        no_active = 0 if (me.active and me.active[0]) else 1

        base = (1000.0 * (len(op.prize) - len(me.prize))
                + 5.0 * (my_en - op_en)
                - 4000.0 * no_active)

        if variant == "frachp":
            # health as a fraction of the board's own maximum, not raw totals
            def frac(f):
                return sum(200.0 * (p.hp or 0) / max(p.maxHp or 1, 1) for p in f)
            base += frac(mf) - frac(of)
        else:
            base += sum(p.hp for p in mf) - sum(p.hp for p in of)

        if variant in ("hand", "all"):
            base += w_hand * (_count(me, "handCount") - _count(op, "handCount"))
        if variant in ("deck", "all"):
            # running out of deck loses the game; value having cards left
            base += w_deck * (_count(me, "deckCount") - _count(op, "deckCount"))
        if variant in ("stage", "all"):
            mine = sum(STAGE_VALUE.get(p.id, 0.0) for p in mf)
            theirs = sum(STAGE_VALUE.get(p.id, 0.0) for p in of)
            base += w_stage * (mine - theirs)
        if variant in ("bench", "all"):
            base += w_bench * (len(mf) - len(of))
        return base

    return ev


def install(m, variant, **kw):
    """Swap the evaluator on an already-imported fork_main module object."""
    if variant != "stock":
        m._leaf_eval = make(variant, **kw)
    return m
