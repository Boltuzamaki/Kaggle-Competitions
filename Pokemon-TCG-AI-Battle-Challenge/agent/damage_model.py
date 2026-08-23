"""Correct damage for conditional / scaling attacks.

domain_policy takes damage straight from the static `Attack.damage` field. For
17 attacks across the current meta decks that field is wrong, and for seven of
them it is literally 0:

    Teal Mask Ogerpon ex  Myriad Leaf Shower   listed  30  -> 30 + 30*(energy on BOTH actives)
    Alakazam              Powerful Hand        listed   0  -> 20 * (cards in your hand)
    Dipplin               Do the Wave          listed   0  -> 20 * (your benched Pokemon)
    Mega Froslass ex      Resentful Refrain    listed   0  -> 50 * (cards in opponent hand)
    Raging Bolt ex        Bellowing Thunder    listed   0  -> 70 * (energy discarded)
    Passimian             Coordinated Throwing listed   0  -> 20 * (your basics in play)
    Rabsca                Psychic              listed  10  -> 10 + 30*(energy on opp active)

The consequences were systematic, in both directions:

  * OFFENCE -- our own Ogerpon reads as 30 damage instead of ~270, so the search's
    knock-out detection never fires, the "take the lethal attack" hard override
    never triggers, and Boss's Orders never gusts for a kill.
  * DEFENCE -- `_lethal_threat()` runs the same function over the OPPONENT's
    attacks, so against Alakazam (the LB-950 agent's deck) incoming damage
    computes as ZERO and we never retreat from lethal.

No amount of weight tuning can fix this: the search was optimising against a
wrong damage function, which is the most likely reason SPSA and card-specific
weights both came back flat.

This module computes damage from the actual board state. Anything it does not
recognise falls back to the static field, so it can only add information.
"""
from __future__ import annotations

# attackId -> (base, per_unit, counter)
# counter names are resolved against the board in `scaled_damage`.
SCALING = {
    120: (30, 30, "energy_both_actives"),   # Myriad Leaf Shower (Teal Mask Ogerpon ex)
}

# Resolved lazily by name so we survive attackId changes between engine builds.
# (name, required text fragment, base, per_unit, counter)
# The text fragment is mandatory: several distinct cards share an attack name
# ("Psychic" appears on 5 cards with listed damage 10/30/40/80) and only the one
# carrying the scaling clause may be rewritten.
_BY_TEXT = [
    ("myriad leaf shower", "for each energy attached to both", 30, 30, "energy_both_actives"),
    ("powerful hand", "for each card in your hand", 0, 20, "my_hand"),
    ("do the wave", "for each of your benched", 0, 20, "my_bench"),
    ("resentful refrain", "for each card in your opponent", 0, 50, "opp_hand"),
    ("coordinated throwing", "for each of your basic", 0, 20, "my_basics"),
    ("full moon rondo", "for each benched", 20, 20, "both_bench"),
    ("raging curse", "for each damage counter", 0, 10, "my_bench_damage"),
    ("psychic", "for each energy attached to your opponent", 10, 30, "energy_opp_active"),
]

_TABLE = None


def _table():
    """attackId -> (base, per_unit, counter), built from attack text."""
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    _TABLE = dict(SCALING)
    try:
        from cg.api import all_attack
        for a in all_attack():
            nm = (a.name or "").strip().lower()
            tx = (a.text or "").strip().lower()
            for key, frag, base, per, counter in _BY_TEXT:
                if nm == key and frag in tx:
                    _TABLE[a.attackId] = (base, per, counter)
    except Exception:
        pass
    return _TABLE


def _count(counter, state, me, attacker_is_me):
    """Evaluate a scaling counter against the board."""
    try:
        atk = me if attacker_is_me else 1 - me
        dfn = 1 - atk
        ap = state.players[atk]
        dp = state.players[dfn]
        a_act = ap.active[0] if ap.active else None
        d_act = dp.active[0] if dp.active else None

        if counter == "energy_both_actives":
            n = len(a_act.energies or []) if a_act else 0
            n += len(d_act.energies or []) if d_act else 0
            return n
        if counter == "energy_opp_active":
            return len(d_act.energies or []) if d_act else 0
        if counter == "my_hand":
            return ap.handCount or 0
        if counter == "opp_hand":
            return dp.handCount or 0
        if counter == "my_bench":
            return len([m for m in (ap.bench or []) if m])
        if counter == "both_bench":
            return (len([m for m in (ap.bench or []) if m])
                    + len([m for m in (dp.bench or []) if m]))
        if counter == "my_basics":
            return len([m for m in list(ap.active or []) + list(ap.bench or []) if m])
        if counter == "my_bench_damage":
            tot = 0
            for m in (ap.bench or []):
                if m:
                    tot += max(0, ((m.maxHp or 0) - (m.hp or 0)) // 10)
            return tot
    except Exception:
        pass
    return 0


def scaled_damage(attack_id, static_damage, state, me, attacker_is_me):
    """Real damage for `attack_id` on this board, or static_damage if unknown."""
    t = _table().get(attack_id)
    if t is None or state is None:
        return static_damage
    base, per, counter = t
    return base + per * _count(counter, state, me, attacker_is_me)


def is_scaling(attack_id):
    return attack_id in _table()
