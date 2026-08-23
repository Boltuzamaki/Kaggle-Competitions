"""Correct damage for THREAT ASSESSMENT ONLY -- offence keeps the static estimate.

The combined scaled-damage fix (scaled_policy.py) was strongly rejected by SPRT:
W10 L53 at 60 seeds. It changed two things at once, and the likely culprit is the
offence half.

Mechanism: Myriad Leaf Shower scales with accumulated Energy. The static
30-damage estimate made attacking look unattractive, so the agent defaulted to
setup (Teal Dance every turn) and only swung when forced -- which is correct play
for this deck. With correct damage it sees "90 now", attacks immediately, and
never reaches the 270-damage turns. Our search is ~1 turn deep, so it cannot see
that waiting compounds; the underestimate was accidentally acting as a patience
heuristic.

That argument does NOT apply to defence. Reading the OPPONENT's damage as zero
(Alakazam's Powerful Hand lists 0) is pure blindness with no compensating benefit
-- we simply never retreat from lethal.

So this variant fixes only `_lethal_threat` and leaves `_plan_attack` untouched.
One delta at a time, per the Orbit Wars RL write-up.
"""
from __future__ import annotations

from domain_policy import (  # noqa: F401
    DomainPolicy, SelectContext, OptionType, AreaType, CardType,
    _CARDS, _ADMG, _ACOST, _weak, _etype, _prize, _get_card,
    _ACTIVE, _BENCH, evaluate_board, domain_agent,
)
import damage_model


class ScaledDefencePolicy(DomainPolicy):
    def _lethal_threat(self):
        act = self.myp.active[0] if self.myp.active else None
        if act is None:
            return False
        mx = 0
        for p in self._op_board():
            if p is None:
                continue
            energy = len(p.energies or []) + 1
            c = _CARDS.get(p.id)
            for aid in (getattr(c, "attacks", None) or []):
                if _ACOST.get(aid, 99) > energy:
                    continue
                d = damage_model.scaled_damage(
                    aid, _ADMG.get(aid, 0), self.st, self.me, attacker_is_me=False)
                try:
                    w = _weak(act.id)
                    if w is not None and w == _etype(p.id):
                        d *= 2
                except Exception:
                    pass
                if d > mx:
                    mx = d
        return mx >= (act.hp or 0)


def rank_options(obs):
    try:
        return ScaledDefencePolicy(obs).choose()
    except Exception:
        n = len(obs.select.option) if obs.select else 0
        return list(range(n))


def scaled_defence_agent(obs_dict, deck):
    from cg.api import to_observation_class
    try:
        if obs_dict.get("select") is None:
            return list(deck)
        obs = to_observation_class(obs_dict)
        n = len(obs.select.option)
        if n == 0:
            return []
        mc = obs.select.maxCount or 1
        r = [i for i in ScaledDefencePolicy(obs).choose() if 0 <= i < n]
        return r[:mc] if r else list(range(min(mc, n)))
    except Exception:
        return domain_agent(obs_dict, deck)
