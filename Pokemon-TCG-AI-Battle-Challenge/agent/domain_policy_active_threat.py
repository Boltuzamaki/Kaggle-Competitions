"""Retreat-policy experiment: confirmed active threat + better switch target.

This preserves the base domain policy except for the two pieces implicated by
EXP-04's follow-up audit:

* only the opponent's active Pokemon is treated as a confirmed next-turn threat;
* SWITCH/TO_ACTIVE choices rank actual attack readiness, HP, prize liability,
  and survival against that active attacker.
"""

from __future__ import annotations

import domain_policy as base

AreaType = base.AreaType
OptionType = base.OptionType
SelectContext = base.SelectContext


class DomainPolicy(base.DomainPolicy):
    def _active_threat_damage(self, target):
        opponent = self.opp.active[0] if self.opp.active else None
        if opponent is None or target is None:
            return 0
        return base._damage(
            opponent.id, len(opponent.energies or []) + 1, target.id
        )

    def _lethal_threat(self):
        active = self.myp.active[0] if self.myp.active else None
        return bool(active and self._active_threat_damage(active) >= active.hp)

    def _score_card(self, option):
        card = base._get_card(
            self.obs,
            option.area,
            option.index,
            getattr(option, "playerIndex", self.me),
        )
        own_choice = getattr(option, "playerIndex", self.me) == self.me
        if (
            card is not None
            and own_choice
            and self.ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE)
        ):
            energy = len(card.energies or [])
            attack = base._best_attack(card.id, energy)
            damage = attack[1] if attack else 0
            score = damage * 12.0 + energy * 250.0 + card.hp * 2.0
            score -= base._prize(card.id) * 120.0
            incoming = self._active_threat_damage(card)
            if incoming >= card.hp:
                score -= 10000.0
            else:
                score -= incoming * 2.0
            return score
        return super()._score_card(option)


def rank_options(obs):
    try:
        return DomainPolicy(obs).choose()
    except Exception:
        n = len(obs.select.option) if obs.select else 0
        return list(range(n))


evaluate_board = base.evaluate_board


def domain_agent(obs_dict, deck):
    try:
        if obs_dict.get("select") is None:
            return list(deck)
        obs = base.to_observation_class(obs_dict)
        n = len(obs.select.option)
        if n == 0:
            return []
        count = obs.select.maxCount or 1
        ranked = [i for i in DomainPolicy(obs).choose() if 0 <= i < n]
        return ranked[:count] if ranked else list(range(min(count, n)))
    except Exception:
        return base.domain_agent(obs_dict, deck)
