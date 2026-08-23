"""domain_policy with correct damage for scaling attacks.

The base policy reads damage from the static `Attack.damage` field, which is
wrong for 14 attacks in the current meta and literally 0 for several. That broke
the search in both directions:

  * our own Teal Mask Ogerpon ex read as 30 damage instead of ~270, so knock-out
    detection, the lethal-attack override and gust-for-KO never fired;
  * `_lethal_threat()` computed ZERO incoming damage from Alakazam's Powerful
    Hand, so we never retreated from lethal against the LB-950 agent's deck.

Only `_plan_attack` and `_lethal_threat` change -- both now evaluate damage
against the live board via damage_model. Everything else is inherited, so this is
a clean single-variable comparison against the baseline policy.
"""
from __future__ import annotations

from domain_policy import (  # noqa: F401
    DomainPolicy, SelectContext, OptionType, AreaType, CardType,
    _CARDS, _ADMG, _ACOST, _best_attack, _weak, _etype, _prize,
    _get_card, _ACTIVE, _BENCH, evaluate_board, domain_agent,
)
import damage_model


class ScaledPolicy(DomainPolicy):
    """DomainPolicy that knows what its attacks actually do."""

    def _real_damage(self, atk_mon, attack_id, def_mon, attacker_is_me=True):
        """Board-aware damage, weakness applied after scaling."""
        static = _ADMG.get(attack_id, 0)
        dmg = damage_model.scaled_damage(
            attack_id, static, self.st, self.me, attacker_is_me)
        try:
            w = _weak(def_mon.id)
            if w is not None and w == _etype(atk_mon.id):
                dmg *= 2
        except Exception:
            pass
        return dmg

    def _affordable_attacks(self, mon, energy):
        c = _CARDS.get(mon.id)
        out = []
        for aid in (getattr(c, "attacks", None) or []):
            if _ACOST.get(aid, 99) <= energy:
                out.append(aid)
        return out

    # ---- offence: pick the attack by REAL damage ----
    def _plan_attack(self):
        best_score, best = -1e18, None
        my = self._my_board()
        opp = self._op_board()
        for ai, atk in enumerate(my):
            if atk is None:
                continue
            energy = len(atk.energies or [])
            eff = energy + (0 if self.st.energyAttached else 1)
            for aid in self._affordable_attacks(atk, eff):
                for ti, tgt in enumerate(opp):
                    if tgt is None or ti != 0:
                        continue
                    dmg = self._real_damage(atk, aid, tgt, attacker_is_me=True)
                    ko = dmg >= (tgt.hp or 0)
                    score = 0.0
                    if ko:
                        score += _prize(tgt.id) * 100000
                        if len(self.opp.prize) <= _prize(tgt.id):
                            score += 500000
                    else:
                        score += dmg * (100.0 / max(tgt.hp or 1, 1))
                    score += 200 if ai == 0 else 0
                    score -= _ACOST.get(aid, 0) * 5
                    if score > best_score:
                        best_score, best = score, (ai, aid, ti, ko)
        self.plan = best

    # ---- defence: see incoming scaling damage ----
    def _lethal_threat(self):
        act = self.myp.active[0] if self.myp.active else None
        if act is None:
            return False
        mx = 0
        for p in self._op_board():
            if p is None:
                continue
            energy = len(p.energies or []) + 1     # assume one attach next turn
            for aid in self._affordable_attacks(p, energy):
                d = self._real_damage(p, aid, act, attacker_is_me=False)
                if d > mx:
                    mx = d
        return mx >= (act.hp or 0)


def rank_options(obs):
    try:
        return ScaledPolicy(obs).choose()
    except Exception:
        n = len(obs.select.option) if obs.select else 0
        return list(range(n))


def scaled_agent(obs_dict, deck):
    from cg.api import to_observation_class
    try:
        if obs_dict.get("select") is None:
            return list(deck)
        obs = to_observation_class(obs_dict)
        n = len(obs.select.option)
        if n == 0:
            return []
        mc = obs.select.maxCount or 1
        r = [i for i in ScaledPolicy(obs).choose() if 0 <= i < n]
        return r[:mc] if r else list(range(min(mc, n)))
    except Exception:
        return domain_agent(obs_dict, deck)
