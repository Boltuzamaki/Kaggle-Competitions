"""Explicit attack-timing (tempo) rules on top of the Ogerpon policy.

Motivation, from a measured result rather than a guess: correcting our damage
model made the agent MUCH worse (SPRT W27 L93). The static 30-damage estimate had
been suppressing attacks, so the agent built Energy instead -- and Myriad Leaf
Shower scales with accumulated Energy (30 + 30 per Energy on both Actives). The
bug was accidentally acting as a patience rule, and patience was worth a lot.

Our 1-turn search cannot discover that on its own: it never sees that waiting
compounds. So encode the tempo decision explicitly and search over it.

Knobs:
  attack_min_energy    do not attack until the active holds this many Energy
                       (unless the attack knocks the target out)
  attack_min_frac      only attack if damage >= this fraction of target HP
                       (again, KO always allowed)
  setup_turns          pure setup for the first N turns
  ko_always            take a knock-out regardless of the above

Defaults are permissive (0,0,0) = current behaviour, so an untuned config is a
no-op and any gain is attributable to the rules.
"""
from __future__ import annotations

import json
import os

from ogerpon_policy import OgerponPolicy, W as _OW  # noqa: F401
from domain_policy import OptionType, _ADMG, _ACOST, _CARDS, domain_agent
import damage_model

T = {
    "attack_min_energy": 0.0,
    "attack_min_frac": 0.0,
    "setup_turns": 0.0,
    "attack_penalty": 900.0,     # how hard to suppress a premature attack
}

for _p in ("tempo_w.json",
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "tempo_w.json"),
           "/kaggle_simulations/agent/tempo_w.json"):
    try:
        if os.path.exists(_p):
            T.update(json.load(open(_p)))
            break
    except Exception:
        pass


class TempoPolicy(OgerponPolicy):
    def _premature(self, o):
        """True if this attack should be deferred in favour of building up."""
        try:
            aid = getattr(o, "attackId", None)
            if aid is None:
                return False
            act = self.myp.active[0] if self.myp.active else None
            opp_act = self.opp.active[0] if self.opp.active else None
            if act is None or opp_act is None:
                return False

            # real damage, so the KO exemption is actually correct
            dmg = damage_model.scaled_damage(
                aid, _ADMG.get(aid, 0), self.st, self.me, attacker_is_me=True)
            if dmg >= (opp_act.hp or 0):
                return False                      # never defer a knock-out

            if self.st.turn < T["setup_turns"]:
                return True
            if len(act.energies or []) < T["attack_min_energy"]:
                return True
            if T["attack_min_frac"] > 0:
                if dmg < T["attack_min_frac"] * (opp_act.hp or 1):
                    return True
        except Exception:
            pass
        return False

    def _score(self, o):
        s = super()._score(o)
        if o.type == OptionType.ATTACK and self._premature(o):
            s -= T["attack_penalty"]
        return s


def rank_options(obs):
    try:
        return TempoPolicy(obs).choose()
    except Exception:
        n = len(obs.select.option) if obs.select else 0
        return list(range(n))


def tempo_agent(obs_dict, deck):
    from cg.api import to_observation_class
    try:
        if obs_dict.get("select") is None:
            return list(deck)
        obs = to_observation_class(obs_dict)
        n = len(obs.select.option)
        if n == 0:
            return []
        mc = obs.select.maxCount or 1
        r = [i for i in TempoPolicy(obs).choose() if 0 <= i < n]
        return r[:mc] if r else list(range(min(mc, n)))
    except Exception:
        return domain_agent(obs_dict, deck)
