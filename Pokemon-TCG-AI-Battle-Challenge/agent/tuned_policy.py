"""domain_policy with its action-priority constants exposed for tuning.

These numbers were hand-written and never tuned. The Approvers chess write-up
tuned their engine's constants with SPSA over ~20M games; SPSA is designed for
exactly this setting -- a very noisy objective where each evaluation is expensive
and the gradient is unavailable. It perturbs ALL parameters simultaneously and
needs only two evaluations per iteration regardless of dimension.

Defaults reproduce domain_policy exactly, so an untuned W is a no-op. Overrides
load from policy_w.json next to this module (or the Kaggle agent dir).
"""
from __future__ import annotations

import json
import os

from domain_policy import (  # noqa: F401
    DomainPolicy, SelectContext, OptionType, AreaType, CardType,
    _CARDS, _get_card, _prize, _ACTIVE, _BENCH, evaluate_board, domain_agent,
)

W = {
    "ability": 30000.0,
    "play_pokemon": 20000.0,
    "play_trainer": 6000.0,
    "evolve": 9000.0,
    "attach_base": 8000.0,
    "attach_planned": 400.0,
    "attach_active": 50.0,
    "retreat_threat": 3000.0,
    "retreat_plan": 2500.0,
    "attack_planned": 1200.0,
    "attack_other": 1000.0,
    "target_prize": 100.0,
    "target_active": 200.0,
    "target_damaged": 300.0,
    "yes_generic": 5.0,
    "number_scale": 1.0,
}

for _p in ("policy_w.json",
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_w.json"),
           "/kaggle_simulations/agent/policy_w.json"):
    try:
        if os.path.exists(_p):
            W.update(json.load(open(_p)))
            break
    except Exception:
        pass


class TunedPolicy(DomainPolicy):
    def _score(self, o):
        t = o.type
        if t == OptionType.NUMBER:
            return float(getattr(o, "number", 0)) * W["number_scale"]
        if t == OptionType.YES:
            return 100.0 if self.ctx == getattr(SelectContext, "IS_FIRST", -99) \
                else W["yes_generic"]
        if t == OptionType.NO:
            return 0.0
        if t == OptionType.ABILITY:
            return W["ability"]
        if t == OptionType.PLAY:
            card = _get_card(self.obs, AreaType.HAND, o.index, self.me)
            c = _CARDS.get(card.id) if card is not None else None
            if c is not None and getattr(c, "cardType", None) == CardType.POKEMON:
                return W["play_pokemon"]
            return W["play_trainer"]
        if t == OptionType.EVOLVE:
            p = _get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.me)
            return W["evolve"] + (len(p.energies or []) if p else 0)
        if t == OptionType.ATTACH:
            base = W["attach_base"]
            try:
                area, idx = o.inPlayArea, o.inPlayIndex
                bi = idx if area == _ACTIVE else idx + len(self.myp.active)
                if bi == self._planned_attacker_index():
                    base += W["attach_planned"]
                if area == _ACTIVE:
                    base += W["attach_active"]
            except Exception:
                pass
            return base
        if t == OptionType.RETREAT:
            if self._lethal_threat() and len(self.myp.bench) > 0:
                return W["retreat_threat"]
            if self.plan and self.plan[0] != 0:
                return W["retreat_plan"]
            return -5.0
        if t == OptionType.ATTACK:
            if self.plan and getattr(o, "attackId", None) == self.plan[1]:
                return W["attack_planned"]
            return W["attack_other"]
        if t == OptionType.END:
            return -100000.0
        return super()._score(o)

    def _score_card(self, o):
        try:
            pidx = getattr(o, "playerIndex", self.me)
            if pidx != self.me and o.area in (_ACTIVE, _BENCH):
                card = _get_card(self.obs, o.area, o.index, pidx)
                if card is not None:
                    s = _prize(card.id) * W["target_prize"]
                    s += W["target_active"] if o.area == _ACTIVE else 0.0
                    hp = getattr(card, "hp", 1) or 1
                    mx = getattr(_CARDS.get(card.id), "hp", hp) or hp
                    s += W["target_damaged"] * (1.0 - hp / max(mx, 1))
                    return s
        except Exception:
            pass
        return super()._score_card(o)


def rank_options(obs):
    try:
        return TunedPolicy(obs).choose()
    except Exception:
        n = len(obs.select.option) if obs.select else 0
        return list(range(n))


def tuned_agent(obs_dict, deck):
    from cg.api import to_observation_class
    try:
        if obs_dict.get("select") is None:
            return list(deck)
        obs = to_observation_class(obs_dict)
        n = len(obs.select.option)
        if n == 0:
            return []
        mc = obs.select.maxCount or 1
        r = [i for i in TunedPolicy(obs).choose() if 0 <= i < n]
        return r[:mc] if r else list(range(min(mc, n)))
    except Exception:
        return domain_agent(obs_dict, deck)
