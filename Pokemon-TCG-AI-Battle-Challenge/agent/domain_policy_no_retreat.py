"""Ablation: keep the legal-KO override but disable threat-driven retreat."""

from __future__ import annotations

import domain_policy_active_threat as active

AreaType = active.AreaType
OptionType = active.OptionType
SelectContext = active.SelectContext
evaluate_board = active.evaluate_board


class DomainPolicy(active.DomainPolicy):
    def _lethal_threat(self):
        return False


def rank_options(obs):
    try:
        return DomainPolicy(obs).choose()
    except Exception:
        n = len(obs.select.option) if obs.select else 0
        return list(range(n))


def domain_agent(obs_dict, deck):
    try:
        if obs_dict.get("select") is None:
            return list(deck)
        obs = active.base.to_observation_class(obs_dict)
        n = len(obs.select.option)
        if n == 0:
            return []
        count = obs.select.maxCount or 1
        ranked = [i for i in DomainPolicy(obs).choose() if 0 <= i < n]
        return ranked[:count] if ranked else list(range(min(count, n)))
    except Exception:
        return active.base.domain_agent(obs_dict, deck)
