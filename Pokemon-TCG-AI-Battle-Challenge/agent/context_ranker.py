"""Inference wrapper for the original hashed contextual outcome ranker."""

from __future__ import annotations

from collections import Counter
import hashlib
import json

import archaludon_policy


def _hash(feature, dimension):
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dimension


def _bucket(value, size=1):
    try:
        return int(float(value or 0)) // size
    except Exception:
        return 0


def _name(value, enum_type):
    name = getattr(value, "name", None)
    if name:
        return name
    for candidate in dir(enum_type):
        if not candidate.startswith("_") and getattr(enum_type, candidate, None) == value:
            return candidate
    return str(value)


def _option_label(policy, option):
    option_type = _name(option.type, archaludon_policy.OptionType)
    if option.type == archaludon_policy.OptionType.PLAY:
        card = archaludon_policy._card(
            policy.obs, archaludon_policy.AreaType.HAND, option.index, policy.me
        )
        return f"PLAY:{getattr(card, 'id', None)}", option_type
    if option.type == archaludon_policy.OptionType.ATTACK:
        return f"ATTACK:{getattr(option, 'attackId', None)}", option_type
    if option.type in (
        archaludon_policy.OptionType.ATTACH,
        archaludon_policy.OptionType.EVOLVE,
    ):
        target = archaludon_policy._card(
            policy.obs, option.inPlayArea, option.inPlayIndex, policy.me
        )
        return f"{option_type}:{getattr(target, 'id', None)}", option_type
    return option_type, option_type


class ContextModel:
    def __init__(self, path):
        payload = json.loads(open(path, encoding="utf-8").read())
        self.dimension = payload["dimension"]
        self.weights = payload["weights"]

    def logit(self, policy, option):
        active = policy.active()
        opponent_active = policy.opponent_active()
        action, option_type = _option_label(policy, option)
        features = [
            "bias",
            "context=MAIN",
            f"active={getattr(active, 'id', None)}",
            f"opp_active={getattr(opponent_active, 'id', None)}",
            f"action={action}",
            f"type={option_type}",
            f"turn={_bucket(policy.state.turn, 2)}",
            f"active_hp={_bucket(getattr(active, 'hp', 0), 50)}",
            f"active_energy={_bucket(archaludon_policy._energy_count(active))}",
            f"bench={_bucket(len(policy.mine.bench))}",
            f"opp_hp={_bucket(getattr(opponent_active, 'hp', 0), 50)}",
            f"opp_bench={_bucket(len(policy.theirs.bench))}",
            f"prizes={_bucket(len(policy.mine.prize))}",
            f"opp_prizes={_bucket(len(policy.theirs.prize))}",
            f"hand={_bucket(policy.mine.handCount, 2)}",
            f"legal={_bucket(len(policy.select.option), 3)}",
            f"action_active={action}|{getattr(active, 'id', None)}",
            f"action_opp={action}|{getattr(opponent_active, 'id', None)}",
        ]
        encoded = Counter(_hash(feature, self.dimension) for feature in features)
        return sum(self.weights[index] * value for index, value in encoded.items())


class ContextBlendPolicy(archaludon_policy.SequencedArchaludonPolicy):
    def __init__(self, obs, model, alpha):
        super().__init__(obs)
        self.context_model = model
        self.alpha = alpha

    def choose(self):
        if self.context != archaludon_policy.SelectContext.MAIN:
            return super().choose()
        override = archaludon_policy.hard_override(self.obs)
        if override is not None:
            return [override]
        options = self.select.option
        base_scores = [self._main_score(option) for option in options]
        base_order = sorted(range(len(options)), key=lambda i: base_scores[i], reverse=True)
        base_rank = {index: rank for rank, index in enumerate(base_order)}
        combined = [
            -base_rank[index] + self.alpha * self.context_model.logit(self, option)
            for index, option in enumerate(options)
        ]
        return [max(range(len(options)), key=lambda index: combined[index])]


def blended_agent(obs_dict, deck, model, alpha):
    return archaludon_policy._agent_with_policy(
        obs_dict,
        deck,
        lambda obs: ContextBlendPolicy(obs, model, alpha),
    )
