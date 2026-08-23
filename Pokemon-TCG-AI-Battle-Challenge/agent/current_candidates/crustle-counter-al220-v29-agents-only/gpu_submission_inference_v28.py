"""Small CPU inference layer distilled from the local GPU preference model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random


def _area(value):
    return {
        0: "deck", 1: "hand", 2: "discard", 3: "active", 4: "bench",
        5: "prize", 6: "stadium", 12: "looking",
    }.get(int(value), str(value)) if str(value).lstrip("-").isdigit() else str(value)


def _bucket(value, cuts):
    try:
        value = float(value or 0)
    except (TypeError, ValueError):
        value = 0
    return sum(value >= cut for cut in cuts)


def _zone(observation, area, player):
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    name = _area(area)
    if name == "deck":
        return select.get("deck") or []
    if name == "stadium":
        return current.get("stadium") or []
    if name == "looking":
        return current.get("looking") or []
    players = current.get("players") or []
    return players[player].get(name) or [] if 0 <= player < len(players) else []


def _card_id(observation, option):
    current = observation.get("current") or {}
    own = int(current.get("yourIndex") or 0)
    card = None
    try:
        if int(option.get("type", -1)) in (7, 8, 9):
            card = _zone(observation, "hand", own)[int(option.get("index"))]
        elif option.get("area") is not None:
            card = _zone(
                observation,
                option.get("area"),
                int(option.get("playerIndex", own)),
            )[int(option.get("index"))]
    except (IndexError, TypeError, ValueError):
        card = None
    return int(card["id"]) if isinstance(card, dict) and card.get("id") is not None else None


def _card_signature(card):
    if not isinstance(card, dict):
        return 0, 0, 0
    energies = card.get("energy") or card.get("energies") or []
    return (
        int(card.get("id") or 0),
        int(card.get("damage") or card.get("damageCounter") or 0),
        len(energies) if isinstance(energies, list) else 0,
    )


def _tokens(observation, option):
    select = observation.get("select") or {}
    current = observation.get("current") or {}
    players = current.get("players") or []
    own = int(current.get("yourIndex") or 0)
    opponent = 1 - own
    context = int(select.get("context") or 0)
    kind = int(option.get("type") or 0)
    base = f"c{context}|t{kind}"
    own_player = players[own] if own < len(players) else {}
    opp_player = players[opponent] if opponent < len(players) else {}
    own_bench = len(own_player.get("bench") or [])
    opp_bench = len(opp_player.get("bench") or [])
    relation = (
        "own" if option.get("playerIndex") == own
        else "opp" if option.get("playerIndex") == opponent else "none"
    )
    tokens = [
        f"context:{context}", base, f"{base}|area:{_area(option.get('area', 'none'))}",
        f"{base}|relation:{relation}",
        f"{base}|indexBucket:{_bucket(option.get('index'), (1,3,6,12,24))}",
        f"{base}|turn:{_bucket(current.get('turn'), (2,5,9,15))}",
        f"{base}|benches:{min(5,own_bench)}:{min(5,opp_bench)}",
    ]
    card_id = _card_id(observation, option)
    attack = option.get("attackId")
    effect = (select.get("effect") or {}).get("id")
    if card_id is not None:
        tokens += [f"{base}|card:{card_id}", f"context:{context}|card:{card_id}"]
    if attack is not None:
        tokens.append(f"{base}|attack:{int(attack)}")
    if effect is not None:
        tokens.append(f"{base}|effect:{int(effect)}")
        if card_id is not None:
            tokens.append(f"effect:{int(effect)}|card:{card_id}")
    if option.get("number") is not None:
        tokens.append(f"{base}|number:{int(option['number'])}")
    if option.get("energyIndex") is not None:
        tokens.append(f"{base}|energyIndex:{int(option['energyIndex'])}")
    own_active = (own_player.get("active") or [None])[0]
    opp_active = (opp_player.get("active") or [None])[0]
    own_id, own_damage, own_energy = _card_signature(own_active)
    opp_id, opp_damage, opp_energy = _card_signature(opp_active)
    state = (
        f"turn{_bucket(current.get('turn'),(2,4,7,11,16))}"
        f"|hand{_bucket(own_player.get('handCount'),(2,5,8,12))}"
        f"|bench{min(5,own_bench)}|obench{min(5,opp_bench)}"
        f"|prize{min(6,len(own_player.get('prize') or []))}"
    )
    pressure = (
        f"od{_bucket(opp_damage,(10,50,100,160,220))}"
        f"|oe{_bucket(opp_energy,(1,2,3,4))}"
        f"|yd{_bucket(own_damage,(10,50,100,160,220))}"
        f"|ye{_bucket(own_energy,(1,2,3,4))}"
    )
    tokens += [
        f"{base}|{state}", f"{base}|{pressure}",
        f"{base}|ownActive:{own_id}", f"{base}|oppActive:{opp_id}",
    ]
    if attack is not None:
        tokens += [
            f"attack:{int(attack)}|{state}",
            f"attack:{int(attack)}|oppActive:{opp_id}",
            f"attack:{int(attack)}|{pressure}",
        ]
    if option.get("number") is not None:
        tokens.append(f"{base}|number:{int(option['number'])}|{state}")
    return tokens


def _features(observation, option, dimension):
    values = {}
    for token in _tokens(observation, option):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") & (dimension - 1)
        sign = 1.0 if digest[4] & 1 else -1.0
        values[index] = values.get(index, 0.0) + sign
    return values


class DistilledSelector:
    def __init__(self, threshold, model_file="selector_weights_v28.json"):
        root = os.path.dirname(os.path.abspath(__file__))
        payload = json.load(open(os.path.join(root, model_file), encoding="utf-8"))
        self.dimension = int(payload["dimension"])
        self.weights = {int(k): float(v) for k, v in payload["weights"].items()}
        self.threshold = float(threshold)

    def score(self, observation, option):
        return sum(
            self.weights.get(index, 0.0) * value
            for index, value in _features(observation, option, self.dimension).items()
        )

    def choose(self, observation, baseline):
        select = observation.get("select") or {}
        options = select.get("option") or []
        if (
            len(options) < 2
            or int(select.get("minCount") or 0) != 1
            or int(select.get("maxCount") or 0) != 1
            or not isinstance(baseline, list)
            or len(baseline) != 1
            or not isinstance(baseline[0], int)
            or not 0 <= baseline[0] < len(options)
        ):
            return baseline
        scores = [self.score(observation, option) for option in options]
        peak = max(scores)
        tied = [index for index, score in enumerate(scores) if math.isclose(score, peak)]
        best = random.choice(tied)
        if best != baseline[0] and scores[best] - scores[baseline[0]] >= self.threshold:
            return [best]
        return baseline
