"""Fast Bayesian card counting and risk calibration for CABT policies."""

from __future__ import annotations

import math
from collections import Counter


def _text(card) -> str:
    skills = getattr(card, "skills", None) or []
    body = " ".join(
        f"{getattr(skill, 'name', '')} {getattr(skill, 'text', '')}"
        for skill in skills
    )
    return f"{getattr(card, 'name', '')} {body}".lower()


def classify_cards(card_data) -> dict[str, set[int]]:
    """Derive tactical classes from the engine's current card database."""
    groups = {
        name: set()
        for name in (
            "pokemon", "basic_pokemon", "evolution", "item", "tool",
            "supporter", "stadium", "energy", "special_energy", "gust",
            "switch", "energy_acceleration", "recovery", "search",
            "hand_disruption", "energy_disruption",
        )
    }
    for card in card_data:
        card_id = int(card.cardId)
        kind = int(card.cardType)
        text = _text(card)
        if kind == 0:
            groups["pokemon"].add(card_id)
            groups["basic_pokemon" if card.basic else "evolution"].add(card_id)
        elif kind == 1:
            groups["item"].add(card_id)
        elif kind == 2:
            groups["tool"].add(card_id)
        elif kind == 3:
            groups["supporter"].add(card_id)
        elif kind == 4:
            groups["stadium"].add(card_id)
        elif kind in (5, 6):
            groups["energy"].add(card_id)
            if kind == 6:
                groups["special_energy"].add(card_id)
        if (
            "boss's orders" in text
            or ("opponent's benched pokÃ©mon" in text and "switch" in text)
        ):
            groups["gust"].add(card_id)
        if "switch your active pokÃ©mon" in text or "retreat cost is 0" in text:
            groups["switch"].add(card_id)
        if "attach" in text and "energy" in text:
            groups["energy_acceleration"].add(card_id)
        if "discard pile" in text and any(
            phrase in text for phrase in ("into your hand", "into your deck", "put")
        ):
            groups["recovery"].add(card_id)
        if "search your deck" in text:
            groups["search"].add(card_id)
        if "opponent's hand" in text and any(
            phrase in text for phrase in ("discard", "shuffle", "fewer")
        ):
            groups["hand_disruption"].add(card_id)
        if "energy" in text and "opponent" in text and any(
            phrase in text for phrase in ("discard", "move", "return")
        ):
            groups["energy_disruption"].add(card_id)
    return groups


def make_templates(rows) -> list[dict]:
    """Collapse duplicate deck lists while preserving their observed frequency."""
    grouped = {}
    for row in rows:
        deck = tuple(sorted(int(card) for card in row["deck"]))
        if len(deck) != 60:
            continue
        target = grouped.setdefault(
            deck,
            {
                "name": row.get("participantId") or row.get("archetype") or "deck",
                "family": row.get("family") or row.get("archetype") or "unknown",
                "deck": list(deck),
                "counts": Counter(deck),
                "priorCount": 0,
            },
        )
        target["priorCount"] += 1
    return list(grouped.values())


def _visible_cards(player, stadium, player_index):
    cards = []
    for card in player.get("discard") or []:
        if isinstance(card, dict):
            cards.append(card)
    for pokemon in (player.get("active") or []) + (player.get("bench") or []):
        if not isinstance(pokemon, dict):
            continue
        cards.append(pokemon)
        for key in ("energyCards", "tools", "preEvolution"):
            cards.extend(
                card for card in (pokemon.get(key) or []) if isinstance(card, dict)
            )
    if stadium and isinstance(stadium[0], dict):
        if int(stadium[0].get("playerIndex", -1)) == player_index:
            cards.append(stadium[0])
    cards.extend(
        card for card in (player.get("prize") or []) if isinstance(card, dict)
    )
    return cards


class OpponentCardCounter:
    """Posterior deck mixture, remaining-card odds, and game-aware risk level."""

    def __init__(self, templates, categories):
        self.templates = templates
        self.categories = categories
        self.seen_serials = set()
        self.seen = Counter()
        self.posterior = [1.0 / max(1, len(templates))] * len(templates)

    def observe(self, observation):
        current = observation.get("current") or {}
        players = current.get("players") or []
        own = int(current.get("yourIndex") or 0)
        opponent = 1 - own
        if opponent >= len(players):
            return
        for card in _visible_cards(
            players[opponent], current.get("stadium") or [], opponent
        ):
            serial = card.get("serial")
            identity = ("serial", serial) if serial is not None else (
                "fallback", card.get("id"), len(self.seen_serials)
            )
            if identity in self.seen_serials:
                continue
            self.seen_serials.add(identity)
            if card.get("id") is not None:
                self.seen[int(card["id"])] += 1
        self._update_posterior()

    def _update_posterior(self):
        if not self.templates:
            self.posterior = []
            return
        pokemon = self.categories["pokemon"]
        energy = self.categories["energy"]
        scores = []
        total_seen = sum(self.seen.values())
        for template in self.templates:
            counts = template["counts"]
            score = math.log(max(1, template["priorCount"]))
            for card_id, observed in self.seen.items():
                available = counts.get(card_id, 0)
                emphasis = 2.2 if card_id in pokemon else 0.55 if card_id in energy else 1.0
                # Robust likelihood: unseen rogue cards hurt but do not collapse
                # the posterior, allowing small deck edits and novel variants.
                score += emphasis * observed * math.log((available + 0.18) / 60.18)
                if observed > available:
                    score -= emphasis * 2.0 * (observed - available)
            score += 0.02 * total_seen
            scores.append(score)
        peak = max(scores)
        values = [math.exp(max(-60.0, score - peak)) for score in scores]
        normalizer = sum(values)
        self.posterior = [value / normalizer for value in values]

    def family_probabilities(self):
        probabilities = Counter()
        for probability, template in zip(self.posterior, self.templates):
            probabilities[template["family"]] += probability
        return dict(probabilities)

    def expected_remaining(self, category):
        ids = self.categories.get(category, set())
        expectation = 0.0
        for probability, template in zip(self.posterior, self.templates):
            remaining = sum(
                max(0, count - self.seen.get(card_id, 0))
                for card_id, count in template["counts"].items()
                if card_id in ids
            )
            expectation += probability * remaining
        return expectation

    def probability_hidden(self, category, hidden_cards):
        """Chance at least one category card occurs in a hidden sample."""
        hidden_cards = max(0, int(hidden_cards))
        if hidden_cards == 0:
            return 0.0
        probability_none = 0.0
        ids = self.categories.get(category, set())
        for posterior, template in zip(self.posterior, self.templates):
            remaining_counts = {
                card_id: max(0, count - self.seen.get(card_id, 0))
                for card_id, count in template["counts"].items()
            }
            population = sum(remaining_counts.values())
            hits = sum(
                count for card_id, count in remaining_counts.items() if card_id in ids
            )
            draws = min(hidden_cards, population)
            none = 1.0
            for offset in range(draws):
                denominator = population - offset
                if denominator <= 0:
                    break
                none *= max(0, population - hits - offset) / denominator
            probability_none += posterior * none
        return 1.0 - probability_none

    def risk_profile(self, observation):
        self.observe(observation)
        current = observation.get("current") or {}
        players = current.get("players") or []
        own = int(current.get("yourIndex") or 0)
        opponent = 1 - own
        if len(players) < 2:
            return {}
        me, rival = players[own], players[opponent]
        my_prizes = len(me.get("prize") or [])
        their_prizes = len(rival.get("prize") or [])
        my_bench = len(me.get("bench") or [])
        their_hand = int(rival.get("handCount") or 0)
        # Positive means we should accept more variance to recover.
        behind = max(-1.0, min(1.0, (my_prizes - their_prizes) / 3.0))
        no_backup = 1.0 if my_bench == 0 else 0.0
        risk_tolerance = max(
            0.05, min(0.95, 0.45 + 0.30 * behind - 0.22 * no_backup)
        )
        return {
            "riskTolerance": risk_tolerance,
            "behind": behind,
            "noBackup": bool(no_backup),
            "opponentHand": their_hand,
            "gustThreat": self.probability_hidden("gust", their_hand + 1),
            "switchThreat": self.probability_hidden("switch", their_hand + 1),
            "energyAccelerationThreat": self.probability_hidden(
                "energy_acceleration", their_hand + 1
            ),
            "recoveryThreat": self.probability_hidden("recovery", their_hand + 1),
            "energyDisruptionThreat": self.probability_hidden(
                "energy_disruption", their_hand + 1
            ),
            "expectedRemainingPokemon": self.expected_remaining("pokemon"),
            "expectedRemainingEnergy": self.expected_remaining("energy"),
            "expectedRemainingTrainers": sum(
                self.expected_remaining(group)
                for group in ("item", "tool", "supporter", "stadium")
            ),
            "familyProbabilities": self.family_probabilities(),
        }
