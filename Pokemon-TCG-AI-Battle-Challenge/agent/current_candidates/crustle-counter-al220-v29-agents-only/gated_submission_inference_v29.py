"""Submission-safe visible-family gate for the distilled GPU selector."""

from __future__ import annotations

import json
import os

from gpu_submission_inference_v28 import DistilledSelector


FAMILY_MARKERS = {
    "grimmsnarl": {646, 647, 648, 860},
    "rocket_mewtwo": {400, 401, 414, 431, 434},
    "dragapult": {119, 120, 121},
    "lucario": {333, 677, 678, 974},
    "alakazam": {245, 741, 742, 743},
    "garchomp": {379, 380, 381},
    "crustle_tusk": {58, 63, 344, 345, 533, 607},
    "festival": {89, 90, 91, 92, 93, 347, 921, 1245},
    "metal": {85, 86, 169, 170, 190, 666},
}


def classify_opponent(observation):
    current = observation.get("current") or {}
    players = current.get("players") or []
    own = int(current.get("yourIndex") or 0)
    if len(players) != 2:
        return "unknown", {}
    opponent = players[1 - own]
    visible = []
    for zone in ("active", "bench", "discard"):
        visible.extend(
            card for card in (opponent.get(zone) or [])
            if isinstance(card, dict)
        )
    visible.extend(
        card for card in (current.get("stadium") or [])
        if isinstance(card, dict)
    )
    ids = {
        int(card["id"])
        for card in visible
        if card.get("id") is not None
    }
    evidence = {
        family: len(markers & ids)
        for family, markers in FAMILY_MARKERS.items()
    }
    family, count = max(evidence.items(), key=lambda item: item[1])
    return (family if count else "unknown"), evidence


class GatedSelector(DistilledSelector):
    def __init__(
        self,
        threshold,
        allowed,
        use_counter=False,
        model_file="selector_weights_v28.json",
        templates_file="selector_templates_v29.json",
    ):
        super().__init__(threshold, model_file=model_file)
        self.normal_threshold = float(threshold)
        self.allowed = set(allowed)
        self.use_counter = bool(use_counter)
        self.templates_file = templates_file
        self.counter = None
        self._counter_factory = None
        if self.use_counter:
            from cg.api import all_card_data
            from opponent_card_counter_v28 import (
                OpponentCardCounter,
                classify_cards,
                make_templates,
            )

            root = os.path.dirname(os.path.abspath(__file__))
            rows = json.load(
                open(os.path.join(root, templates_file), encoding="utf-8")
            )
            templates = make_templates(rows)
            categories = classify_cards(all_card_data())
            self._counter_factory = lambda: OpponentCardCounter(
                templates, categories
            )
            self.counter = self._counter_factory()

    def reset(self):
        if self._counter_factory is not None:
            self.counter = self._counter_factory()

    def choose(self, observation, baseline):
        family, evidence = classify_opponent(observation)
        enabled = family in self.allowed and evidence.get(family, 0) > 0
        threshold = self.normal_threshold if enabled else 999.0
        if enabled and self.counter is not None:
            profile = self.counter.risk_profile(observation)
            tolerance = float(profile.get("riskTolerance", 0.45))
            threshold += (0.45 - tolerance) * 1.2
            if profile.get("noBackup"):
                threshold += 0.55
            if profile.get("gustThreat", 0.0) >= 0.55:
                threshold += 0.25
        self.threshold = threshold
        return super().choose(observation, baseline)

