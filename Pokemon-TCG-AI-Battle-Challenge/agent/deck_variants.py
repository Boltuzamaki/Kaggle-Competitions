"""Small legal deck-count hypotheses for EXP-19.

Every variant keeps the clean sequenced Archaludon policy unchanged. Public
deck or policy code is not imported here.
"""

from __future__ import annotations

from collections import Counter

from archaludon_policy import ARCHALUDON_DECK


# Card IDs in our independently built Archaludon list.
NIGHT_STRETCHER = 1097
POKEGEAR = 1122
JUMBO_ICE_CREAM = 1147
POKE_PAD = 1152
BOSS_ORDERS = 1182
JUDGE = 1213
RELICANTH = 57
METAL_ENERGY = 8


def _make_variant(*, remove: dict[int, int], add: dict[int, int]) -> list[int]:
    counts = Counter(ARCHALUDON_DECK)
    for card_id, amount in remove.items():
        if counts[card_id] < amount:
            raise ValueError(f"cannot remove {amount} copies of card {card_id}")
        counts[card_id] -= amount
    for card_id, amount in add.items():
        counts[card_id] += amount
    deck = list(counts.elements())
    if len(deck) != 60:
        raise ValueError(f"variant has {len(deck)} cards, expected 60")
    return deck


BASELINE = list(ARCHALUDON_DECK)

# Remove two situational Supporters; improve recovery and raw Metal access.
CONSISTENCY = _make_variant(
    remove={BOSS_ORDERS: 1, JUDGE: 1},
    add={NIGHT_STRETCHER: 1, METAL_ENERGY: 1},
)

# Trade one gust and one heal for stronger hand disruption plus Metal access.
ANTI_ALAKAZAM = _make_variant(
    remove={BOSS_ORDERS: 1, JUMBO_ICE_CREAM: 1},
    add={JUDGE: 1, METAL_ENERGY: 1},
)

# Reduce indirect Supporter search and gust density; maximize recovery and Metal.
RECOVERY = _make_variant(
    remove={POKEGEAR: 1, BOSS_ORDERS: 1},
    add={NIGHT_STRETCHER: 1, METAL_ENERGY: 1},
)

# Test whether Relicanth's previous-evolution attack access is worth one slot.
NO_RELICANTH = _make_variant(
    remove={RELICANTH: 1},
    add={METAL_ENERGY: 1},
)

# Test whether the fourth conditional heal is weaker than the twelfth Energy.
LESS_HEALING = _make_variant(
    remove={JUMBO_ICE_CREAM: 1},
    add={METAL_ENERGY: 1},
)

# EXP-21: retain Relicanth and isolate the opportunity cost of Energy copy 12.
RELIC_LESS_BOSS = _make_variant(
    remove={BOSS_ORDERS: 1},
    add={METAL_ENERGY: 1},
)

RELIC_NO_JUDGE = _make_variant(
    remove={JUDGE: 1},
    add={METAL_ENERGY: 1},
)

RELIC_LESS_GEAR = _make_variant(
    remove={POKEGEAR: 1},
    add={METAL_ENERGY: 1},
)

RELIC_LESS_STRETCHER = _make_variant(
    remove={NIGHT_STRETCHER: 1},
    add={METAL_ENERGY: 1},
)

RELIC_LESS_PAD = _make_variant(
    remove={POKE_PAD: 1},
    add={METAL_ENERGY: 1},
)


VARIANTS = {
    "baseline": BASELINE,
    "consistency": CONSISTENCY,
    "anti-alakazam": ANTI_ALAKAZAM,
    "recovery": RECOVERY,
    "no-relicanth": NO_RELICANTH,
    "less-healing": LESS_HEALING,
    "relic-less-boss": RELIC_LESS_BOSS,
    "relic-no-judge": RELIC_NO_JUDGE,
    "relic-less-gear": RELIC_LESS_GEAR,
    "relic-less-stretcher": RELIC_LESS_STRETCHER,
    "relic-less-pad": RELIC_LESS_PAD,
}
