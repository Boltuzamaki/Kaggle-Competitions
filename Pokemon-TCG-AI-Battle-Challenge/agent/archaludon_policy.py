"""Clean-room Archaludon/Cinderace policy built from card rules and game data.

This implementation does not import or call a public agent. The public agents
in ``tools/public_agents.py`` are frozen local benchmarks only. Decisions here
are derived from the official card metadata and these deck principles:

* start Cinderace to use Turbo Flare as early as possible;
* develop Duraludon, then evolve into Archaludon ex;
* route Metal Energy toward a three-energy attacker;
* discard Metal Energy when Assemble Alloy can recover it;
* protect and heal the powered Archaludon, then take efficient knockouts.
"""

from __future__ import annotations

from collections import Counter
import math

try:
    from cg.api import (
        AreaType,
        CardType,
        EnergyType,
        OptionType,
        SelectContext,
        all_attack,
        all_card_data,
        to_observation_class,
    )
    _OK = True
except Exception:
    _OK = False


# Deck counts were built from official card rules and high-level metagame hints.
# The policy below is our independent implementation. Counts sum to exactly 60.
ARCHALUDON_DECK = (
    [169] * 4       # Duraludon
    + [190] * 4     # Archaludon ex
    + [666] * 4     # Cinderace
    + [57]          # Relicanth
    + [1121] * 4    # Ultra Ball
    + [1122] * 4    # Pokegear 3.0
    + [1147] * 4    # Jumbo Ice Cream
    + [1152] * 4    # Poke Pad
    + [1097] * 3    # Night Stretcher
    + [1159]        # Hero's Cape
    + [1182] * 3    # Boss's Orders
    + [1185] * 4    # Explorer's Guidance
    + [1213]        # Judge
    + [1227] * 4    # Lillie's Determination
    + [1244] * 4    # Full Metal Lab
    + [8] * 11      # Basic Metal Energy
)

DURALUDON = 169
ARCHALUDON_EX = 190
CINDERACE = 666
RELICANTH = 57
METAL_ENERGY = 8

ULTRA_BALL = 1121
POKEGEAR = 1122
JUMBO_ICE_CREAM = 1147
POKE_PAD = 1152
NIGHT_STRETCHER = 1097
HERO_CAPE = 1159
BOSS = 1182
EXPLORER = 1185
JUDGE = 1213
LILLIE = 1227
FULL_METAL_LAB = 1244
ALAKAZAM_CARD = 743

RAZOR_FIN = 61
HAMMER_IN = 223
RAGING_HAMMER = 224
METAL_DEFENDER = 253
TURBO_FLARE = 965

CRUEL_ARROW = 183
COSMIC_BEAM = 980
POWERFUL_HAND = 1072

if _OK:
    _CARDS = {card.cardId: card for card in all_card_data()}
    _ATTACKS = {attack.attackId: attack for attack in all_attack()}
else:
    _CARDS = {}
    _ATTACKS = {}


def _card(obs, area, index, player):
    try:
        state = obs.current
        ps = state.players[player]
        if area == AreaType.DECK:
            return obs.select.deck[index]
        if area == AreaType.HAND:
            return ps.hand[index]
        if area == AreaType.DISCARD:
            return ps.discard[index]
        if area == AreaType.ACTIVE:
            return ps.active[index]
        if area == AreaType.BENCH:
            return ps.bench[index]
        if area == AreaType.PRIZE:
            return ps.prize[index]
        if area == AreaType.STADIUM:
            return state.stadium[index]
        if area == AreaType.LOOKING:
            return state.looking[index]
    except Exception:
        return None
    return None


def _stable_value(value):
    """Return a fully comparable representation for deterministic tie breaks."""
    if value is None:
        return ("none", "")
    if isinstance(value, (bool, int, float, str)):
        return (type(value).__name__, repr(value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (str(key), _stable_value(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ),
        )
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_stable_value(item) for item in value))
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return ("enum", type(value).__name__, repr(enum_value))
    return ("repr", repr(value))


def _option_semantic_key(option):
    """Describe an option without using its position in the legal-option list."""
    try:
        if isinstance(option, dict):
            return _stable_value(option)
        if hasattr(option, "model_dump"):
            return _stable_value(option.model_dump())
        if hasattr(option, "_asdict"):
            return _stable_value(option._asdict())
        if hasattr(option, "dict"):
            return _stable_value(option.dict())
        fields = getattr(type(option), "__annotations__", {})
        if fields:
            return _stable_value(
                {name: getattr(option, name, None) for name in fields}
            )
        if hasattr(option, "__dict__"):
            return _stable_value(vars(option))
    except Exception:
        pass
    names = (
        "type",
        "area",
        "index",
        "playerIndex",
        "benchIndex",
        "attackId",
        "abilityId",
        "skillId",
        "number",
        "cardId",
        "target",
    )
    return _stable_value(
        {name: getattr(option, name, None) for name in names}
    )


def _prizes(card_id):
    data = _CARDS.get(card_id)
    if data is None:
        return 1
    if getattr(data, "megaEx", False):
        return 3
    if getattr(data, "ex", False):
        return 2
    return 1


def _damage(attack_id, attacker, defender):
    attack = _ATTACKS.get(attack_id)
    damage = getattr(attack, "damage", 0) or 0
    if attack_id == RAGING_HAMMER and attacker is not None:
        damage += max(0, (getattr(attacker, "maxHp", 0) or 0) - attacker.hp)
    attacker_data = _CARDS.get(attacker.id) if attacker is not None else None
    defender_data = _CARDS.get(defender.id) if defender is not None else None
    attack_type = getattr(attacker_data, "energyType", None)
    if defender_data is not None and getattr(defender_data, "weakness", None) == attack_type:
        damage *= 2
    if defender_data is not None and getattr(defender_data, "resistance", None) == attack_type:
        damage = max(0, damage - 30)
    return damage


def _energy_count(pokemon):
    return len(getattr(pokemon, "energies", None) or []) if pokemon else 0


def _can_pay_after_one_attachment(pokemon, attack):
    """Whether an attack can be paid after one matching basic attachment."""
    if pokemon is None or attack is None:
        return False
    requirements = list(getattr(attack, "energies", None) or [])
    attached = list(getattr(pokemon, "energies", None) or [])
    pokemon_data = _CARDS.get(getattr(pokemon, "id", None))
    matching_type = getattr(pokemon_data, "energyType", EnergyType.COLORLESS)
    attached.append(matching_type)

    specific = Counter(
        energy for energy in requirements if energy != EnergyType.COLORLESS
    )
    available = Counter(attached)
    rainbow = available[EnergyType.RAINBOW]
    spent = 0
    for energy_type, needed in specific.items():
        direct = min(needed, available[energy_type])
        missing = needed - direct
        rainbow_used = min(missing, rainbow)
        if direct + rainbow_used < needed:
            return False
        rainbow -= rainbow_used
        spent += needed
    colorless = sum(
        energy == EnergyType.COLORLESS for energy in requirements
    )
    return len(attached) - spent >= colorless


class ArchaludonPolicy:
    def __init__(self, obs):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.me = self.state.yourIndex
        self.opponent = 1 - self.me
        self.mine = self.state.players[self.me]
        self.theirs = self.state.players[self.opponent]
        self.hand = Counter(card.id for card in (self.mine.hand or []))
        self.discard = Counter(card.id for card in self.mine.discard)
        self.field = Counter(
            pokemon.id
            for pokemon in list(self.mine.active) + list(self.mine.bench)
            if pokemon is not None
        )

    def my_board(self):
        return list(self.mine.active) + list(self.mine.bench)

    def opponent_board(self):
        return list(self.theirs.active) + list(self.theirs.bench)

    def active(self):
        return self.mine.active[0] if self.mine.active else None

    def opponent_active(self):
        return self.theirs.active[0] if self.theirs.active else None

    def _ready_score(self, pokemon):
        if pokemon is None:
            return -1e9
        energy = _energy_count(pokemon)
        score = pokemon.hp * 2.0 + energy * 500.0
        if pokemon.id == ARCHALUDON_EX:
            score += 15000.0 + (25000.0 if energy >= 3 else energy * 5000.0)
            if pokemon.tools:
                score += 3000.0
        elif pokemon.id == DURALUDON:
            score += 6000.0 + energy * 2500.0
            if energy >= 3:
                score += 6000.0
        elif pokemon.id == CINDERACE:
            score += 9000.0 if energy >= 1 and self.state.turn <= 3 else 500.0
        elif pokemon.id == RELICANTH:
            score += 1000.0
        return score

    def _best_power_target(self):
        board = self.my_board()
        ranked = sorted(
            enumerate(board),
            key=lambda pair: (
                pair[1] is not None and pair[1].id in (ARCHALUDON_EX, DURALUDON),
                pair[1] is not None and _energy_count(pair[1]) < 3,
                self._ready_score(pair[1]),
            ),
            reverse=True,
        )
        return ranked[0][0] if ranked and ranked[0][1] is not None else 0

    def _attack_score(self, option):
        active = self.active()
        target = self.opponent_active()
        attack_id = getattr(option, "attackId", None)
        damage = _damage(attack_id, active, target)
        score = 2000.0 + damage * 20.0
        if target is not None and damage >= target.hp:
            prize = _prizes(target.id)
            score += prize * 100000.0
            if len(self.theirs.prize) <= prize:
                score += 1000000.0
        if attack_id == TURBO_FLARE:
            needy = sum(
                max(0, 3 - _energy_count(pokemon))
                for pokemon in self.mine.bench
                if pokemon.id in (DURALUDON, ARCHALUDON_EX)
            )
            score += 180000.0 if needy else 1000.0
        elif attack_id == METAL_DEFENDER:
            score += 30000.0
        elif attack_id == RAGING_HAMMER:
            score += max(0, damage - 80) * 25.0
        return score

    def _main_score(self, option):
        option_type = option.type
        if option_type == OptionType.END:
            return -1e9
        if option_type == OptionType.ATTACK:
            return self._attack_score(option)
        if option_type == OptionType.ABILITY:
            return 95000.0
        if option_type == OptionType.EVOLVE:
            evolved = _card(self.obs, option.area, option.index, self.me)
            target = _card(self.obs, option.inPlayArea, option.inPlayIndex, self.me)
            if evolved and evolved.id == ARCHALUDON_EX and target and target.id == DURALUDON:
                return 110000.0 + _energy_count(target) * 1000.0
            return 30000.0
        if option_type == OptionType.ATTACH:
            target = _card(
                self.obs, option.inPlayArea, option.inPlayIndex, self.me
            )
            if target is None:
                return 1000.0
            energy = _energy_count(target)
            active = self.active()
            if target.id == CINDERACE and target is active and energy == 0 and self.state.turn <= 3:
                return 105000.0
            if target.id == ARCHALUDON_EX:
                return 80000.0 - max(0, energy - 2) * 30000.0
            if target.id == DURALUDON:
                bonus = 15000.0 if active and active.id == CINDERACE else 0.0
                return 70000.0 + bonus - max(0, energy - 2) * 25000.0
            return 5000.0
        if option_type == OptionType.RETREAT:
            active = self.active()
            bench_best = max((self._ready_score(p) for p in self.mine.bench), default=-1e9)
            active_score = self._ready_score(active)
            if active and active.id == CINDERACE and bench_best > active_score:
                return 85000.0
            if active and active.hp <= 80 and bench_best > active_score:
                return 60000.0
            return -1000.0
        if option_type == OptionType.PLAY:
            card = _card(self.obs, AreaType.HAND, option.index, self.me)
            return self._play_score(card)
        return 100.0

    def _play_score(self, card):
        if card is None:
            return 0.0
        card_id = card.id
        if card_id == DURALUDON:
            return 90000.0 if len(self.mine.bench) < self.mine.benchMax else -1000.0
        if card_id == FULL_METAL_LAB:
            current = self.state.stadium[0].id if self.state.stadium else None
            return -500.0 if current == FULL_METAL_LAB else 65000.0
        if card_id == HERO_CAPE:
            return 70000.0 if self.field[ARCHALUDON_EX] else 3000.0
        if card_id == JUMBO_ICE_CREAM:
            active = self.active()
            if active and _energy_count(active) >= 3 and active.maxHp - active.hp >= 50:
                return 100000.0 + (active.maxHp - active.hp) * 100.0
            return -5000.0
        if card_id == ULTRA_BALL:
            if self.field[DURALUDON] and not self.hand[ARCHALUDON_EX]:
                return 88000.0
            if not self.field[DURALUDON] and not self.hand[DURALUDON]:
                return 70000.0
            return 30000.0
        if card_id == POKE_PAD:
            return 76000.0 if self.field[DURALUDON] < 2 else 25000.0
        if card_id == POKEGEAR:
            has_supporter = any(
                getattr(_CARDS.get(cid), "cardType", None) == CardType.SUPPORTER
                for cid in self.hand
            )
            return 55000.0 if not has_supporter else 18000.0
        if card_id == NIGHT_STRETCHER:
            useful = any(self.discard[cid] for cid in (ARCHALUDON_EX, DURALUDON, METAL_ENERGY))
            return 58000.0 if useful else -2000.0
        if card_id == EXPLORER:
            alloy_ready = bool(self.field[DURALUDON] and self.hand[ARCHALUDON_EX])
            return 62000.0 + (15000.0 if alloy_ready else 0.0)
        if card_id == LILLIE:
            hand_count = self.mine.handCount
            return 78000.0 if hand_count <= 4 else 35000.0
        if card_id == JUDGE:
            return 50000.0 if self.theirs.handCount >= 6 else 15000.0
        if card_id == BOSS:
            active = self.active()
            if active is None:
                return 5000.0
            possible = max(
                (
                    _prizes(target.id) * 10000.0
                    + (50000.0 if target.hp <= 220 else 0.0)
                    for target in self.theirs.bench
                ),
                default=0.0,
            )
            return 55000.0 + possible
        return 10000.0

    def _selection_score(self, option):
        if option.type == OptionType.NUMBER:
            return float(getattr(option, "number", 0))
        if option.type == OptionType.YES:
            if self.context == SelectContext.IS_FIRST:
                return -100.0  # prefer going second for immediate Turbo Flare
            return 100.0
        if option.type == OptionType.NO:
            return 100.0 if self.context == SelectContext.IS_FIRST else 0.0
        if option.type == OptionType.SKILL:
            return 1000.0 if getattr(option, "cardId", None) == ARCHALUDON_EX else 100.0
        if option.type not in (
            OptionType.CARD,
            OptionType.TOOL_CARD,
            OptionType.ENERGY_CARD,
            OptionType.ENERGY,
        ):
            return 0.0
        card = _card(
            self.obs,
            option.area,
            option.index,
            getattr(option, "playerIndex", self.me),
        )
        if card is None:
            return 0.0

        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return {CINDERACE: 100000.0, DURALUDON: 20000.0, RELICANTH: 5000.0}.get(card.id, 0.0)
        if self.context == SelectContext.SETUP_BENCH_POKEMON:
            return {DURALUDON: 50000.0, RELICANTH: 12000.0, CINDERACE: -1000.0}.get(card.id, 0.0)
        if self.context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            return self._ready_score(card)
        if self.context in (SelectContext.TO_BENCH, SelectContext.TO_FIELD):
            return {DURALUDON: 60000.0, RELICANTH: 10000.0}.get(card.id, 0.0)
        if self.context == SelectContext.TO_HAND:
            return self._to_hand_score(card)
        if self.context in (SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM):
            return self._discard_score(card)
        if self.context in (SelectContext.ATTACH_FROM, SelectContext.EFFECT_TARGET):
            owner = getattr(option, "playerIndex", self.me)
            if owner == self.me and hasattr(card, "energies"):
                return self._attach_target_score(card)
            if owner == self.opponent and hasattr(card, "hp"):
                return self._opponent_target_score(card)
        if self.context == SelectContext.HEAL:
            return (getattr(card, "maxHp", 0) or 0) - getattr(card, "hp", 0)
        if self.context in (SelectContext.ATTACH_TO, SelectContext.TO_HAND_ENERGY):
            return 100.0 if card.id == METAL_ENERGY else 0.0
        return 10.0

    def _to_hand_score(self, card):
        if card.id == ARCHALUDON_EX:
            return 100000.0 if self.field[DURALUDON] else 50000.0
        if card.id == DURALUDON:
            return 90000.0 if not self.field[DURALUDON] else 45000.0
        if card.id == METAL_ENERGY:
            return 55000.0
        if card.id == RELICANTH:
            return 15000.0 if self.field[ARCHALUDON_EX] else 3000.0
        if card.id == CINDERACE:
            return 1000.0
        data = _CARDS.get(card.id)
        if data and data.cardType == CardType.SUPPORTER:
            return 30000.0
        return 10000.0

    def _discard_score(self, card):
        # Selection ranks highest first: expendable cards get the highest value.
        if card.id == METAL_ENERGY:
            alloy_line = self.field[DURALUDON] and self.hand[ARCHALUDON_EX]
            return 90000.0 if alloy_line else 30000.0
        if card.id == CINDERACE and self.active() and self.active().id != CINDERACE:
            return 80000.0
        if self.hand[card.id] >= 3:
            return 50000.0
        if card.id in (HERO_CAPE, ARCHALUDON_EX, DURALUDON):
            return -50000.0
        return 15000.0

    def _attach_target_score(self, pokemon):
        energy = _energy_count(pokemon)
        if pokemon.id == ARCHALUDON_EX:
            return 100000.0 - max(0, energy - 2) * 50000.0
        if pokemon.id == DURALUDON:
            return 90000.0 - max(0, energy - 2) * 45000.0
        if pokemon.id == CINDERACE and pokemon is self.active() and energy == 0:
            return 80000.0
        return 1000.0

    def _opponent_target_score(self, pokemon):
        active = self.active()
        damage = 220 if active and active.id == ARCHALUDON_EX and _energy_count(active) >= 3 else 0
        score = _prizes(pokemon.id) * 20000.0 - pokemon.hp * 10.0
        if damage >= pokemon.hp:
            score += 100000.0 + _prizes(pokemon.id) * 100000.0
        return score

    def choose(self):
        options = self.select.option
        if not options:
            return []
        if self.context == SelectContext.MAIN:
            scores = [self._main_score(option) for option in options]
        else:
            scores = [self._selection_score(option) for option in options]
        ranked = sorted(
            range(len(options)),
            key=lambda i: (scores[i], _option_semantic_key(options[i])),
            reverse=True,
        )

        minimum = max(0, self.select.minCount or 0)
        maximum = min(len(options), self.select.maxCount or 1)
        count = minimum
        while count < maximum and scores[ranked[count]] > 0:
            count += 1
        if minimum == 0 and count == 0 and maximum > 0 and self.context == SelectContext.MAIN:
            count = 1
        return ranked[:count]


class BossGateMixin:
    """Spend Boss's Orders only when gusting creates a better knockout."""

    def _available_attack_ids(self):
        if self.context == SelectContext.MAIN:
            legal = [
                getattr(option, "attackId", None)
                for option in self.select.option
                if option.type == OptionType.ATTACK
            ]
            if legal:
                return [attack_id for attack_id in legal if attack_id is not None]

        active = self.active()
        energy = _energy_count(active)
        if active is None:
            return []
        if active.id == CINDERACE:
            return [TURBO_FLARE] if energy >= 1 else []
        if active.id == DURALUDON:
            result = [HAMMER_IN] if energy >= 1 else []
            if energy >= 3:
                result.append(RAGING_HAMMER)
            return result
        if active.id == ARCHALUDON_EX:
            result = [METAL_DEFENDER] if energy >= 3 else []
            if self.field[RELICANTH]:
                if energy >= 1:
                    result.append(HAMMER_IN)
                if energy >= 3:
                    result.append(RAGING_HAMMER)
            return result
        if active.id == RELICANTH:
            return [RAZOR_FIN] if energy >= 2 else []
        return []

    def _best_damage(self, target):
        active = self.active()
        return max(
            (_damage(attack_id, active, target) for attack_id in self._available_attack_ids()),
            default=0,
        )

    def _boss_targets(self):
        return [
            target
            for target in self.theirs.bench
            if self._best_damage(target) >= target.hp
        ]

    def _play_score(self, card):
        if card is None or card.id != BOSS:
            return super()._play_score(card)
        targets = self._boss_targets()
        if not targets:
            return -5000.0
        current = self.opponent_active()
        current_prizes = (
            _prizes(current.id)
            if current is not None and self._best_damage(current) >= current.hp
            else 0
        )
        best_prizes = max(_prizes(target.id) for target in targets)
        winning_gust = len(self.mine.prize) <= best_prizes
        if current_prizes and best_prizes <= current_prizes and not winning_gust:
            return -1000.0
        return 300000.0 + best_prizes * 100000.0

    def _opponent_target_score(self, pokemon):
        damage = self._best_damage(pokemon)
        if damage >= pokemon.hp:
            prizes = _prizes(pokemon.id)
            return 500000.0 + prizes * 100000.0 - pokemon.hp
        return -pokemon.hp


class BossGatedArchaludonPolicy(BossGateMixin, ArchaludonPolicy):
    pass


class SequencedArchaludonPolicy(ArchaludonPolicy):
    """Defer nonlethal attacks until useful main-phase actions are exhausted."""

    def _attack_score(self, option):
        score = super()._attack_score(option)
        active = self.active()
        target = self.opponent_active()
        damage = _damage(getattr(option, "attackId", None), active, target)
        if target is not None and damage >= target.hp:
            return score
        # Positive and above END/negative no-ops, but below every useful setup,
        # draw, evolution, attachment, healing, and search action.
        return min(score, 12000.0)


class SequencedBossGatedArchaludonPolicy(BossGateMixin, SequencedArchaludonPolicy):
    pass


class PrizeRaceBossMixin:
    """Spend Boss only when the reachable bench prize race is strictly faster.

    This is intentionally narrower than a generic Boss heuristic. It compares
    targets using attacks that are legal from the current active Pokemon, the
    prizes awarded by a knockout, attacks required at current damage, and
    whether that knockout ends the game. It does not copy or call any benchmark
    policy.
    """

    def _available_attack_ids(self):
        if self.context == SelectContext.MAIN:
            legal = [
                getattr(option, "attackId", None)
                for option in self.select.option
                if option.type == OptionType.ATTACK
            ]
            if legal:
                return [attack_id for attack_id in legal if attack_id is not None]

        active = self.active()
        if active is None:
            return []
        energy = _energy_count(active)
        if active.id == CINDERACE:
            return [TURBO_FLARE] if energy >= 1 else []
        if active.id == DURALUDON:
            result = [HAMMER_IN] if energy >= 1 else []
            if energy >= 3:
                result.append(RAGING_HAMMER)
            return result
        if active.id == ARCHALUDON_EX:
            result = [METAL_DEFENDER] if energy >= 3 else []
            if self.field[RELICANTH]:
                if energy >= 1:
                    result.append(HAMMER_IN)
                if energy >= 3:
                    result.append(RAGING_HAMMER)
            return result
        if active.id == RELICANTH:
            return [RAZOR_FIN] if energy >= 2 else []
        return []

    def _prize_race(self, target):
        if target is None:
            return None
        active = self.active()
        damage = max(
            (
                _damage(attack_id, active, target)
                for attack_id in self._available_attack_ids()
            ),
            default=0,
        )
        if damage <= 0:
            return None
        attacks = max(1, math.ceil(target.hp / damage))
        prizes = min(len(self.mine.prize), _prizes(target.id))
        wins_game = prizes >= len(self.mine.prize)
        value = prizes * 100000.0 / attacks
        if wins_game:
            value += 1000000.0
        if attacks == 1:
            value += 20000.0
        return value, attacks, prizes, wins_game

    def _best_bench_race(self):
        races = [
            (self._prize_race(target), target)
            for target in self.theirs.bench
            if target is not None
        ]
        races = [pair for pair in races if pair[0] is not None]
        return max(races, key=lambda pair: pair[0][0], default=(None, None))

    def _play_score(self, card):
        if card is None or card.id != BOSS:
            return super()._play_score(card)
        best_race, _ = self._best_bench_race()
        if best_race is None:
            return -5000.0
        current_race = self._prize_race(self.opponent_active())
        current_value = current_race[0] if current_race is not None else 0.0
        improvement = best_race[0] - current_value
        if improvement <= 0:
            return -5000.0
        # A winning gust outranks every setup action. Other improvements remain
        # useful but do not outrank an already legal knockout attack.
        if best_race[3] and not (current_race and current_race[3]):
            return 900000.0 + improvement
        return 70000.0 + min(improvement, 180000.0)

    def _opponent_target_score(self, pokemon):
        race = self._prize_race(pokemon)
        if race is None:
            return -getattr(pokemon, "hp", 0)
        return race[0] - getattr(pokemon, "hp", 0) * 0.01


class PrizeRaceSequencedArchaludonPolicy(
    PrizeRaceBossMixin, SequencedArchaludonPolicy
):
    pass


class TacticalEscapeMixin:
    """Active-only next-attack threat model with knockout preservation."""

    def _opponent_attack_damage(self, attacker, defender, attack_id):
        if attacker is None or defender is None:
            return 0
        if attack_id == POWERFUL_HAND:
            # Powerful Hand places counters, so weakness, resistance, and Full
            # Metal Lab do not change the amount.
            return max(0, self.theirs.handCount * 20)
        if attack_id == CRUEL_ARROW:
            damage = 100
        elif attack_id == COSMIC_BEAM:
            has_lunatone = any(
                getattr(pokemon, "id", None) == 675
                for pokemon in self.theirs.bench
            )
            damage = 70 if has_lunatone else 0
        else:
            damage = _damage(attack_id, attacker, defender)

        stadium_id = self.state.stadium[0].id if self.state.stadium else None
        defender_data = _CARDS.get(getattr(defender, "id", None))
        if (
            stadium_id == FULL_METAL_LAB
            and getattr(defender_data, "energyType", None) == EnergyType.METAL
        ):
            damage = max(0, damage - 30)
        return damage

    def _opponent_best_next_damage(self, defender):
        attacker = self.opponent_active()
        attacker_data = _CARDS.get(getattr(attacker, "id", None))
        attacks = getattr(attacker_data, "attacks", None) or []
        return max(
            (
                self._opponent_attack_damage(attacker, defender, attack_id)
                for attack_id in attacks
                if _can_pay_after_one_attachment(attacker, _ATTACKS.get(attack_id))
            ),
            default=0,
        )

    def _under_next_attack_ko(self, defender=None):
        defender = defender or self.active()
        return bool(
            defender is not None
            and self._opponent_best_next_damage(defender) >= defender.hp
        )

    def _has_immediate_ko(self):
        target = self.opponent_active()
        active = self.active()
        if target is None or active is None or self.context != SelectContext.MAIN:
            return False
        return any(
            option.type == OptionType.ATTACK
            and _damage(getattr(option, "attackId", None), active, target) >= target.hp
            for option in self.select.option
        )

    def _main_score(self, option):
        score = super()._main_score(option)
        if option.type != OptionType.RETREAT:
            return score
        if not self._under_next_attack_ko() or self._has_immediate_ko():
            return score
        survivable = [
            pokemon
            for pokemon in self.mine.bench
            if not self._under_next_attack_ko(pokemon)
        ]
        if not survivable:
            return score
        return 700000.0 + max(self._ready_score(pokemon) for pokemon in survivable)

    def _selection_score(self, option):
        score = super()._selection_score(option)
        if self.context not in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            return score
        # Do not leak tactical survival scoring into ordinary retreats or
        # post-knockout replacement selection. The first EXP-26 screen showed
        # that this global scope changed many unrelated choices.
        if not self._under_next_attack_ko():
            return score
        card = _card(
            self.obs,
            option.area,
            option.index,
            getattr(option, "playerIndex", self.me),
        )
        if card is None:
            return score
        incoming = self._opponent_best_next_damage(card)
        survival_margin = card.hp - incoming
        if survival_margin > 0:
            return score + 500000.0 + survival_margin * 100.0
        return score - 200000.0 + survival_margin * 100.0


class TacticalSequencedArchaludonPolicy(
    TacticalEscapeMixin, SequencedArchaludonPolicy
):
    pass


class SupporterTimingMixin:
    """Allow deterministic draw and disruption Supporters only in useful states."""

    def _play_score(self, card):
        if card is None:
            return super()._play_score(card)
        if card.id == JUDGE:
            own_after_play = max(0, self.mine.handCount - 1)
            own_draw_gain = 4 - own_after_play
            opponent_active = self.opponent_active()
            stops_hand_lethal = bool(
                opponent_active is not None
                and opponent_active.id == ALAKAZAM_CARD
                and self.theirs.handCount >= 7
            )
            beneficial_disruption = bool(
                self.theirs.handCount >= 7 and own_draw_gain >= 0
            )
            if stops_hand_lethal:
                return 100000.0 + self.theirs.handCount * 1000.0
            if beneficial_disruption:
                return 65000.0 + own_draw_gain * 10000.0
            return -5000.0
        if card.id == LILLIE:
            draw_to = 8 if len(self.mine.prize) == 6 else 6
            own_after_play = max(0, self.mine.handCount - 1)
            net_gain = draw_to - own_after_play
            if net_gain <= 0:
                return -5000.0
            return 45000.0 + net_gain * 12000.0
        if card.id == EXPLORER and self.mine.deckCount <= 8:
            return -5000.0
        return super()._play_score(card)


class SupporterTimedSequencedArchaludonPolicy(
    SupporterTimingMixin, SequencedArchaludonPolicy
):
    pass


class TurnMacroMixin:
    """Choose a visible turn objective before comparing action scores."""

    def _legal_knockout_options(self):
        active = self.active()
        target = self.opponent_active()
        if active is None or target is None or self.context != SelectContext.MAIN:
            return []
        return [
            option
            for option in self.select.option
            if option.type == OptionType.ATTACK
            and _damage(getattr(option, "attackId", None), active, target) >= target.hp
        ]

    def _turn_objective(self):
        target = self.opponent_active()
        knockouts = self._legal_knockout_options()
        if knockouts and target is not None:
            if _prizes(target.id) >= len(self.mine.prize):
                return "finish"
            return "take-prize"

        active = self.active()
        missing_hp = (
            max(0, active.maxHp - active.hp) if active is not None else 0
        )
        if active is not None and missing_hp >= 50 and self.hand[JUMBO_ICE_CREAM]:
            return "stabilize"

        metal_ready = any(
            pokemon.id in (DURALUDON, ARCHALUDON_EX)
            and _energy_count(pokemon) >= 3
            for pokemon in self.my_board()
            if pokemon is not None
        )
        if not self.field[ARCHALUDON_EX] or not metal_ready:
            return "setup"
        return "press"

    def _main_score(self, option):
        score = super()._main_score(option)
        objective = self._turn_objective()
        if objective in ("finish", "take-prize"):
            if option in self._legal_knockout_options():
                return score + (2000000.0 if objective == "finish" else 500000.0)
            return score
        if objective == "stabilize" and option.type == OptionType.PLAY:
            card = _card(self.obs, AreaType.HAND, option.index, self.me)
            if card is not None and card.id == JUMBO_ICE_CREAM:
                return score + 250000.0
            return score
        if objective != "setup":
            return score

        if option.type in (OptionType.EVOLVE, OptionType.ATTACH, OptionType.ABILITY):
            return score + 120000.0
        if option.type == OptionType.PLAY:
            card = _card(self.obs, AreaType.HAND, option.index, self.me)
            if card is None:
                return score
            if card.id in (BOSS, JUDGE):
                return -5000.0
            if card.id in (
                DURALUDON,
                ULTRA_BALL,
                POKE_PAD,
                POKEGEAR,
                EXPLORER,
                LILLIE,
            ):
                return score + 90000.0
        return score


class MacroSequencedArchaludonPolicy(TurnMacroMixin, SequencedArchaludonPolicy):
    pass


class EnergyOpportunityMixin:
    """Rank attachments by unlock value and likely Energy survival."""

    _opponent_attack_damage = TacticalEscapeMixin._opponent_attack_damage
    _opponent_best_next_damage = TacticalEscapeMixin._opponent_best_next_damage

    def _attack_ids_at_energy(self, pokemon, energy):
        if pokemon.id == CINDERACE:
            return [TURBO_FLARE] if energy >= 1 else []
        if pokemon.id == DURALUDON:
            result = [HAMMER_IN] if energy >= 1 else []
            if energy >= 3:
                result.append(RAGING_HAMMER)
            return result
        if pokemon.id == ARCHALUDON_EX:
            result = [METAL_DEFENDER] if energy >= 3 else []
            if self.field[RELICANTH]:
                if energy >= 1:
                    result.append(HAMMER_IN)
                if energy >= 3:
                    result.append(RAGING_HAMMER)
            return result
        if pokemon.id == RELICANTH:
            return [RAZOR_FIN] if energy >= 2 else []
        return []

    def _attachment_opportunity_score(self, target):
        if target is None:
            return 1000.0
        before_energy = _energy_count(target)
        after_energy = before_energy + 1
        before_attacks = self._attack_ids_at_energy(target, before_energy)
        after_attacks = self._attack_ids_at_energy(target, after_energy)
        newly_unlocked = [attack for attack in after_attacks if attack not in before_attacks]

        score = {
            ARCHALUDON_EX: 70000.0,
            DURALUDON: 65000.0,
            CINDERACE: 12000.0,
            RELICANTH: 4000.0,
        }.get(target.id, 1000.0)
        if newly_unlocked:
            score += 90000.0
            score += max(
                _damage(attack_id, target, self.opponent_active())
                for attack_id in newly_unlocked
            ) * 200.0
        if (
            target.id == CINDERACE
            and target is self.active()
            and before_energy == 0
            and self.state.turn <= 3
        ):
            score += 120000.0

        if before_energy >= 3:
            score -= 160000.0 + (before_energy - 3) * 40000.0

        if target is self.active():
            incoming = self._opponent_best_next_damage(target)
            likely_ko = incoming >= target.hp
            opponent = self.opponent_active()
            knockout_after_attach = bool(
                opponent is not None
                and any(
                    _damage(attack_id, target, opponent) >= opponent.hp
                    for attack_id in after_attacks
                )
            )
            if likely_ko and not knockout_after_attach:
                score -= 170000.0 + after_energy * 15000.0
            target_data = _CARDS.get(target.id)
            retreat_cost = getattr(target_data, "retreatCost", 0)
            if before_energy < retreat_cost <= after_energy and likely_ko:
                score += 45000.0
        else:
            score += 20000.0
        return score

    def _main_score(self, option):
        if option.type != OptionType.ATTACH:
            return super()._main_score(option)
        target = _card(
            self.obs, option.inPlayArea, option.inPlayIndex, self.me
        )
        return self._attachment_opportunity_score(target)


class EnergyOpportunitySequencedArchaludonPolicy(
    EnergyOpportunityMixin, SequencedArchaludonPolicy
):
    pass


class RecoveryInventoryMixin:
    """Spend recovery and discard resources from visible inventory state.

    Hidden prize cards remain unknown, so this planner uses only our original
    deck counts and cards that are currently visible in hand, discard, and play.
    It deliberately avoids treating unseen cards as guaranteed draws.
    """

    _original_counts = Counter(ARCHALUDON_DECK)

    def _attached_metal(self):
        return sum(
            _energy_count(pokemon)
            for pokemon in self.my_board()
            if pokemon is not None
            and pokemon.id in (DURALUDON, ARCHALUDON_EX, CINDERACE, RELICANTH)
        )

    def _visible_inventory(self, card_id):
        visible = self.hand[card_id] + self.discard[card_id]
        if card_id == DURALUDON:
            # An evolved Archaludon still consumes one Duraludon from the deck.
            visible += self.field[DURALUDON] + self.field[ARCHALUDON_EX]
        elif card_id in (ARCHALUDON_EX, CINDERACE, RELICANTH):
            visible += self.field[card_id]
        elif card_id == METAL_ENERGY:
            visible += self._attached_metal()
        return visible

    def _unseen_inventory(self, card_id):
        return max(
            0,
            self._original_counts[card_id] - self._visible_inventory(card_id),
        )

    def _energy_deficit(self):
        return sum(
            max(0, 3 - _energy_count(pokemon))
            for pokemon in self.my_board()
            if pokemon is not None
            and pokemon.id in (DURALUDON, ARCHALUDON_EX)
        )

    def _needs_recovery(self, card_id):
        if card_id == ARCHALUDON_EX:
            return bool(self.field[DURALUDON] and not self.hand[ARCHALUDON_EX])
        if card_id == DURALUDON:
            developed = self.field[DURALUDON] + self.field[ARCHALUDON_EX]
            return bool(
                developed < 2
                and not self.hand[DURALUDON]
                and len(self.mine.bench) < self.mine.benchMax
            )
        if card_id == METAL_ENERGY:
            alloy_ready = bool(self.field[DURALUDON] and self.hand[ARCHALUDON_EX])
            return bool(
                self._energy_deficit() > 0
                and not self.hand[METAL_ENERGY]
                and not alloy_ready
            )
        return False

    def _play_score(self, card):
        if card is None or card.id != NIGHT_STRETCHER:
            return super()._play_score(card)
        recoverable = [
            card_id
            for card_id in (ARCHALUDON_EX, DURALUDON, METAL_ENERGY)
            if self.discard[card_id] and self._needs_recovery(card_id)
        ]
        if not recoverable:
            return -5000.0
        if ARCHALUDON_EX in recoverable:
            return 98000.0
        if DURALUDON in recoverable:
            return 88000.0
        return 68000.0

    def _to_hand_score(self, card):
        if card.id in (ARCHALUDON_EX, DURALUDON, METAL_ENERGY):
            if self._needs_recovery(card.id):
                urgency = {
                    ARCHALUDON_EX: 140000.0,
                    DURALUDON: 125000.0,
                    METAL_ENERGY: 80000.0,
                }
                return urgency[card.id]
            # Keep normal search behavior for unseen deck choices, but make an
            # unnecessary recovery target unattractive.
            if self.discard[card.id]:
                return -5000.0
        return super()._to_hand_score(card)

    def _discard_score(self, card):
        if card.id == METAL_ENERGY:
            total_visible_energy = self._visible_inventory(METAL_ENERGY)
            alloy_ready = bool(self.field[DURALUDON] and self.hand[ARCHALUDON_EX])
            if alloy_ready and self.discard[METAL_ENERGY] < 2:
                return 100000.0
            if (
                self._energy_deficit() > 0
                and self.hand[METAL_ENERGY] <= 1
                and total_visible_energy <= 6
            ):
                return -35000.0
            if self.hand[METAL_ENERGY] >= 2:
                return 55000.0
            return 8000.0
        if card.id == NIGHT_STRETCHER:
            valuable_discard = any(
                self.discard[card_id] and self._needs_recovery(card_id)
                for card_id in (ARCHALUDON_EX, DURALUDON, METAL_ENERGY)
            )
            if valuable_discard and self.hand[NIGHT_STRETCHER] <= 1:
                return -45000.0
        if card.id in (ARCHALUDON_EX, DURALUDON):
            if self.hand[card.id] <= 1 and self._unseen_inventory(card.id) <= 1:
                return -90000.0
        return super()._discard_score(card)


class RecoveryInventorySequencedArchaludonPolicy(
    RecoveryInventoryMixin, SequencedArchaludonPolicy
):
    pass


class EndTurnRegretMixin:
    """Prefer END over a legal action whose domain score is harmful."""

    def _main_score(self, option):
        if option.type == OptionType.END:
            return 0.0
        return super()._main_score(option)


class EndTurnRegretSequencedArchaludonPolicy(
    EndTurnRegretMixin, SequencedArchaludonPolicy
):
    pass


class BenchCapacityMixin:
    """Reserve bench space for the two-stage Archaludon attack plan."""

    def _bench_slots(self):
        return max(0, self.mine.benchMax - len(self.mine.bench))

    def _developed_lines(self):
        return self.field[DURALUDON] + self.field[ARCHALUDON_EX]

    def _bench_card_score(self, card):
        if card is None:
            return None
        if card.id == DURALUDON:
            if self._developed_lines() >= 2:
                return -5000.0
            return 105000.0
        if card.id == RELICANTH:
            if not self.field[ARCHALUDON_EX] or self._bench_slots() <= 1:
                return -5000.0
            return 18000.0
        if card.id == CINDERACE:
            # Cinderace is the preferred setup active, not a late bench filler.
            return -5000.0 if self.active() is not None else 20000.0
        return None

    def _main_score(self, option):
        if option.type != OptionType.PLAY:
            return super()._main_score(option)
        card = _card(self.obs, AreaType.HAND, option.index, self.me)
        bench_score = self._bench_card_score(card)
        if bench_score is not None:
            return bench_score
        return super()._main_score(option)

    def _selection_score(self, option):
        score = super()._selection_score(option)
        if self.context not in (SelectContext.TO_BENCH, SelectContext.TO_FIELD):
            return score
        card = _card(
            self.obs,
            option.area,
            option.index,
            getattr(option, "playerIndex", self.me),
        )
        bench_score = self._bench_card_score(card)
        return bench_score if bench_score is not None else score


class BenchCapacitySequencedArchaludonPolicy(
    BenchCapacityMixin, SequencedArchaludonPolicy
):
    pass


class EvolutionStackPreservationMixin:
    """Avoid spending search and discard resources outside a viable ex line."""

    def _play_score(self, card):
        if card is None or card.id != ULTRA_BALL:
            return super()._play_score(card)
        needs_archaludon = bool(
            self.field[DURALUDON] and not self.hand[ARCHALUDON_EX]
        )
        needs_duraludon = bool(
            not self.field[DURALUDON]
            and not self.field[ARCHALUDON_EX]
            and not self.hand[DURALUDON]
            and len(self.mine.bench) < self.mine.benchMax
        )
        if needs_archaludon:
            return 108000.0
        if needs_duraludon:
            return 92000.0
        return -5000.0

    def _discard_score(self, card):
        if card.id == ARCHALUDON_EX:
            if self.field[DURALUDON] and self.hand[ARCHALUDON_EX] <= 1:
                return -220000.0
            return -120000.0
        if card.id == DURALUDON:
            developed = self.field[DURALUDON] + self.field[ARCHALUDON_EX]
            if developed < 2 and self.hand[DURALUDON] <= 1:
                return -200000.0
            return -110000.0
        return super()._discard_score(card)


class EvolutionStackSequencedArchaludonPolicy(
    EvolutionStackPreservationMixin, SequencedArchaludonPolicy
):
    pass


class DamageBreakpointMixin:
    """Score attacks from exact current damage after visible board modifiers."""

    def _current_attack_ids(self):
        if self.context == SelectContext.MAIN:
            legal = [
                getattr(option, "attackId", None)
                for option in self.select.option
                if option.type == OptionType.ATTACK
            ]
            if legal:
                return [attack_id for attack_id in legal if attack_id is not None]
        active = self.active()
        energy = _energy_count(active)
        if active is None:
            return []
        if active.id == CINDERACE:
            return [TURBO_FLARE] if energy >= 1 else []
        if active.id == DURALUDON:
            result = [HAMMER_IN] if energy >= 1 else []
            if energy >= 3:
                result.append(RAGING_HAMMER)
            return result
        if active.id == ARCHALUDON_EX:
            result = [METAL_DEFENDER] if energy >= 3 else []
            if self.field[RELICANTH]:
                if energy >= 1:
                    result.append(HAMMER_IN)
                if energy >= 3:
                    result.append(RAGING_HAMMER)
            return result
        if active.id == RELICANTH:
            return [RAZOR_FIN] if energy >= 2 else []
        return []

    def _effective_damage(self, attack_id, target):
        damage = _damage(attack_id, self.active(), target)
        stadium_id = self.state.stadium[0].id if self.state.stadium else None
        target_data = _CARDS.get(getattr(target, "id", None))
        if (
            stadium_id == FULL_METAL_LAB
            and getattr(target_data, "energyType", None) == EnergyType.METAL
        ):
            damage = max(0, damage - 30)
        return damage

    def _attack_score(self, option):
        active = self.active()
        target = self.opponent_active()
        attack_id = getattr(option, "attackId", None)
        damage = self._effective_damage(attack_id, target)
        score = 2000.0 + damage * 20.0
        if target is not None and damage >= target.hp:
            prize = _prizes(target.id)
            score += prize * 100000.0
            if len(self.theirs.prize) <= prize:
                score += 1000000.0
        if attack_id == TURBO_FLARE:
            needy = sum(
                max(0, 3 - _energy_count(pokemon))
                for pokemon in self.mine.bench
                if pokemon.id in (DURALUDON, ARCHALUDON_EX)
            )
            score += 180000.0 if needy else 1000.0
        elif attack_id == METAL_DEFENDER:
            score += 30000.0
        elif attack_id == RAGING_HAMMER:
            score += max(0, damage - 80) * 25.0
        return score

    def _play_score(self, card):
        if card is None or card.id != BOSS:
            return super()._play_score(card)
        reachable = [
            target
            for target in self.theirs.bench
            if any(
                self._effective_damage(attack_id, target) >= target.hp
                for attack_id in self._current_attack_ids()
            )
        ]
        if not reachable:
            return -5000.0
        best = max(reachable, key=lambda target: (_prizes(target.id), -target.hp))
        return 260000.0 + _prizes(best.id) * 100000.0

    def _opponent_target_score(self, pokemon):
        damage = max(
            (
                self._effective_damage(attack_id, pokemon)
                for attack_id in self._current_attack_ids()
            ),
            default=0,
        )
        if damage >= pokemon.hp:
            return 500000.0 + _prizes(pokemon.id) * 100000.0 - pokemon.hp
        return -pokemon.hp


class DamageBreakpointSequencedArchaludonPolicy(
    DamageBreakpointMixin, SequencedArchaludonPolicy
):
    pass


class StadiumTimingMixin:
    """Play Full Metal Lab only when its visible net damage value is positive."""

    _opponent_attack_damage = TacticalEscapeMixin._opponent_attack_damage
    _opponent_best_next_damage = TacticalEscapeMixin._opponent_best_next_damage

    def _metal_pokemon(self, pokemon):
        data = _CARDS.get(getattr(pokemon, "id", None))
        return getattr(data, "energyType", None) == EnergyType.METAL

    def _play_score(self, card):
        if card is None or card.id != FULL_METAL_LAB:
            return super()._play_score(card)
        current = self.state.stadium[0].id if self.state.stadium else None
        if current == FULL_METAL_LAB:
            return -5000.0
        active = self.active()
        opponent = self.opponent_active()
        own_metal = sum(
            self._metal_pokemon(pokemon)
            for pokemon in self.my_board()
            if pokemon is not None
        )
        opponent_metal = sum(
            self._metal_pokemon(pokemon)
            for pokemon in self.opponent_board()
            if pokemon is not None
        )
        score = (own_metal - opponent_metal) * 18000.0
        if active is not None and self._metal_pokemon(active):
            incoming_without_lab = self._opponent_best_next_damage(active)
            incoming_with_lab = max(0, incoming_without_lab - 30)
            if incoming_without_lab >= active.hp > incoming_with_lab:
                score += 260000.0
        if opponent is not None and self._metal_pokemon(opponent):
            current_damage = max(
                (
                    _damage(attack_id, active, opponent)
                    for attack_id in DamageBreakpointMixin._current_attack_ids(self)
                ),
                default=0,
            )
            if current_damage >= opponent.hp > max(0, current_damage - 30):
                score -= 320000.0
        return 60000.0 + score if score > 0 else -5000.0


class StadiumTimingSequencedArchaludonPolicy(
    StadiumTimingMixin, SequencedArchaludonPolicy
):
    pass


class HealingBreakpointMixin:
    """Use 80 healing only when it changes the incoming attack count."""

    _opponent_attack_damage = TacticalEscapeMixin._opponent_attack_damage
    _opponent_best_next_damage = TacticalEscapeMixin._opponent_best_next_damage

    def _play_score(self, card):
        if card is None or card.id != JUMBO_ICE_CREAM:
            return super()._play_score(card)
        active = self.active()
        if active is None or _energy_count(active) < 3:
            return -5000.0
        missing = max(0, active.maxHp - active.hp)
        healed = min(80, missing)
        if healed <= 0:
            return -5000.0
        incoming = self._opponent_best_next_damage(active)
        if incoming <= 0:
            return -5000.0
        attacks_before = max(1, math.ceil(active.hp / incoming))
        attacks_after = max(1, math.ceil((active.hp + healed) / incoming))
        if attacks_after <= attacks_before:
            return -5000.0
        prevents_immediate_ko = active.hp <= incoming < active.hp + healed
        return (
            340000.0
            if prevents_immediate_ko
            else 150000.0 + (attacks_after - attacks_before) * 50000.0
        )


class HealingBreakpointSequencedArchaludonPolicy(
    HealingBreakpointMixin, SequencedArchaludonPolicy
):
    pass


class RiskAwareRetreatMixin:
    """Choose an escape target by survival, readiness, and prize liability."""

    _opponent_attack_damage = TacticalEscapeMixin._opponent_attack_damage
    _opponent_best_next_damage = TacticalEscapeMixin._opponent_best_next_damage
    _under_next_attack_ko = TacticalEscapeMixin._under_next_attack_ko
    _has_immediate_ko = TacticalEscapeMixin._has_immediate_ko

    def _retreat_target_score(self, pokemon):
        if pokemon is None:
            return -1e9
        incoming = self._opponent_best_next_damage(pokemon)
        survival_margin = pokemon.hp - incoming
        readiness = self._ready_score(pokemon)
        prize_liability = _prizes(pokemon.id) * 45000.0
        attack_ready = 0.0
        data = _CARDS.get(pokemon.id)
        for attack_id in getattr(data, "attacks", None) or []:
            if _can_pay_after_one_attachment(pokemon, _ATTACKS.get(attack_id)):
                attack_ready = 60000.0
                break
        survival = (
            350000.0 + survival_margin * 250.0
            if survival_margin > 0
            else survival_margin * 600.0
        )
        return survival + readiness + attack_ready - prize_liability

    def _main_score(self, option):
        score = super()._main_score(option)
        if option.type != OptionType.RETREAT:
            return score
        if self._has_immediate_ko():
            return score
        active = self.active()
        if active is None or not self.mine.bench:
            return score
        active_value = self._retreat_target_score(active)
        best_replacement = max(
            (self._retreat_target_score(pokemon) for pokemon in self.mine.bench),
            default=-1e9,
        )
        threatened = self._under_next_attack_ko(active)
        improvement = best_replacement - active_value
        if threatened and best_replacement > -1e8:
            return 650000.0 + max(0.0, improvement)
        if improvement > 90000.0:
            return 75000.0 + improvement
        return score

    def _selection_score(self, option):
        score = super()._selection_score(option)
        if self.context not in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            return score
        card = _card(
            self.obs,
            option.area,
            option.index,
            getattr(option, "playerIndex", self.me),
        )
        if card is None:
            return score
        return self._retreat_target_score(card)


class RiskAwareRetreatSequencedArchaludonPolicy(
    RiskAwareRetreatMixin, SequencedArchaludonPolicy
):
    pass


class HandQualityMixin:
    """Value draw Supporters from visible playability and next-turn continuity."""

    _search_cards = {ULTRA_BALL, POKEGEAR, POKE_PAD, NIGHT_STRETCHER}
    _supporters = {BOSS, EXPLORER, JUDGE, LILLIE}

    def _visible_hand_quality(self, excluded_card_id=None):
        score = 0.0
        counts = Counter(self.hand)
        if excluded_card_id is not None and counts[excluded_card_id]:
            counts[excluded_card_id] -= 1
        active = self.active()
        has_line = bool(self.field[DURALUDON] or counts[DURALUDON])
        has_evolution = bool(counts[ARCHALUDON_EX])
        has_energy = bool(counts[METAL_ENERGY])
        if has_line:
            score += 2.0
        if has_line and has_evolution:
            score += 3.0
        if has_energy:
            score += 2.0
        if active is not None and _energy_count(active) >= 2:
            score += 1.0
        score += min(2.0, sum(counts[card_id] for card_id in self._search_cards))
        score += min(1.5, sum(counts[card_id] for card_id in self._supporters))
        useful_unique = sum(
            bool(counts[card_id])
            for card_id in (
                DURALUDON,
                ARCHALUDON_EX,
                METAL_ENERGY,
                ULTRA_BALL,
                POKE_PAD,
                NIGHT_STRETCHER,
            )
        )
        score += useful_unique * 0.5
        dead_duplicates = sum(max(0, count - 2) for count in counts.values())
        return score - dead_duplicates * 0.75

    def _play_score(self, card):
        if card is None or card.id not in self._supporters:
            return super()._play_score(card)
        base = super()._play_score(card)
        current_quality = self._visible_hand_quality(excluded_card_id=card.id)
        hand_after_play = max(0, self.mine.handCount - 1)
        if card.id == LILLIE:
            expected_draws = max(0, 6 - hand_after_play)
            improvement = expected_draws * 1.2 - current_quality * 0.22
            return base + improvement * 12000.0
        if card.id == JUDGE:
            own_gain = 4 - hand_after_play
            opponent_loss = self.theirs.handCount - 4
            improvement = own_gain * 1.1 + opponent_loss * 0.8 - current_quality * 0.3
            return base + improvement * 15000.0
        if card.id == EXPLORER:
            continuity_gap = max(0.0, 7.0 - current_quality)
            return base + continuity_gap * 11000.0
        if card.id == BOSS and current_quality < 2.0:
            # Preserve the once-per-turn Supporter action when the hand cannot
            # sustain the following turn, unless Boss already creates a prize.
            return base - 45000.0
        return base


class HandQualitySequencedArchaludonPolicy(
    HandQualityMixin, SequencedArchaludonPolicy
):
    pass


def _agent_with_policy(obs_dict, deck, policy_class):
    deck = list(deck or ARCHALUDON_DECK)
    try:
        if obs_dict.get("select") is None:
            return deck
        obs = to_observation_class(obs_dict)
        if obs.select is None or not obs.select.option:
            return []
        override = hard_override(obs)
        if override is not None:
            return [override]
        action = policy_class(obs).choose()
        minimum = max(0, obs.select.minCount or 0)
        maximum = min(len(obs.select.option), obs.select.maxCount or 1)
        if len(action) < minimum:
            used = set(action)
            remaining = sorted(
                (
                    i
                    for i in range(len(obs.select.option))
                    if i not in used
                ),
                key=lambda i: _option_semantic_key(obs.select.option[i]),
                reverse=True,
            )
            action += remaining[: minimum - len(action)]
        return action[:maximum]
    except Exception:
        try:
            select = obs_dict.get("select")
            if select is None:
                return deck
            options = select.get("option") or []
            minimum = select.get("minCount", select.get("maxCount", 1)) or 0
            ranked = sorted(
                range(len(options)),
                key=lambda i: _option_semantic_key(options[i]),
                reverse=True,
            )
            return ranked[: min(minimum, len(options))]
        except Exception:
            return deck


def archaludon_agent(obs_dict, deck=None):
    return _agent_with_policy(obs_dict, deck, ArchaludonPolicy)


def archaludon_sequenced_agent(obs_dict, deck=None):
    return _agent_with_policy(obs_dict, deck, SequencedArchaludonPolicy)


def archaludon_boss_gated_agent(obs_dict, deck=None):
    return _agent_with_policy(obs_dict, deck, BossGatedArchaludonPolicy)


def archaludon_seq_boss_agent(obs_dict, deck=None):
    return _agent_with_policy(obs_dict, deck, SequencedBossGatedArchaludonPolicy)


def archaludon_prize_race_agent(obs_dict, deck=None):
    return _agent_with_policy(obs_dict, deck, PrizeRaceSequencedArchaludonPolicy)


def archaludon_tactical_agent(obs_dict, deck=None):
    return _agent_with_policy(obs_dict, deck, TacticalSequencedArchaludonPolicy)


def archaludon_supporter_timed_agent(obs_dict, deck=None):
    return _agent_with_policy(obs_dict, deck, SupporterTimedSequencedArchaludonPolicy)


def archaludon_macro_agent(obs_dict, deck=None):
    return _agent_with_policy(obs_dict, deck, MacroSequencedArchaludonPolicy)


def archaludon_energy_opportunity_agent(obs_dict, deck=None):
    return _agent_with_policy(obs_dict, deck, EnergyOpportunitySequencedArchaludonPolicy)


def archaludon_recovery_inventory_agent(obs_dict, deck=None):
    return _agent_with_policy(
        obs_dict,
        deck,
        RecoveryInventorySequencedArchaludonPolicy,
    )


def archaludon_end_turn_regret_agent(obs_dict, deck=None):
    return _agent_with_policy(
        obs_dict,
        deck,
        EndTurnRegretSequencedArchaludonPolicy,
    )


def archaludon_bench_capacity_agent(obs_dict, deck=None):
    return _agent_with_policy(
        obs_dict,
        deck,
        BenchCapacitySequencedArchaludonPolicy,
    )


def archaludon_evolution_stack_agent(obs_dict, deck=None):
    return _agent_with_policy(
        obs_dict,
        deck,
        EvolutionStackSequencedArchaludonPolicy,
    )


def archaludon_damage_breakpoint_agent(obs_dict, deck=None):
    return _agent_with_policy(
        obs_dict,
        deck,
        DamageBreakpointSequencedArchaludonPolicy,
    )


def archaludon_stadium_timing_agent(obs_dict, deck=None):
    return _agent_with_policy(
        obs_dict,
        deck,
        StadiumTimingSequencedArchaludonPolicy,
    )


def archaludon_healing_breakpoint_agent(obs_dict, deck=None):
    return _agent_with_policy(
        obs_dict,
        deck,
        HealingBreakpointSequencedArchaludonPolicy,
    )


def archaludon_risk_retreat_agent(obs_dict, deck=None):
    return _agent_with_policy(
        obs_dict,
        deck,
        RiskAwareRetreatSequencedArchaludonPolicy,
    )


def archaludon_hand_quality_agent(obs_dict, deck=None):
    return _agent_with_policy(
        obs_dict,
        deck,
        HandQualitySequencedArchaludonPolicy,
    )


# Search adapter used by hybrid_agent. These functions expose only our own
# clean-room policy and evaluation; no public benchmark implementation enters
# the candidate path.
def rank_options(obs):
    try:
        policy = SequencedArchaludonPolicy(obs)
        if policy.context == SelectContext.MAIN:
            scores = [policy._main_score(option) for option in policy.select.option]
        else:
            scores = [policy._selection_score(option) for option in policy.select.option]
        return sorted(
            range(len(scores)),
            key=lambda index: (
                scores[index],
                _option_semantic_key(policy.select.option[index]),
            ),
            reverse=True,
        )
    except Exception:
        return list(range(len(obs.select.option))) if obs.select else []


def hard_override(obs):
    try:
        if obs.select is None or obs.select.context != SelectContext.MAIN:
            return None
        policy = SequencedArchaludonPolicy(obs)
        target = policy.opponent_active()
        if target is None:
            return None
        lethal = []
        for index, option in enumerate(obs.select.option):
            if option.type != OptionType.ATTACK:
                continue
            damage = _damage(getattr(option, "attackId", None), policy.active(), target)
            if damage >= target.hp:
                lethal.append(
                    (damage, _option_semantic_key(option), index)
                )
        return max(lethal)[2] if lethal else None
    except Exception:
        return None


def evaluate_board(cur, me):
    if cur.result >= 0:
        return 1e9 if cur.result == me else (-1e9 if cur.result == (1 - me) else 0.0)
    mine, theirs = cur.players[me], cur.players[1 - me]
    value = (len(theirs.prize) - len(mine.prize)) * 100000.0

    def board(player):
        return [pokemon for pokemon in list(player.active) + list(player.bench) if pokemon]

    for pokemon in board(mine):
        energy = _energy_count(pokemon)
        value += pokemon.hp * 1.5 + energy * 800.0
        if pokemon.id == ARCHALUDON_EX:
            value += 5000.0 + (7000.0 if energy >= 3 else 0.0)
        elif pokemon.id == DURALUDON:
            value += 1200.0 + (2500.0 if energy >= 3 else 0.0)
    for pokemon in board(theirs):
        value -= pokemon.hp * 0.6 + _prizes(pokemon.id) * 700.0

    active = mine.active[0] if mine.active else None
    target = theirs.active[0] if theirs.active else None
    if active is not None and target is not None:
        affordable = []
        energy = _energy_count(active)
        if active.id == CINDERACE and energy >= 1:
            affordable.append(TURBO_FLARE)
        elif active.id == DURALUDON:
            if energy >= 1:
                affordable.append(HAMMER_IN)
            if energy >= 3:
                affordable.append(RAGING_HAMMER)
        elif active.id == ARCHALUDON_EX and energy >= 3:
            affordable.append(METAL_DEFENDER)
        if any(_damage(attack_id, active, target) >= target.hp for attack_id in affordable):
            value += _prizes(target.id) * 12000.0
    return value


def domain_agent(obs_dict, deck):
    return archaludon_sequenced_agent(obs_dict, deck)


if __name__ == "__main__":
    assert len(ARCHALUDON_DECK) == 60
    print(f"Clean-room Archaludon deck: {len(ARCHALUDON_DECK)} cards")
    print(f"Deck-select: {len(archaludon_agent({'select': None}))} cards")
