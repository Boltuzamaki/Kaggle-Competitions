"""Deck-aware Kaggle agent for Pokémon TCG AI Battle.

The agent chooses one of three streamlined decks at game start and then uses a
strategy-aware, aggressive policy rather than random legal actions.
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from typing import Iterable

from cg.api import (
    AreaType,
    Card,
    CardData,
    CardType,
    EnergyType,
    Observation,
    Option,
    OptionType,
    Pokemon,
    SelectContext,
    SelectType,
    all_attack,
    all_card_data,
    to_observation_class,
)

# ---------------------------------------------------------------------------
# Streamlined 60-card decks designed for a rule-based AI.
# ---------------------------------------------------------------------------

DECK_ZOROARK_NAME = "Aggro N's Zoroark ex"
DECK_ZOROARK = [
    # Pokémon (14)
    292, 292, 292, 292,              # N's Zorua
    293, 293, 293, 293,              # N's Zoroark ex
    906, 906, 906,                   # N's Zekrom (Night Joker donor)
    141, 141,                        # Pecharunt ex
    140,                             # Fezandipiti ex
    # Trainers (36)
    1086, 1086, 1086, 1086,         # Buddy-Buddy Poffin
    1121, 1121, 1121, 1121,         # Ultra Ball
    1195, 1195, 1195, 1195,         # Janine's Secret Art
    1113, 1113, 1113, 1113,         # N's PP Up
    1227, 1227, 1227, 1227,         # Lillie's Determination
    1182, 1182, 1182,                # Boss's Orders
    1162, 1162, 1162,                # Binding Mochi
    1253, 1253, 1253,                # N's Castle
    1097, 1097,                      # Night Stretcher
    1122, 1122,                      # Pokégear 3.0
    1092,                            # Secret Box (ACE SPEC)
    1211,                            # Black Belt's Training
    1118,                            # Energy Retrieval
    # Energy (10)
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
]

DECK_MEWTWO_NAME = "Pure Team Rocket Mewtwo ex"
DECK_MEWTWO = [
    # Pokémon (18) — every Pokémon is a Team Rocket Pokémon.
    431, 431, 431, 431,              # Team Rocket's Mewtwo ex
    400, 400, 400, 400,              # Team Rocket's Tarountula
    401, 401, 401, 401,              # Team Rocket's Spidops
    24, 24, 24,                      # Team Rocket's Kangaskhan ex
    433, 433,                        # Team Rocket's Chingling
    463,                             # Team Rocket's Murkrow
    # Trainers (30)
    1220, 1220, 1220, 1220,         # Team Rocket's Proton
    1216, 1216, 1216, 1216,         # Team Rocket's Ariana
    1218, 1218, 1218,                # Team Rocket's Giovanni
    1134, 1134, 1134, 1134,         # Team Rocket's Transceiver
    1121, 1121, 1121, 1121,         # Ultra Ball
    1094, 1094,                      # Bug Catching Set (edge v1: one slot to Energy)
    1116, 1116,                      # Energy Switch
    1097, 1097,                      # Night Stretcher
    1158,                            # Maximum Belt (ACE SPEC)
    1257, 1257, 1257,                # Team Rocket's Factory
    # Energy (12)
    1, 1, 1, 1, 1, 1, 1, 1, 1,     # Basic Grass Energy
    15, 15, 15, 15,                 # Team Rocket's Energy
]

DECK_VENUSAUR_NAME = "Linear Mega Venusaur ex"
DECK_VENUSAUR = [
    # Pokémon (20)
    96, 96, 96, 96,                 # Teal Mask Ogerpon ex
    650, 650, 650, 650,             # Bulbasaur
    651, 651,                        # Ivysaur
    652, 652, 652,                   # Mega Venusaur ex
    917, 917, 917,                   # Chikorita
    709, 709,                        # Bayleef
    710, 710,                        # Meganium
    # Trainers (28)
    1094, 1094, 1094, 1094,         # Bug Catching Set
    1121, 1121, 1121, 1121,         # Ultra Ball
    1079, 1079, 1079,                # Rare Candy
    1261, 1261, 1261, 1261,         # Forest of Vitality
    1227, 1227, 1227, 1227,         # Lillie's Determination
    1225, 1225, 1225,                # Hilda
    1182, 1182,                      # Boss's Orders
    1080,                            # Unfair Stamp (ACE SPEC)
    1229,                            # Wally's Compassion
    1097, 1097,                      # Night Stretcher
    # Energy (12)
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
]

DECKS = {
    "zoroark": (DECK_ZOROARK_NAME, tuple(DECK_ZOROARK)),
    "mewtwo": (DECK_MEWTWO_NAME, tuple(DECK_MEWTWO)),
    "venusaur": (DECK_VENUSAUR_NAME, tuple(DECK_VENUSAUR)),
}
DECK_ORDER = ("zoroark", "mewtwo", "venusaur")

for _key, (_name, _deck) in DECKS.items():
    if len(_deck) != 60:
        raise RuntimeError(f"{_name} has {len(_deck)} cards, not 60")

_CARD: dict[int, CardData] = {c.cardId: c for c in all_card_data()}
_ATTACK = {a.attackId: a for a in all_attack()}

TEAM_ROCKET_SUPPORTERS = {1216, 1217, 1218, 1219, 1220}
TEAM_ROCKET_POKEMON = {
    cid for cid, data in _CARD.items()
    if data.cardType == CardType.POKEMON and data.name.startswith("Team Rocket")
}
N_POKEMON = {292, 293, 906}


@dataclass
class _Choice:
    index: int
    score: float


class BattlePolicy:
    """Stateful policy. Kaggle uses one instance through the module-level agent."""

    def __init__(
        self,
        fixed_deck: str | None = None,
        seed: int | None = None,
        announce: bool = False,
    ) -> None:
        self.fixed_deck = fixed_deck
        self.rng = random.Random(seed) if seed is not None else random.SystemRandom()
        self.announce = announce
        self.deck_key = fixed_deck or "zoroark"
        self.deck = list(DECKS[self.deck_key][1])
        self.last_turn = -1
        self.supporter_this_turn: int | None = None
        self.scored_decisions = 0
        self.tie_decisions = 0
        self.tie_widths: dict[int, int] = {}
        self.tie_contexts: dict[str, int] = {}
        self.meaningful_tie_decisions = 0
        self.meaningful_tie_contexts: dict[str, int] = {}

    # ------------------------------ lifecycle ------------------------------

    def _select_deck(self) -> list[int]:
        forced = os.environ.get("PTCG_DECK", "").strip().lower()
        aliases = {
            "1": "zoroark", "zoroark": "zoroark", "z": "zoroark",
            "2": "mewtwo", "mewtwo": "mewtwo", "m": "mewtwo",
            "3": "venusaur", "venusaur": "venusaur", "v": "venusaur",
        }
        if self.fixed_deck in DECKS:
            key = self.fixed_deck
        elif forced in aliases:
            key = aliases[forced]
        else:
            # Controlled arena testing favored Mewtwo across the reconstructed
            # metagame. Avoid giving one third of evaluation games to a weaker
            # random deck while retaining PTCG_DECK overrides for experiments.
            key = "mewtwo"

        self.deck_key = key
        self.deck = list(DECKS[key][1])
        self.last_turn = -1
        self.supporter_this_turn = None
        self.scored_decisions = 0
        self.tie_decisions = 0
        self.tie_widths = {}
        self.tie_contexts = {}
        self.meaningful_tie_decisions = 0
        self.meaningful_tie_contexts = {}
        if self.announce and os.environ.get("PTCG_QUIET", "0") != "1":
            print(f"[PTCG] selected deck: {DECKS[key][0]}", file=sys.stderr, flush=True)
        return list(self.deck)

    def act(self, obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return self._select_deck()
        if not obs.select.option:
            return []

        self._ingest(obs)

        if obs.select.type == SelectType.MAIN:
            return [self._choose_main(obs)]
        if obs.select.type == SelectType.YES_NO:
            return [self._choose_yes_no(obs)]
        if obs.select.type == SelectType.ATTACK:
            return [self._choose_attack_option(obs)]
        if obs.select.type == SelectType.COUNT:
            return [self._choose_count_option(obs)]

        return self._choose_multi(obs)

    def _ingest(self, obs: Observation) -> None:
        if obs.current is not None and obs.current.turn != self.last_turn:
            self.last_turn = obs.current.turn
            self.supporter_this_turn = None
        for log in obs.logs:
            # PLAY == 10 in the public API. Avoid importing LogType just for this.
            if int(log.type) == 10 and log.cardId in _CARD:
                if _CARD[log.cardId].cardType == CardType.SUPPORTER:
                    self.supporter_this_turn = log.cardId

    # ---------------------------- state helpers ----------------------------

    @staticmethod
    def _me(obs: Observation):
        assert obs.current is not None
        return obs.current.players[obs.current.yourIndex]

    @staticmethod
    def _opp(obs: Observation):
        assert obs.current is not None
        return obs.current.players[1 - obs.current.yourIndex]

    @staticmethod
    def _all_in_play(player) -> list[Pokemon]:
        out: list[Pokemon] = []
        if player.active and player.active[0] is not None:
            out.append(player.active[0])
        out.extend(player.bench)
        return out

    def _count_in_play(self, obs: Observation, ids: set[int] | None = None) -> int:
        pokes = self._all_in_play(self._me(obs))
        if ids is None:
            return len(pokes)
        return sum(p.id in ids for p in pokes)

    def _count_id_in_play(self, obs: Observation, card_id: int) -> int:
        return sum(p.id == card_id for p in self._all_in_play(self._me(obs)))

    def _hand_ids(self, obs: Observation) -> list[int]:
        hand = self._me(obs).hand or []
        return [c.id for c in hand]

    def _discard_ids(self, obs: Observation) -> list[int]:
        return [c.id for c in self._me(obs).discard]

    def _active(self, obs: Observation) -> Pokemon | None:
        active = self._me(obs).active
        return active[0] if active and active[0] is not None else None

    def _opponent_active(self, obs: Observation) -> Pokemon | None:
        active = self._opp(obs).active
        return active[0] if active and active[0] is not None else None

    @staticmethod
    def _energy_units(pokemon: Pokemon | None) -> int:
        return len(pokemon.energies) if pokemon is not None else 0

    @staticmethod
    def _has_tool(pokemon: Pokemon | None, card_id: int) -> bool:
        return bool(pokemon and any(c.id == card_id for c in pokemon.tools))

    def _pokemon_from_area(
        self,
        obs: Observation,
        area: AreaType | None,
        index: int | None,
        player_index: int | None = None,
    ) -> Pokemon | None:
        if obs.current is None or area is None or index is None:
            return None
        pi = obs.current.yourIndex if player_index is None else player_index
        player = obs.current.players[pi]
        if area == AreaType.ACTIVE:
            return player.active[index] if 0 <= index < len(player.active) else None
        if area == AreaType.BENCH:
            return player.bench[index] if 0 <= index < len(player.bench) else None
        return None

    def _card_from_area(
        self,
        obs: Observation,
        area: AreaType | None,
        index: int | None,
        player_index: int | None = None,
    ) -> Card | Pokemon | None:
        if obs.current is None or area is None or index is None:
            return None
        pi = obs.current.yourIndex if player_index is None else player_index
        player = obs.current.players[pi]
        seq = None
        if area == AreaType.DECK:
            seq = obs.select.deck if obs.select is not None else None
        elif area == AreaType.HAND:
            seq = player.hand
        elif area == AreaType.DISCARD:
            seq = player.discard
        elif area == AreaType.ACTIVE:
            seq = player.active
        elif area == AreaType.BENCH:
            seq = player.bench
        elif area == AreaType.PRIZE:
            seq = player.prize
        elif area == AreaType.STADIUM:
            seq = obs.current.stadium
        elif area == AreaType.LOOKING:
            seq = obs.current.looking
        if seq is None or not (0 <= index < len(seq)):
            return None
        return seq[index]

    def _option_card_id(self, obs: Observation, option: Option) -> int | None:
        if obs.current is None:
            return None
        me = self._me(obs)
        if option.type in (OptionType.PLAY, OptionType.ATTACH, OptionType.EVOLVE):
            if me.hand is not None and option.index is not None and 0 <= option.index < len(me.hand):
                return me.hand[option.index].id
            return None
        obj = self._card_from_area(obs, option.area, option.index, option.playerIndex)
        return obj.id if obj is not None else None

    def _option_target_pokemon(self, obs: Observation, option: Option) -> Pokemon | None:
        if option.type in (OptionType.ATTACH, OptionType.EVOLVE):
            return self._pokemon_from_area(
                obs, option.inPlayArea, option.inPlayIndex,
                obs.current.yourIndex if obs.current is not None else None,
            )
        return self._pokemon_from_area(obs, option.area, option.index, option.playerIndex)

    def _effect_id(self, obs: Observation) -> int | None:
        if obs.select is None:
            return None
        if obs.select.effect is not None:
            return obs.select.effect.id
        if obs.select.contextCard is not None:
            return obs.select.contextCard.id
        return None

    # ----------------------------- attack logic ----------------------------

    def _attack_damage(self, obs: Observation, attack_id: int) -> int:
        attack = _ATTACK.get(attack_id)
        if attack is None:
            return 0
        damage = int(attack.damage)
        active = self._active(obs)
        opponent = self._opponent_active(obs)

        if attack_id == 560:  # Rocket Rush
            damage = 30 * self._count_in_play(obs, TEAM_ROCKET_POKEMON)
        elif attack_id == 608:  # Erasure Ball
            benched_energy_cards = sum(len(p.energyCards) for p in self._me(obs).bench)
            damage = 160 + 60 * min(2, benched_energy_cards)
        elif attack_id == 7:  # Wicked Impact
            damage = 220 if self.supporter_this_turn in TEAM_ROCKET_SUPPORTERS else 120
        elif attack_id == 403:  # Night Joker
            bench_ids = {p.id for p in self._me(obs).bench}
            if 906 in bench_ids:
                damage = 250
            elif 293 in bench_ids:
                damage = 0
            else:
                donor_damage = []
                for p in self._me(obs).bench:
                    if p.id in N_POKEMON:
                        donor_damage.extend(_ATTACK[aid].damage for aid in _CARD[p.id].attacks)
                damage = max(donor_damage, default=20)
        elif attack_id == 120:  # Myriad Leaf Shower
            own = self._energy_units(active)
            other = self._energy_units(opponent)
            damage = 30 + 30 * (own + other)
        elif attack_id == 355:  # Back Draft
            basic_energy = sum(
                _CARD.get(c.id) is not None and _CARD[c.id].cardType == CardType.BASIC_ENERGY
                for c in self._opp(obs).discard
            )
            damage = 30 * basic_energy
        elif attack_id == 184:  # Irritated Outburst
            taken = 6 - len(self._opp(obs).prize)
            damage = 60 * max(0, taken)

        if self._has_tool(active, 1162) and self._me(obs).poisoned:
            damage += 40
        if self._has_tool(active, 1158) and opponent is not None:
            if _CARD[opponent.id].ex or _CARD[opponent.id].megaEx:
                damage += 50
        if self.supporter_this_turn == 1211 and opponent is not None:
            if _CARD[opponent.id].ex or _CARD[opponent.id].megaEx:
                damage += 40
        return max(0, damage)

    def _attack_score(self, obs: Observation, attack_id: int) -> float:
        attack = _ATTACK.get(attack_id)
        if attack is None:
            return -1000.0
        damage = self._attack_damage(obs, attack_id)
        score = 1050.0 + 3.0 * damage

        # Useful zero/small-damage attacks.
        utility = {
            611: 180,   # Chiming Commotion: random hand discard, no Energy
            652: 120,   # Murkrow Deceit: search a Supporter
            945: 180,   # Celebi Traverse Time
            323: 220,   # Budew item lock
            939: 60,    # Bind Down
            1322: 50,   # Growl
        }
        score += utility.get(attack_id, 0)

        opponent = self._opponent_active(obs)
        if opponent is not None and damage >= opponent.hp and damage > 0:
            prizes = 3 if _CARD[opponent.id].megaEx else 2 if _CARD[opponent.id].ex else 1
            score += 3200 + 450 * prizes
        if obs.current is not None and obs.current.turnActionCount >= 5:
            score += 450
        if obs.current is not None and obs.current.turnActionCount >= 9:
            score += 900
        return score

    def _choose_attack_option(self, obs: Observation) -> int:
        assert obs.select is not None
        choices = [
            _Choice(i, self._attack_score(obs, int(option.attackId or -1)))
            for i, option in enumerate(obs.select.option)
        ]
        self._record_meaningful_tie(obs, choices, "attack")
        return self._argmax(choices, "attack")

    # --------------------------- main phase logic --------------------------

    def _choose_main(self, obs: Observation) -> int:
        assert obs.select is not None

        # Hard loop guard for repeatable Abilities such as Solar Transfer.
        # After enough setup actions, attack immediately; if no attack exists,
        # end the turn rather than cycling cards forever.
        if obs.current is not None and obs.current.turnActionCount >= 12:
            attacks = [
                _Choice(i, self._attack_score(obs, int(o.attackId or -1)))
                for i, o in enumerate(obs.select.option)
                if o.type == OptionType.ATTACK
            ]
            if attacks:
                return self._argmax(attacks, "loop_guard_attack")
            ends = [i for i, o in enumerate(obs.select.option) if o.type == OptionType.END]
            if ends:
                return ends[0]

        choices = [
            _Choice(i, self._main_score(obs, option))
            for i, option in enumerate(obs.select.option)
        ]
        self._record_meaningful_tie(obs, choices, "main")
        return self._argmax(choices, "main")

    def _main_score(self, obs: Observation, option: Option) -> float:
        if option.type == OptionType.ATTACK:
            return self._attack_score(obs, int(option.attackId or -1))
        if option.type == OptionType.EVOLVE:
            return self._evolve_score(obs, option)
        if option.type == OptionType.ATTACH:
            return self._attach_score(obs, option)
        if option.type == OptionType.ABILITY:
            return self._ability_score(obs, option)
        if option.type == OptionType.PLAY:
            return self._play_score(obs, option)
        if option.type == OptionType.RETREAT:
            return self._retreat_score(obs)
        if option.type == OptionType.END:
            return -50.0
        if option.type == OptionType.DISCARD:
            return 80.0
        return 0.0

    def _evolve_score(self, obs: Observation, option: Option) -> float:
        cid = self._option_card_id(obs, option)
        target = self._option_target_pokemon(obs, option)
        active = self._active(obs)
        score = 1300.0
        if cid is None:
            return score
        if self.deck_key == "zoroark":
            if cid == 293:
                score = 2150 if target is not None and active is not None and target.serial == active.serial else 1900
        elif self.deck_key == "mewtwo":
            if cid == 401:
                score = 1850
        elif self.deck_key == "venusaur":
            if cid == 652:
                score = 2200
            elif cid == 710:
                score = 2050
            elif cid in {651, 709}:
                score = 1750
        return score

    def _attach_score(self, obs: Observation, option: Option) -> float:
        cid = self._option_card_id(obs, option)
        target = self._option_target_pokemon(obs, option)
        if cid is None or target is None:
            return -500.0
        data = _CARD[cid]
        is_active = bool(self._active(obs) and target.serial == self._active(obs).serial)
        units = self._energy_units(target)
        backup_energy_bonus = 0.0
        if self.deck_key == "zoroark" and not is_active and self._me(obs).bench:
            active = self._active(obs)
            if active is not None:
                active_cost = min(
                    (len(_ATTACK[a].energies) for a in _CARD[active.id].attacks if a in _ATTACK),
                    default=0,
                )
                if self._energy_units(active) >= active_cost:
                    backup_energy_bonus = 700.0

        if data.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            if self.deck_key == "zoroark":
                if target.id in {292, 293}:
                    return 2200 - 220 * units + (180 if is_active else 0) + backup_energy_bonus
                if target.id in {141, 140}:
                    return 650 - 100 * units + backup_energy_bonus
                if target.id == 906:
                    return -800.0 + backup_energy_bonus  # Zekrom is mainly a Night Joker donor.
            elif self.deck_key == "mewtwo":
                if target.id == 431:
                    bonus = 350 if cid == 15 and units < 2 else 0
                    return 2150 + bonus - 170 * units + (180 if is_active else 0)
                if target.id == 24:
                    return 1850 - 160 * units + (140 if is_active else 0)
                if target.id == 401:
                    return 1700 - 150 * units + (120 if is_active else 0)
                if target.id == 400:
                    return 850 - 100 * units
                return 300.0
            elif self.deck_key == "venusaur":
                if target.id == 652:
                    return 2250 - 150 * units + (220 if is_active else 0)
                if target.id == 96:
                    return 1950 - 150 * units + (180 if is_active else 0)
                if target.id == 710:
                    return 1700 - 140 * units + (150 if is_active else 0)
                return 700 - 100 * units

        if data.cardType == CardType.TOOL:
            opponent = self._opponent_active(obs)
            if cid == 1162 and target.id == 293:
                return 1900 + (250 if is_active else 0)
            if cid == 1158 and target.id in {431, 24}:
                ex_bonus = 250 if opponent and (_CARD[opponent.id].ex or _CARD[opponent.id].megaEx) else 0
                return 1750 + ex_bonus + (150 if is_active else 0)
            return 300.0
        return 200.0

    def _ability_score(self, obs: Observation, option: Option) -> float:
        cid = self._option_card_id(obs, option)
        if cid is None:
            return 200.0
        hand_count = self._me(obs).handCount
        if cid == 293:  # Trade
            return 1850 if hand_count >= 3 else 650
        if cid == 141:  # Subjugating Chains
            return 2200 if self._best_ready_bench(obs, {293}) is not None else 400
        if cid == 140:  # Flip the Script
            return 2000
        if cid == 401:  # Charging Up
            return 1900
        if cid == 96:   # Teal Dance
            return 2050 if 1 in self._hand_ids(obs) else 500
        if cid == 652:  # Solar Transfer
            active = self._active(obs)
            if active and active.id in {652, 710}:
                need = 4 - self._energy_units(active)
                bench_energy = sum(self._energy_units(p) for p in self._me(obs).bench)
                return 2100 if need > 0 and bench_energy > 0 else -200
            return -250
        if cid == 1257:  # Team Rocket's Factory
            return 1800
        return 650.0

    def _play_score(self, obs: Observation, option: Option) -> float:
        cid = self._option_card_id(obs, option)
        if cid is None:
            return 100.0
        data = _CARD[cid]
        hand_count = self._me(obs).handCount
        attack_available = any(
            o.type == OptionType.ATTACK for o in (obs.select.option if obs.select else [])
        )

        if data.cardType == CardType.POKEMON:
            return self._bench_card_score(obs, cid, from_hand=True)

        if self.deck_key == "zoroark":
            scores = {
                1086: 1750 if self._count_in_play(obs, {292, 293}) < 2 else 700,
                1121: 1650 if self._missing_zoroark_piece(obs) else 750,
                1195: 2250 if self._janine_is_live(obs) else 550,
                1113: 1800 if 7 in self._discard_ids(obs) else 350,
                1227: 1700 if hand_count <= 4 else 500,
                1182: 1850 if attack_available else 250,
                1253: 1300 if not obs.current.stadium else 350,
                1097: 1150 if any(x in self._discard_ids(obs) for x in {292, 293, 906, 7}) else 250,
                1122: 900 if hand_count <= 4 else 350,
                1092: 1550 if hand_count >= 6 else 200,
                1211: 1900 if attack_available and self._opponent_is_ex(obs) else 250,
                1118: 1100 if self._discard_ids(obs).count(7) >= 2 else 250,
                1213: 850 if hand_count <= 3 else 200,
            }
            score = scores.get(cid, 350.0)
            if not self._me(obs).bench:
                score += {
                    1086: 2000.0, 1121: 1050.0, 1097: 850.0,
                }.get(cid, 0.0)
            return score

        if self.deck_key == "mewtwo":
            tr_count = self._count_in_play(obs, TEAM_ROCKET_POKEMON)
            scores = {
                1220: 2200 if tr_count < 4 else 600,
                1216: 1850 if hand_count <= 5 else 650,
                1218: 1900 if attack_available else 350,
                1134: 1750 if tr_count < 4 or hand_count <= 4 else 750,
                1121: 1700 if self._missing_mewtwo_piece(obs) else 700,
                1094: 1550 if tr_count < 5 else 600,
                1116: 1500 if self._active_needs_energy(obs) else 300,
                1097: 1100 if any(x in self._discard_ids(obs) for x in {431, 400, 401, 1}) else 250,
                1257: 1450 if not obs.current.stadium else 350,
            }
            score = scores.get(cid, 350.0)
            if not self._me(obs).bench:
                score += {
                    1220: 600.0, 1094: 500.0, 1121: 400.0,
                    1134: 300.0, 1097: 250.0,
                }.get(cid, 0.0)
            return score

        # Venusaur
        scores = {
            1094: 1650,
            1121: 1700 if self._missing_venusaur_piece(obs) else 700,
            1079: 2200 if self._rare_candy_live(obs) else 250,
            1261: 1850 if not obs.current.stadium else 300,
            1227: 1700 if hand_count <= 4 else 500,
            1225: 1800,
            1182: 1850 if attack_available else 250,
            1080: 2000,  # If offered, its use condition is already satisfied.
            1229: 1900 if self._damaged_mega(obs) else 250,
            1097: 1050 if any(x in self._discard_ids(obs) for x in {650, 651, 652, 917, 709, 710, 1}) else 250,
        }
        return scores.get(cid, 350.0)

    def _retreat_score(self, obs: Observation) -> float:
        active = self._active(obs)
        if active is None:
            return -500
        ready = self._best_ready_bench(obs)
        if ready is not None:
            active_value = self._ready_value(obs, active)
            ready_value = self._ready_value(obs, ready)
            if ready_value > active_value + 150:
                return 2050 + ready_value - active_value
        preferred = self._active_priority(active.id)
        best_bench = max((self._active_priority(p.id) for p in self._me(obs).bench), default=-1000)
        if best_bench > preferred + 50:
            return 1200 + best_bench - preferred
        return 100.0

    # -------------------------- card/board priorities ----------------------

    def _active_priority(self, cid: int) -> float:
        if self.deck_key == "zoroark":
            return {293: 220, 292: 150, 141: 70, 140: 50, 906: 10}.get(cid, 0)
        if self.deck_key == "mewtwo":
            return {431: 230, 24: 200, 401: 170, 433: 100, 463: 80, 400: 70}.get(cid, 0)
        return {652: 240, 96: 210, 710: 180, 651: 80, 709: 70, 650: 60, 917: 50}.get(cid, 0)

    def _opening_active_priority(self, cid: int) -> float:
        """Lead with a one-prize pivot while the real attacker develops on Bench."""
        if self.deck_key == "zoroark":
            return {292: 500, 906: 90, 140: 20, 141: 10}.get(cid, 0)
        if self.deck_key == "mewtwo":
            return {433: 520, 463: 470, 400: 330, 431: 40, 24: 20}.get(cid, 0)
        return {917: 500, 650: 350, 96: 20}.get(cid, 0)

    def _bench_card_score(self, obs: Observation, cid: int, from_hand: bool = False) -> float:
        me = self._me(obs)
        if len(me.bench) >= me.benchMax:
            return -1000
        in_play = [p.id for p in self._all_in_play(me)]
        if self.deck_key == "zoroark":
            desired = {292: 2, 293: 2, 906: 1, 141: 1, 140: 1}
            base = {292: 1700, 906: 1850, 141: 1150, 140: 800, 293: 300}.get(cid, 0)
        elif self.deck_key == "mewtwo":
            desired = {431: 2, 400: 2, 401: 2, 24: 1, 433: 1, 463: 1}
            base = {431: 1900, 400: 1750, 24: 1600, 433: 1250, 463: 1050, 401: 200}.get(cid, 0)
            if self._count_in_play(obs, TEAM_ROCKET_POKEMON) < 4 and cid in TEAM_ROCKET_POKEMON:
                base += 350
        else:
            desired = {96: 2, 650: 2, 917: 1, 651: 1, 709: 1, 652: 1, 710: 1}
            base = {96: 1850, 650: 1650, 917: 1550, 651: 250, 709: 250, 652: 250, 710: 250}.get(cid, 0)
        have = in_play.count(cid)
        if have >= desired.get(cid, 0):
            score = 250 if from_hand and cid in desired else -100
        else:
            score = base - 160 * have
        if not me.bench:
            if self.deck_key == "zoroark":
                score += 1800.0
            elif self.deck_key == "mewtwo":
                score += 900.0
        return score

    def _ready_value(self, obs: Observation, pokemon: Pokemon) -> float:
        units = self._energy_units(pokemon)
        value = self._active_priority(pokemon.id)
        if self.deck_key == "zoroark":
            if pokemon.id == 293 and units >= 2 and self._count_id_in_play(obs, 906) >= 1:
                value += 500
        elif self.deck_key == "mewtwo":
            if pokemon.id == 431 and units >= 3 and self._count_in_play(obs, TEAM_ROCKET_POKEMON) >= 4:
                value += 550
            elif pokemon.id == 24 and units >= 3:
                value += 450
            elif pokemon.id == 401 and units >= 2:
                value += 350
        else:
            if pokemon.id == 652 and units >= 4:
                value += 550
            elif pokemon.id == 96 and units >= 3:
                value += 420
            elif pokemon.id == 710 and units >= 4:
                value += 320
        return value

    def _best_ready_bench(self, obs: Observation, only_ids: set[int] | None = None) -> Pokemon | None:
        candidates = [p for p in self._me(obs).bench if only_ids is None or p.id in only_ids]
        if not candidates:
            return None
        best = max(candidates, key=lambda p: self._ready_value(obs, p))
        threshold = 500 if only_ids is not None else 400
        return best if self._ready_value(obs, best) >= threshold else None

    def _missing_zoroark_piece(self, obs: Observation) -> bool:
        ids = [p.id for p in self._all_in_play(self._me(obs))] + self._hand_ids(obs)
        return (292 in ids and 293 not in ids) or 906 not in ids

    def _missing_mewtwo_piece(self, obs: Observation) -> bool:
        ids = [p.id for p in self._all_in_play(self._me(obs))] + self._hand_ids(obs)
        return 431 not in ids or (400 in ids and 401 not in ids)

    def _missing_venusaur_piece(self, obs: Observation) -> bool:
        ids = [p.id for p in self._all_in_play(self._me(obs))] + self._hand_ids(obs)
        return (650 in ids and 652 not in ids) or (917 in ids and 710 not in ids)

    def _janine_is_live(self, obs: Observation) -> bool:
        dark = [p for p in self._all_in_play(self._me(obs)) if _CARD[p.id].energyType == EnergyType.DARKNESS]
        return bool(dark and 7 in self.deck)

    def _rare_candy_live(self, obs: Observation) -> bool:
        hand = self._hand_ids(obs)
        basic_old = any(p.id == 650 and not p.appearThisTurn for p in self._all_in_play(self._me(obs)))
        return 652 in hand and basic_old

    def _damaged_mega(self, obs: Observation) -> bool:
        return any(_CARD[p.id].megaEx and p.hp < p.maxHp for p in self._all_in_play(self._me(obs)))

    def _opponent_is_ex(self, obs: Observation) -> bool:
        p = self._opponent_active(obs)
        return bool(p and (_CARD[p.id].ex or _CARD[p.id].megaEx))

    def _active_needs_energy(self, obs: Observation) -> bool:
        p = self._active(obs)
        if p is None:
            return False
        need = {431: 3, 24: 3, 401: 2}.get(p.id, 0)
        return self._energy_units(p) < need

    # ------------------------ non-main selections -------------------------

    def _choose_yes_no(self, obs: Observation) -> int:
        assert obs.select is not None
        yes = [i for i, o in enumerate(obs.select.option) if o.type == OptionType.YES]
        no = [i for i, o in enumerate(obs.select.option) if o.type == OptionType.NO]
        ctx = obs.select.context
        effect = self._effect_id(obs)

        choose_yes = True
        if ctx == SelectContext.ACTIVATE:
            if effect == 293:
                choose_yes = self._me(obs).handCount >= 2
            elif effect == 431:
                choose_yes = sum(len(p.energyCards) for p in self._me(obs).bench) > 0
            elif effect == 652:
                active = self._active(obs)
                choose_yes = bool(active and active.id in {652, 710} and self._energy_units(active) < 4)
        elif ctx == SelectContext.COIN_HEAD:
            choose_yes = bool(self.rng.randrange(2))
        elif ctx in {SelectContext.IS_FIRST, SelectContext.MULLIGAN, SelectContext.FIRST_EFFECT}:
            choose_yes = True

        pool = yes if choose_yes and yes else no if no else yes
        return pool[0] if pool else 0

    def _choose_count_option(self, obs: Observation) -> int:
        assert obs.select is not None
        options = obs.select.option
        return self._argmax(
            [_Choice(i, float(int(option.number or 0))) for i, option in enumerate(options)],
            context="count",
        )

    def _choose_multi(self, obs: Observation) -> list[int]:
        assert obs.select is not None
        select = obs.select
        options = select.option
        ctx = select.context

        if ctx == SelectContext.DISCARD:
            ranked = sorted(
                range(len(options)),
                key=lambda i: self._discard_keep_value(obs, self._option_card_id(obs, options[i])),
            )
            return ranked[: select.minCount]

        if ctx == SelectContext.DISCARD_ENERGY_CARD and self._effect_id(obs) == 431:
            ranked = sorted(
                range(len(options)),
                key=lambda i: self._energy_discard_score(obs, options[i]),
                reverse=True,
            )
            count = min(select.maxCount, max(select.minCount, 2))
            return ranked[:count]

        scored = [_Choice(i, self._selection_score(obs, option)) for i, option in enumerate(options)]
        if scored:
            best = max(c.score for c in scored)
            width = sum(abs(c.score - best) < 1e-9 for c in scored)
            self._record_tie(width, f"multi:{obs.select.context}")
            self._record_meaningful_tie(obs, scored, f"multi:{obs.select.context}")
        scored.sort(key=lambda c: (c.score, self.rng.random()), reverse=True)

        variable_benefit = ctx in {
            SelectContext.SETUP_BENCH_POKEMON,
            SelectContext.TO_BENCH,
            SelectContext.TO_FIELD,
            SelectContext.TO_HAND,
            SelectContext.LOOK,
            SelectContext.HEAL,
            SelectContext.REMOVE_DAMAGE_COUNTER,
        }
        if variable_benefit and select.maxCount > select.minCount:
            threshold = 350.0 if ctx == SelectContext.SETUP_BENCH_POKEMON else 100.0
            chosen = [c.index for c in scored if c.score >= threshold][: select.maxCount]
            if len(chosen) < select.minCount:
                chosen = [c.index for c in scored[: select.minCount]]
            return chosen

        count = select.minCount
        if select.maxCount == select.minCount:
            count = select.minCount
        elif ctx in {
            SelectContext.TO_HAND,
            SelectContext.TO_BENCH,
            SelectContext.TO_FIELD,
            SelectContext.LOOK,
            SelectContext.HEAL,
            SelectContext.REMOVE_DAMAGE_COUNTER,
        }:
            count = select.maxCount
        return [c.index for c in scored[:count]]

    def _selection_score(self, obs: Observation, option: Option) -> float:
        assert obs.select is not None and obs.current is not None
        ctx = obs.select.context
        cid = self._option_card_id(obs, option)
        target = self._option_target_pokemon(obs, option)

        if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
            return self._opening_active_priority(cid or -1)
        if ctx in {SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD}:
            return self._bench_card_score(obs, cid or -1)
        if ctx in {SelectContext.TO_HAND, SelectContext.LOOK}:
            return self._search_score(obs, cid)
        if ctx in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            if target is None:
                return 0
            if option.playerIndex == obs.current.yourIndex:
                return self._ready_value(obs, target)
            data = _CARD[target.id]
            prize = 3 if data.megaEx else 2 if data.ex else 1
            return 500 + 3 * (target.maxHp - target.hp) - target.hp + 250 * prize
        if ctx in {SelectContext.ATTACH_FROM, SelectContext.EFFECT_TARGET} and target is not None:
            if self._effect_id(obs) == 652:
                active = self._active(obs)
                return 5000.0 if active is not None and target.serial == active.serial else -1000.0
            return self._energy_target_selection_score(obs, target)
        if ctx == SelectContext.ATTACH_TO:
            return self._energy_card_selection_score(obs, cid)
        if ctx in {
            SelectContext.SWITCH_ENERGY_CARD,
            SelectContext.SWITCH_ENERGY,
            SelectContext.DISCARD_ENERGY_CARD,
        }:
            return self._energy_discard_score(obs, option)
        if ctx in {SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY, SelectContext.DAMAGE}:
            if target is None:
                return 0
            enemy = option.playerIndex != obs.current.yourIndex
            data = _CARD[target.id]
            prize = 3 if data.megaEx else 2 if data.ex else 1
            return (1200 if enemy else -500) + 300 * prize - target.hp
        if ctx in {SelectContext.REMOVE_DAMAGE_COUNTER, SelectContext.HEAL}:
            return (target.maxHp - target.hp) if target is not None else 0
        if ctx in {SelectContext.EVOLVES_FROM, SelectContext.EVOLVES_TO, SelectContext.EVOLVE}:
            return self._search_score(obs, cid)
        if option.type == OptionType.ATTACK and option.attackId is not None:
            return self._attack_score(obs, option.attackId)
        if option.type == OptionType.ENERGY:
            return float(option.count or 0)
        if option.type == OptionType.SKILL:
            return 100.0
        return self._search_score(obs, cid)

    def _search_score(self, obs: Observation, cid: int | None) -> float:
        if cid is None:
            return 0.0
        ids_in_play = [p.id for p in self._all_in_play(self._me(obs))]
        hand = self._hand_ids(obs)

        if self.deck_key == "zoroark":
            base = {
                293: 2200, 906: 2050, 292: 1900, 1195: 1850, 7: 1750,
                1086: 1500, 1121: 1450, 1113: 1400, 1162: 1350,
                1253: 1200, 1227: 1150, 1182: 1050, 141: 1000,
                140: 850, 1097: 800, 1092: 750, 1211: 700,
            }.get(cid, 300)
            if cid == 293 and 292 not in ids_in_play:
                base -= 800
            if cid == 906 and 906 in ids_in_play:
                base -= 900
            if cid == 292 and ids_in_play.count(292) + ids_in_play.count(293) >= 2:
                base -= 700
        elif self.deck_key == "mewtwo":
            base = {
                431: 2200, 401: 2050, 400: 1950, 24: 1800, 1220: 1750,
                1216: 1550, 1134: 1500, 15: 1450, 1: 1350, 1121: 1300,
                1094: 1200, 1257: 1100, 1218: 1050, 433: 1000,
                463: 800, 1116: 750, 1158: 700,
            }.get(cid, 300)
            if cid == 401 and 400 not in ids_in_play:
                base -= 700
            if cid == 431 and 431 in ids_in_play:
                base -= 500
            if cid == 1220 and self._count_in_play(obs, TEAM_ROCKET_POKEMON) >= 4:
                base -= 900
        else:
            base = {
                652: 2250, 710: 2100, 651: 1950, 709: 1900, 650: 1800,
                917: 1750, 96: 1700, 1: 1600, 1261: 1500, 1079: 1450,
                1225: 1350, 1121: 1300, 1094: 1200, 1227: 1100,
                1182: 900, 1229: 750, 1080: 700,
            }.get(cid, 300)
            if cid == 652 and not any(x in ids_in_play for x in {650, 651}):
                base -= 700
            if cid == 710 and not any(x in ids_in_play for x in {917, 709}):
                base -= 700
            if cid == 1261 and obs.current.stadium:
                base -= 900

        # Penalize excessive duplicates already visible in hand/in play.
        visible_count = ids_in_play.count(cid) + hand.count(cid)
        score = base - max(0, visible_count - 1) * 170
        if not self._me(obs).bench and cid in _CARD and _CARD[cid].cardType == CardType.POKEMON:
            if self.deck_key == "zoroark":
                score += 1500.0 if _CARD[cid].basic else -450.0
            elif self.deck_key == "mewtwo":
                score += 700.0 if _CARD[cid].basic else -250.0
        return score

    def _discard_keep_value(self, obs: Observation, cid: int | None) -> float:
        if cid is None:
            return -1000
        data = _CARD[cid]
        hand = self._hand_ids(obs)
        duplicates = hand.count(cid)
        value = 500.0
        if data.cardType == CardType.POKEMON:
            value = 1100.0
        elif data.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            value = 850.0
        elif data.cardType == CardType.SUPPORTER:
            value = 620.0
        elif data.cardType in (CardType.ITEM, CardType.TOOL, CardType.STADIUM):
            value = 520.0

        key = {
            "zoroark": {292, 293, 906, 1195, 1086, 1121},
            "mewtwo": {431, 400, 401, 24, 1220, 1216, 1134},
            "venusaur": {96, 650, 651, 652, 917, 709, 710, 1079, 1261},
        }[self.deck_key]
        if cid in key:
            value += 500
        if duplicates > 1:
            value -= 180 * (duplicates - 1)

        effect = self._effect_id(obs)
        if self.deck_key == "zoroark":
            if cid == 7 and effect in {293, 1121, 1092} and 1113 in hand:
                value = 120  # Deliberately fuel N's PP Up.
            if cid == 906 and self._count_id_in_play(obs, 906) >= 1:
                value -= 650
            if cid in {1253, 1086} and self._count_in_play(obs, {292, 293}) >= 2:
                value -= 300
        elif self.deck_key == "mewtwo":
            if cid == 1 and effect in {1121} and self._count_id_in_play(obs, 401) >= 1:
                value = 180  # Deliberately fuel Charging Up.
            if cid == 1220 and self._count_in_play(obs, TEAM_ROCKET_POKEMON) >= 4:
                value -= 500
        else:
            if cid == 1261 and obs.current and obs.current.stadium:
                value -= 500
        return value

    def _energy_target_selection_score(self, obs: Observation, target: Pokemon) -> float:
        active = self._active(obs)
        is_active = bool(active and target.serial == active.serial)
        units = self._energy_units(target)
        if self.deck_key == "zoroark":
            if target.id in {292, 293}:
                return 1800 - 180 * units + (250 if is_active else 0)
            return 200
        if self.deck_key == "mewtwo":
            if target.id == 431:
                return 1900 - 160 * units + (250 if is_active else 0)
            if target.id == 24:
                return 1650 - 150 * units + (180 if is_active else 0)
            if target.id == 401:
                return 1450 - 140 * units + (100 if is_active else 0)
            return 200
        if target.id == 652:
            return 2000 - 150 * units + (300 if is_active else 0)
        if target.id == 96:
            return 1700 - 140 * units + (180 if is_active else 0)
        if target.id == 710:
            return 1500 - 140 * units + (150 if is_active else 0)
        return 300

    def _energy_card_selection_score(self, obs: Observation, cid: int | None) -> float:
        effect = self._effect_id(obs)
        if cid is None:
            return 0
        if effect == 401:
            return 1000 if cid == 1 else -500
        if effect == 1113:
            return 1000 if cid == 7 else -500
        if self.deck_key == "zoroark":
            return 900 if cid == 7 else 0
        if self.deck_key in {"mewtwo", "venusaur"}:
            return 900 if cid == 1 else 500 if cid == 15 else 0
        return 0

    def _energy_discard_score(self, obs: Observation, option: Option) -> float:
        parent = self._option_target_pokemon(obs, option)
        active = self._active(obs)
        if parent is None:
            return 0
        is_active = bool(active and parent.serial == active.serial)
        if self._effect_id(obs) == 431:
            # Feed Erasure Ball from benched Spidops/Kangaskhan; never strip Mewtwo.
            if is_active:
                return -1000
            return {401: 1000, 24: 800, 400: 600, 431: 100}.get(parent.id, 300)
        # Energy Switch / Solar Transfer: move from a loaded bench engine to Active.
        if self.deck_key == "venusaur":
            return {96: 1000, 710: 700, 652: 400}.get(parent.id, 300) - (500 if is_active else 0)
        if self.deck_key == "mewtwo":
            return {401: 1000, 24: 700, 431: 300}.get(parent.id, 300) - (500 if is_active else 0)
        return 300 - (500 if is_active else 0)

    # ------------------------------- utility -------------------------------

    def _argmax(self, choices: Iterable[_Choice], context: str = "argmax") -> int:
        choices = list(choices)
        best_score = max(c.score for c in choices)
        tied = [c.index for c in choices if abs(c.score - best_score) < 1e-9]
        self._record_tie(len(tied), context)
        return self.rng.choice(tied)

    def _record_tie(self, width: int, context: str) -> None:
        self.scored_decisions += 1
        if width > 1:
            self.tie_decisions += 1
            self.tie_widths[width] = self.tie_widths.get(width, 0) + 1
            self.tie_contexts[context] = self.tie_contexts.get(context, 0) + 1

    def _record_meaningful_tie(self, obs: Observation, choices: Iterable[_Choice], context: str) -> None:
        choices = list(choices)
        if not choices or obs.select is None:
            return
        best = max(c.score for c in choices)
        tied = [c for c in choices if abs(c.score - best) < 1e-9]
        signatures = set()
        for choice in tied:
            option = obs.select.option[choice.index]
            target = self._option_target_pokemon(obs, option)
            signatures.add((
                int(option.type), self._option_card_id(obs, option), option.attackId,
                getattr(target, "id", None), getattr(target, "serial", None),
                option.number, option.playerIndex,
            ))
        if len(signatures) > 1:
            self.meaningful_tie_decisions += 1
            self.meaningful_tie_contexts[context] = self.meaningful_tie_contexts.get(context, 0) + 1


_POLICY = BattlePolicy(announce=True)


def agent(obs_dict: dict) -> list[int]:
    """Kaggle entry point."""
    return _POLICY.act(obs_dict)
