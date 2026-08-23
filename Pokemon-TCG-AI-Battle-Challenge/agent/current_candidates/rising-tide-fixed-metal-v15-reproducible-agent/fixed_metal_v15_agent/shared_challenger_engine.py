"""General policy components offered equally to every local challenger deck."""

from __future__ import annotations

from typing import Any


def build_policies(base: Any) -> dict[str, type]:
    """Build isolated and combined policies against the supplied v8 engine."""

    def ready(policy, pokemon) -> bool:
        costs = [
            len(base._ATTACK[attack_id].energies)
            for attack_id in base._CARD[pokemon.id].attacks
            if attack_id in base._ATTACK
        ]
        return bool(costs) and len(pokemon.energyCards) >= min(costs)

    class SetupPolicy(base.HardGeneralPolicy):
        """Complete a useful attachment before an attack ends the turn."""

        def _main_score(self, obs, option):
            score = super()._main_score(obs, option)
            if (
                obs.select is None
                or option.type != base.OptionType.ATTACH
                or not any(
                    item.type == base.OptionType.ATTACK
                    for item in obs.select.option
                )
            ):
                return score
            target = self._option_target_pokemon(obs, option)
            card_id = self._option_card_id(obs, option)
            if target is None or card_id is None:
                return score
            if base._CARD[card_id].cardType not in {
                base.CardType.BASIC_ENERGY,
                base.CardType.SPECIAL_ENERGY,
            }:
                return score
            costs = [
                len(base._ATTACK[attack_id].energies)
                for attack_id in base._CARD[target.id].attacks
                if attack_id in base._ATTACK
            ]
            if not costs:
                return score
            units = self._energy_units(target)
            need = min(costs)
            active = self._active(obs)
            is_bench = (
                active is None or target.serial != active.serial
            )
            if units < need <= units + 1:
                score += 500.0 + 400.0 * int(is_bench)
            return score

    class RotationPolicy(base.HardGeneralPolicy):
        """Rotate only under a near-certain KO and only to a ready attacker."""

        def _reachable_next_damage(self, pokemon, defender) -> int:
            units = len(pokemon.energyCards) + 1
            best = 0
            for attack_id in base._CARD[pokemon.id].attacks:
                attack = base._ATTACK.get(attack_id)
                if (
                    attack is None
                    or len(attack.energies) > units
                ):
                    continue
                damage = int(attack.damage)
                if (
                    damage > 0
                    and base._CARD[defender.id].weakness
                    == base._CARD[pokemon.id].energyType
                ):
                    damage *= 2
                elif (
                    damage > 0
                    and base._CARD[defender.id].resistance
                    == base._CARD[pokemon.id].energyType
                ):
                    damage = max(0, damage - 30)
                best = max(best, damage)
            return best

        def _retreat_score(self, obs):
            score = super()._retreat_score(obs)
            active = self._active(obs)
            opponent = self._opponent_active(obs)
            if (
                active is None
                or opponent is None
                or self._reachable_next_damage(opponent, active)
                < active.hp
            ):
                return score
            ready_bench = [
                pokemon
                for pokemon in self._me(obs).bench
                if ready(self, pokemon)
            ]
            if not ready_bench:
                return score
            if (
                base._maximum_attack_damage(active.id)
                >= opponent.hp
            ):
                return score
            best = max(
                ready_bench,
                key=lambda pokemon: (
                    base._CARD[opponent.id].weakness
                    == base._CARD[pokemon.id].energyType,
                    base._maximum_attack_damage(pokemon.id),
                    pokemon.hp,
                ),
            )
            bonus = 700.0
            if (
                base._CARD[opponent.id].weakness
                == base._CARD[best.id].energyType
            ):
                bonus += 500.0
            return score + bonus

    class DevelopmentPolicy(base.HardGeneralPolicy):
        """Value generic search, draw, evolution, and attack acceleration."""

        def _evolve_score(self, obs, option):
            score = super()._evolve_score(obs, option)
            card_id = self._option_card_id(obs, option)
            target = self._option_target_pokemon(obs, option)
            if card_id is None or target is None:
                return score
            new = base._CARD[card_id]
            old = base._CARD[target.id]
            active = self._active(obs)
            is_bench = (
                active is None or active.serial != target.serial
            )
            value = (
                1650.0
                + 1.8 * max(0, int(new.hp) - int(old.hp))
                + 0.9
                * max(
                    0,
                    base._maximum_attack_damage(card_id)
                    - base._maximum_attack_damage(target.id),
                )
                + 260.0 * int(bool(new.skills))
                + 180.0 * self._energy_units(target)
                + 220.0
                * int(
                    is_bench
                    and not any(
                        ready(self, pokemon)
                        for pokemon in self._me(obs).bench
                    )
                )
            )
            return max(score, value)

        def _play_score(self, obs, option):
            score = super()._play_score(obs, option)
            card_id = self._option_card_id(obs, option)
            if card_id is None:
                return score
            card = base._CARD[card_id]
            text = " ".join(
                skill.text for skill in card.skills
            ).lower()
            hand = self._me(obs).handCount
            deck = self._me(obs).deckCount
            if (
                "search your deck for a pok" in text
                and len(self._all_in_play(self._me(obs))) < 2
            ):
                score = max(score, 2100.0)
            if "draw " in text or "draws " in text:
                score = max(
                    score,
                    1700.0
                    if hand <= 4 and deck > 8
                    else 700.0,
                )
            if "look at the top" in text and "put" in text:
                score = max(
                    score,
                    1650.0
                    if hand <= 5 and deck > 10
                    else 650.0,
                )
            return score

        def _attack_score(self, obs, attack_id):
            score = super()._attack_score(obs, attack_id)
            attack = base._ATTACK.get(attack_id)
            if attack is None:
                return score
            text = attack.text.lower()
            if (
                "attach up to" in text
                and "energy" in text
                and self._me(obs).bench
                and not any(
                    ready(self, pokemon)
                    for pokemon in self._me(obs).bench
                )
            ):
                score += 520.0
            return score

    class SharedFullPolicy(
        SetupPolicy, RotationPolicy, DevelopmentPolicy
    ):
        """All shared rules combined; retained only if certification wins."""

    class AdaptiveDevelopmentPolicy(DevelopmentPolicy):
        """Use development planning only when the deck has evolutions."""

        def __init__(self, deck, *args, **kwargs):
            super().__init__(deck, *args, **kwargs)
            self._development_enabled = any(
                base._CARD[card_id].cardType
                == base.CardType.POKEMON
                and (
                    base._CARD[card_id].stage1
                    or base._CARD[card_id].stage2
                )
                for card_id in deck
            )

        def _evolve_score(self, obs, option):
            if not self._development_enabled:
                return base.HardGeneralPolicy._evolve_score(
                    self, obs, option
                )
            return super()._evolve_score(obs, option)

        def _play_score(self, obs, option):
            if not self._development_enabled:
                return base.HardGeneralPolicy._play_score(
                    self, obs, option
                )
            return super()._play_score(obs, option)

        def _attack_score(self, obs, attack_id):
            if not self._development_enabled:
                return base.HardGeneralPolicy._attack_score(
                    self, obs, attack_id
                )
            return super()._attack_score(obs, attack_id)

    class Stage2DevelopmentPolicy(AdaptiveDevelopmentPolicy):
        """Conservative form: enable the planner only for Stage-2 decks."""

        def __init__(self, deck, *args, **kwargs):
            super().__init__(deck, *args, **kwargs)
            self._development_enabled = any(
                base._CARD[card_id].cardType
                == base.CardType.POKEMON
                and base._CARD[card_id].stage2
                for card_id in deck
            )

    class ReplyAwarePolicy(SharedFullPolicy):
        """Two-ply threat/reply heuristic with a ready-counterattack objective."""

        def _predicted_damage(self, attacker, defender) -> int:
            if attacker is None or defender is None:
                return 0
            units = len(attacker.energyCards) + 1
            best = 0
            for attack_id in base._CARD[attacker.id].attacks:
                attack = base._ATTACK.get(attack_id)
                if (
                    attack is None
                    or len(attack.energies) > units
                ):
                    continue
                damage = int(attack.damage)
                if (
                    damage > 0
                    and base._CARD[defender.id].weakness
                    == base._CARD[attacker.id].energyType
                ):
                    damage *= 2
                elif (
                    damage > 0
                    and base._CARD[defender.id].resistance
                    == base._CARD[attacker.id].energyType
                ):
                    damage = max(0, damage - 30)
                best = max(best, damage)
            return best

        def _reply_threat(self, obs) -> int:
            active = self._active(obs)
            if active is None:
                return 0
            opponent = self._opp(obs)
            attackers = self._all_in_play(opponent)
            if not attackers:
                return 0
            # Assume the opponent finds one attachment and chooses its best
            # currently exposed attacker. This is a bounded one-reply model,
            # not a costly stochastic rollout.
            return max(
                self._predicted_damage(attacker, active)
                for attacker in attackers
            )

        def _damage_after_attachment(
            self, pokemon, defender
        ) -> int:
            return self._predicted_damage(pokemon, defender)

        def _attach_score(self, obs, option):
            score = super()._attach_score(obs, option)
            target = self._option_target_pokemon(obs, option)
            card_id = self._option_card_id(obs, option)
            active = self._active(obs)
            opponent = self._opponent_active(obs)
            if (
                target is None
                or card_id is None
                or active is None
                or base._CARD[card_id].cardType
                not in {
                    base.CardType.BASIC_ENERGY,
                    base.CardType.SPECIAL_ENERGY,
                }
            ):
                return score
            threatened = self._reply_threat(obs) >= active.hp
            is_active = target.serial == active.serial
            need = min(
                (
                    len(base._ATTACK[attack_id].energies)
                    for attack_id in base._CARD[target.id].attacks
                    if attack_id in base._ATTACK
                ),
                default=99,
            )
            becomes_ready = (
                len(target.energyCards) < need
                <= len(target.energyCards) + 1
            )
            if threatened and not is_active and becomes_ready:
                score += 850.0
                if (
                    opponent is not None
                    and self._damage_after_attachment(
                        target, opponent
                    )
                    >= opponent.hp
                ):
                    score += 450.0
            elif (
                threatened
                and is_active
                and ready(self, active)
                and any(
                    ready(self, pokemon)
                    for pokemon in self._me(obs).bench
                )
            ):
                score -= 300.0
            return score

        def _play_score(self, obs, option):
            score = super()._play_score(obs, option)
            card_id = self._option_card_id(obs, option)
            active = self._active(obs)
            if (
                card_id is not None
                and active is not None
                and base._CARD[card_id].cardType
                == base.CardType.POKEMON
                and len(self._all_in_play(self._me(obs))) < 2
                and self._reply_threat(obs) >= active.hp
            ):
                score += 950.0
            return score

        def _retreat_score(self, obs):
            score = super()._retreat_score(obs)
            active = self._active(obs)
            if active is None:
                return score
            threatened = self._reply_threat(obs) >= active.hp
            energy_ids = {
                card_id
                for card_id in self.deck
                if base._CARD[card_id].cardType
                in {
                    base.CardType.BASIC_ENERGY,
                    base.CardType.SPECIAL_ENERGY,
                }
            }
            total_energy = sum(
                card_id in energy_ids for card_id in self.deck
            )
            visible_energy = sum(
                card.id in energy_ids
                for card in (
                    (self._me(obs).hand or [])
                    + self._me(obs).discard
                )
            )
            visible_energy += sum(
                len(pokemon.energyCards)
                for pokemon in self._all_in_play(self._me(obs))
            )
            remaining = max(0, total_energy - visible_energy)
            retreat_cost = int(
                base._CARD[active.id].retreatCost
            )
            if (
                not threatened
                and remaining <= retreat_cost + 2
                and score > 100.0
            ):
                return min(score, 100.0)
            return score

    class DeckAdaptivePolicy(SharedFullPolicy):
        """Keep archetype parameters active after small legal deck changes."""

        def __init__(self, deck, *args, **kwargs):
            super().__init__(deck, *args, **kwargs)
            ids = set(deck)
            if (
                {677, 678}.issubset(ids)
                and deck.count(6) >= 8
            ):
                self.backup_setup_mode = "earth"
            elif 666 in ids and deck.count(3) >= 6:
                self.backup_setup_mode = "fire"

    class DeckAdaptiveReplyPolicy(
        ReplyAwarePolicy, DeckAdaptivePolicy
    ):
        """Deck adaptation plus bounded opponent reply/counterplay."""

    class EnergyPreservingReplyPolicy(ReplyAwarePolicy):
        """Extract movable Energy from a doomed active before giving a prize."""

        def _play_score(self, obs, option):
            score = super()._play_score(obs, option)
            card_id = self._option_card_id(obs, option)
            active = self._active(obs)
            if (
                card_id == 1116
                and active is not None
                and active.energyCards
                and self._me(obs).bench
                and self._reply_threat(obs) >= active.hp
            ):
                score = max(score, 2650.0)
            return score

        def _selection_score(self, obs, option):
            score = super()._selection_score(obs, option)
            if (
                obs.select is None
                or self._effect_id(obs) != 1116
            ):
                return score
            target = self._option_target_pokemon(obs, option)
            if target is None:
                return score
            active = self._active(obs)
            is_active = (
                active is not None
                and target.serial == active.serial
            )
            threatened = (
                active is not None
                and self._reply_threat(obs) >= active.hp
            )
            if obs.select.context in {
                base.SelectContext.SWITCH_ENERGY_CARD,
                base.SelectContext.SWITCH_ENERGY,
                base.SelectContext.DISCARD_ENERGY_CARD,
            }:
                if threatened and is_active:
                    return 3200.0 + 100.0 * len(
                        target.energyCards
                    )
                return 300.0 - 500.0 * int(is_active)
            if obs.select.context in {
                base.SelectContext.ATTACH_FROM,
                base.SelectContext.EFFECT_TARGET,
            }:
                if threatened and not is_active:
                    need = min(
                        (
                            len(base._ATTACK[attack_id].energies)
                            for attack_id in base._CARD[
                                target.id
                            ].attacks
                            if attack_id in base._ATTACK
                        ),
                        default=3,
                    )
                    return (
                        2600.0
                        + 450.0
                        * int(
                            len(target.energyCards) + 1 >= need
                        )
                        + target.hp
                    )
                if threatened and is_active:
                    return -1200.0
            return score

    return {
        "baseline": base.HardGeneralPolicy,
        "setup": SetupPolicy,
        "rotation": RotationPolicy,
        "development": DevelopmentPolicy,
        "adaptiveDevelopment": AdaptiveDevelopmentPolicy,
        "stage2Development": Stage2DevelopmentPolicy,
        "sharedFull": SharedFullPolicy,
        "replyAware": ReplyAwarePolicy,
        "deckAdaptive": DeckAdaptivePolicy,
        "deckAdaptiveReply": DeckAdaptiveReplyPolicy,
        "energyPreservingReply": EnergyPreservingReplyPolicy,
    }
