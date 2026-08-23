"""Card-specific priority weights for the Teal Mask Ogerpon ex deck.

Why card-specific: three independent sources say this is the gap between us and
the agents above us.

  * romanrozen (LB ~950): 74 unique weights keyed to individual cards in specific
    situations (play_abra_early, poffin_late, pokepad_need, sacred_ash_hi, ...).
  * llccqq624 (LB 797): 323 card-specific if-branches and NO forward search at all
    -- it beats our search-based agent by ~190 points on card knowledge alone.
  * discussion 713608: nine methods plateaued because they "encoded only option
    type and damage" and could not see card synergies.

Our domain_policy scores EVERY trainer at a flat 6000 and every Pokemon at 20000.
This module gives each card in this deck its own weight, conditioned on the game
state, and feeds that into hybrid search rather than replacing it -- the mistake
that made grimmsnarl_policy (flat weights, no search) score 443.6.

Deck engine:
  Teal Dance (ability)  attach a Basic {G} from hand every turn, free, and draw.
  Myriad Leaf Shower    30 damage +30 for EACH Energy on BOTH Active Pokemon.
So Energy is doubly valuable: acceleration AND damage scaling. Hero's Cape (+100)
and Lively Stadium (+30) turn a 210 HP Ogerpon into a 340 HP attacker, and with
only 4 Pokemon in the deck, finding and protecting one is critical.

All weights are tunable; SPSA writes overrides to ogerpon_w.json.
"""
from __future__ import annotations

import json
import os

from domain_policy import (  # noqa: F401
    DomainPolicy, SelectContext, OptionType, AreaType, CardType,
    _CARDS, _get_card, _prize, _ACTIVE, _BENCH, evaluate_board, domain_agent,
)

# --- the deck ---
OGERPON = 96          # Teal Mask Ogerpon ex, 210 HP, Teal Dance + Myriad Leaf Shower
BUG_CATCHING_SET = 1094
ENERGY_SEARCH = 1119
POKEGEAR = 1122
ENERGY_RETRIEVAL = 1118
TERA_ORB = 1127
JUMBO_ICE_CREAM = 1147
TOOL_SCRAPPER = 1137
HEROS_CAPE = 1159
JUDGE = 1213
LILLIES = 1227
BOSS_ORDERS = 1182
HARLEQUIN = 1223
BRIAR = 1201
NS_PLAN = 1221
LIVELY_STADIUM = 1251
GRASS_ENERGY = 1
GROW_GRASS = 18

_HAND = getattr(AreaType, "HAND", 2)

W = {
    # engine
    "ability_teal_dance": 34000.0,   # free energy + draw every turn: never skip
    "play_ogerpon": 21000.0,         # only 4 in the deck; board presence is life
    # finders -- weighted by what the deck is missing
    "tera_orb_need": 19000.0,        # finds our ONLY attacker
    "tera_orb_have": 5000.0,
    "bug_set_early": 17000.0,        # top 7 -> Pokemon + basic G
    "bug_set_late": 8000.0,
    "energy_search_need": 12000.0,
    "energy_search_have": 4000.0,
    "energy_retrieval": 9000.0,      # only useful with energy in discard
    "pokegear": 10000.0,
    # durability -- Ogerpon is the whole deck
    "heros_cape": 18000.0,           # +100 HP on our sole attacker
    "lively_stadium": 14000.0,       # +30 HP to basics
    "jumbo_ice_cream_hurt": 16000.0, # heal 80 when damaged and 3+ energy
    "jumbo_ice_cream_fresh": 1500.0,
    "tool_scrapper": 6000.0,
    # draw / disruption -- value depends on hand size
    "lillies_thin": 15000.0,
    "lillies_wide": 3000.0,
    "judge_thin": 11000.0,           # symmetric; good when our hand is bad
    "judge_wide": 2500.0,
    "harlequin_thin": 9000.0,
    "harlequin_wide": 2000.0,
    "ns_plan": 7000.0,               # move energy bench -> active
    # tempo
    "boss_kill": 26000.0,            # gust something we can knock out
    "boss_plain": 7000.0,
    "briar_live": 24000.0,           # only legal at opponent 2 prizes: near-lethal
    # energy attachment: damage scales with total energy, so attach aggressively
    "attach_ogerpon_active": 12000.0,
    "attach_generic": 8000.0,
    "grow_grass_bonus": 600.0,
    "attack_planned": 1400.0,
    "attack_other": 1000.0,
}

for _p in ("ogerpon_w.json",
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "ogerpon_w.json"),
           "/kaggle_simulations/agent/ogerpon_w.json"):
    try:
        if os.path.exists(_p):
            W.update(json.load(open(_p)))
            break
    except Exception:
        pass


class OgerponPolicy(DomainPolicy):
    # ---- small state helpers ----
    def _hand_ids(self):
        try:
            return [c.id for c in (self.myp.hand or []) if c is not None]
        except Exception:
            return []

    def _energy_in_hand(self):
        h = self._hand_ids()
        return sum(1 for c in h if c in (GRASS_ENERGY, GROW_GRASS))

    def _my_active(self):
        try:
            return self.myp.active[0] if self.myp.active else None
        except Exception:
            return None

    def _active_energy(self):
        a = self._my_active()
        return len(a.energies or []) if a else 0

    def _active_hurt(self):
        a = self._my_active()
        if not a:
            return 0
        return max(0, (a.maxHp or 0) - (a.hp or 0))

    def _energy_in_discard(self):
        try:
            return sum(1 for c in (self.myp.discard or [])
                       if c is not None and c.id in (GRASS_ENERGY, GROW_GRASS))
        except Exception:
            return 0

    def _opp_prizes(self):
        try:
            return len(self.opp.prize or [])
        except Exception:
            return 6

    # ---- scoring ----
    def _card_id(self, o):
        """Resolve the card an option refers to.

        Option.cardId is None for PLAY/EVOLVE -- the engine identifies the card by
        option.index into the HAND area. Reading cardId directly left every
        card-specific branch below unreachable, so this policy was measured as a
        generic one, and the SPSA / memetic runs that tuned these weights were
        optimising parameters with no effect.
        """
        cid = getattr(o, "cardId", None)
        if cid is not None:
            return cid
        try:
            c = _get_card(self.obs, _HAND, o.index, self.me)
            return c.id if c is not None else None
        except Exception:
            return None

    def _score(self, o):
        t = o.type
        cid = self._card_id(o)

        if t == OptionType.ABILITY:
            # Teal Dance is the deck's whole engine: free energy AND a card.
            return W["ability_teal_dance"]

        if t == OptionType.PLAY:
            hand = self._hand_ids()
            turn = getattr(self.st, "turn", 0)
            if cid == OGERPON:
                return W["play_ogerpon"]
            if cid == TERA_ORB:
                have = OGERPON in hand or any(
                    p is not None and p.id == OGERPON for p in self._my_board())
                return W["tera_orb_have"] if have else W["tera_orb_need"]
            if cid == BUG_CATCHING_SET:
                return W["bug_set_early"] if turn <= 6 else W["bug_set_late"]
            if cid == ENERGY_SEARCH:
                return W["energy_search_have"] if self._energy_in_hand() >= 2 \
                    else W["energy_search_need"]
            if cid == ENERGY_RETRIEVAL:
                return W["energy_retrieval"] if self._energy_in_discard() >= 2 else 2000.0
            if cid == POKEGEAR:
                return W["pokegear"]
            if cid == HEROS_CAPE:
                return W["heros_cape"]
            if cid == LIVELY_STADIUM:
                return W["lively_stadium"]
            if cid == JUMBO_ICE_CREAM:
                return W["jumbo_ice_cream_hurt"] if (
                    self._active_hurt() >= 60 and self._active_energy() >= 3
                ) else W["jumbo_ice_cream_fresh"]
            if cid == TOOL_SCRAPPER:
                return W["tool_scrapper"]
            if cid == BOSS_ORDERS:
                return W["boss_kill"] if (self.plan and self.plan[3]) else W["boss_plain"]
            if cid == BRIAR:
                return W["briar_live"] if self._opp_prizes() == 2 else 500.0
            if cid == NS_PLAN:
                return W["ns_plan"]
            thin = len(hand) <= 3
            if cid == LILLIES:
                return W["lillies_thin"] if thin else W["lillies_wide"]
            if cid == JUDGE:
                return W["judge_thin"] if thin else W["judge_wide"]
            if cid == HARLEQUIN:
                return W["harlequin_thin"] if thin else W["harlequin_wide"]

        if t == OptionType.ATTACH:
            try:
                mon = _get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.me)
                if mon is not None and mon.id == OGERPON and o.inPlayArea == _ACTIVE:
                    base = W["attach_ogerpon_active"]
                    if cid == GROW_GRASS:
                        base += W["grow_grass_bonus"]
                    return base
            except Exception:
                pass
            return W["attach_generic"]

        if t == OptionType.ATTACK:
            if self.plan and getattr(o, "attackId", None) == self.plan[1]:
                return W["attack_planned"]
            return W["attack_other"]

        return super()._score(o)


def rank_options(obs):
    try:
        return OgerponPolicy(obs).choose()
    except Exception:
        n = len(obs.select.option) if obs.select else 0
        return list(range(n))


def ogerpon_agent(obs_dict, deck):
    from cg.api import to_observation_class
    try:
        if obs_dict.get("select") is None:
            return list(deck)
        obs = to_observation_class(obs_dict)
        n = len(obs.select.option)
        if n == 0:
            return []
        mc = obs.select.maxCount or 1
        r = [i for i in OgerponPolicy(obs).choose() if 0 <= i < n]
        return r[:mc] if r else list(range(min(mc, n)))
    except Exception:
        return domain_agent(obs_dict, deck)
