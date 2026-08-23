"""Our own Marnie's Grimmsnarl ex policy for the PTCG AI Battle.

This is an ORIGINAL implementation. It extends our deck-agnostic
`domain_policy.DomainPolicy` with the handful of decisions that generic,
card-data-driven scoring cannot see, all derived from the printed card text:

  * Punk Up  -- evolving into Marnie's Grimmsnarl ex from hand searches up to 5
    Basic {D} Energy onto your Marnie's Pokemon. That single trigger is the
    deck's whole acceleration plan, so Rare Candy -> Grimmsnarl ex is worth far
    more than the generic "play a trainer" score the base policy gives it.
  * Shadow Bullet -- 180 to the Active *plus 30 to a benched Pokemon*. The base
    attack planner only ever evaluates the Active, so the snipe target is chosen
    blindly. We aim it to set up or complete a knock-out.
  * Adrena-Brain (Munkidori) -- move up to 3 damage counters from one of your
    Pokemon to one of the opponent's. Generic scoring treats the ability as
    undifferentiated "free value"; the source/target choice is what makes it
    strong (heal the 320 HP wall, finish a damaged attacker).

No public agent code, weights, or decision logic is used. The deck list itself
is mined replay data (see agent/meta_decks.py).
"""

from __future__ import annotations

from domain_policy import (
    DomainPolicy, _CARDS, _prize, _damage, _get_card,
    _OK, _ACTIVE, _BENCH,
)

if _OK:
    from cg.api import (
        AreaType, OptionType, SelectContext, EnergyType, CardType,
        to_observation_class,
    )

from meta_decks import GRIMMSNARL as GRIMMSNARL_DECK

# --- the deck's cards ------------------------------------------------------
IMPIDIMP = 646          # Marnie's Impidimp     70 HP  basic
MORGREM = 647           # Marnie's Morgrem     100 HP  stage 1
GRIMMSNARL_EX = 648     # Marnie's Grimmsnarl ex 320 HP stage 2, Punk Up
MUNKIDORI = 112         # Adrena-Brain: move up to 3 damage counters
SNORUNT = 860
FROSLASS = 104          # Freezing Shroud: 1 counter on each Ability Pokemon
SPIKEMUTH_GYM = 1259    # search a Marnie's Pokemon each turn
PETREL = 1219           # search any Trainer
LILLIE_DET = 1227       # shuffle hand, draw 6 (8 if 6 prizes left)
BOSS_ORDERS = 1182
RARE_CANDY = 1079
POFFIN = 1086           # 2 basics with <=70 HP onto bench
POKE_PAD = 1152
NIGHT_STRETCHER = 1097
UNFAIR_STAMP = 1080
DARK_ENERGY = 7

MARNIES = {IMPIDIMP, MORGREM, GRIMMSNARL_EX}
SHADOW_BULLET = 937     # 180 + 30 to a benched Pokemon, cost {D}{D}
SHADOW_BULLET_BENCH = 30
_DARK = getattr(EnergyType, "DARKNESS", 7) if _OK else 7

# --- tunable priorities ----------------------------------------------------
# Defaults reproduce the validated policy exactly. A sweep writes overrides to
# grimm_w.json (searched next to this module and in the Kaggle agent dir), so
# tuning never requires editing code.
WEIGHTS = {
    # trainers / evolution
    "rare_candy_combo": 26000.0, "evolve_grimmsnarl": 25000.0,
    "poffin_early": 21000.0, "poffin_late": 12000.0,
    "spikemuth": 18000.0, "unfair_stamp": 15000.0, "poke_pad": 13000.0,
    "petrel_combo": 12000.0, "petrel_plain": 9000.0, "night_stretcher": 10000.0,
    "lillie_thin": 11000.0, "lillie_wide": 4000.0, "lillie_hold_combo": 1000.0,
    # Punk Up / energy routing
    "grimm_needs_energy": 9800.0, "grimm_active_bonus": 300.0,
    "grimm_surplus_bench": 8300.0, "grimm_surplus_active": 8100.0,
    "morgrem_needs": 9000.0, "morgrem_surplus": 7800.0,
    "munkidori_needs": 9400.0, "munkidori_surplus": 6500.0,
    "impidimp_needs": 8600.0, "impidimp_surplus": 7000.0,
    # targeting
    "snipe_finish": 50000.0, "snipe_prize_bonus": 1000.0,
    "target_prize": 300.0, "target_damaged": 400.0, "target_bench_ex": 250.0,
    "heal_per_counter": 10.0, "heal_grimm_bonus": 400.0, "heal_active_bonus": 150.0,
    "gust_kill": 40000.0, "gust_prize": 200.0,
}

try:  # optional sweep overrides
    import json as _json
    import os as _os
    for _p in ("grimm_w.json",
               _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "grimm_w.json"),
               "/kaggle_simulations/agent/grimm_w.json"):
        if _os.path.exists(_p):
            WEIGHTS.update(_json.load(open(_p)))
            break
except Exception:
    pass

W = WEIGHTS


def _dark_count(mon):
    """Basic {D} energy attached (Shadow Bullet costs two)."""
    try:
        return sum(1 for e in (mon.energies or []) if e == _DARK)
    except Exception:
        return len(getattr(mon, "energies", None) or [])


def _missing_hp(mon):
    try:
        return max(0, (mon.maxHp or 0) - (mon.hp or 0))
    except Exception:
        return 0


class GrimmsnarlPolicy(DomainPolicy):
    """DomainPolicy plus the Grimmsnarl-specific reads described above."""

    # ---- board queries ----
    def _hand_ids(self):
        try:
            return [c.id for c in (self.myp.hand or []) if c is not None]
        except Exception:
            return []

    def _in_play_ids(self):
        return [p.id for p in self._my_board() if p is not None]

    def _rare_candy_ready(self):
        """Holding Grimmsnarl ex with a non-fresh Impidimp already in play."""
        hand = self._hand_ids()
        if GRIMMSNARL_EX not in hand:
            return False
        for p in self._my_board():
            if p is not None and p.id == IMPIDIMP and not getattr(p, "appearThisTurn", False):
                return True
        return False

    def _have_grimmsnarl_in_play(self):
        return any(p is not None and p.id == GRIMMSNARL_EX for p in self._my_board())

    # ---- option scoring ----
    def _score(self, o):
        t = o.type
        cid = getattr(o, "cardId", None)

        if t == OptionType.PLAY:
            # Rare Candy is the deck. Skipping Morgrem puts a 320 HP attacker up
            # a full turn early AND fires Punk Up for five Energy at once.
            if cid == RARE_CANDY and self._rare_candy_ready():
                return W["rare_candy_combo"]
            # Poffin only hits <=70 HP basics -- Impidimp and Snorunt. It is at
            # its best on the opening turns, when the bench is still empty.
            if cid == POFFIN:
                empty = sum(1 for p in self.myp.bench if p is None) if self.myp.bench else 0
                return W["poffin_early"] if (self.st.turn <= 4 or empty >= 2) else W["poffin_late"]
            # A free Marnie's tutor every turn, for both players -- but we are
            # the deck built to abuse it.
            if cid == SPIKEMUTH_GYM:
                return W["spikemuth"]
            if cid == UNFAIR_STAMP:
                return W["unfair_stamp"]
            if cid == POKE_PAD:
                return W["poke_pad"]
            if cid == PETREL:
                # Fetch whatever the line is missing (usually Rare Candy).
                return W["petrel_combo"] if GRIMMSNARL_EX in self._hand_ids() else W["petrel_plain"]
            if cid == NIGHT_STRETCHER:
                return W["night_stretcher"]
            if cid == LILLIE_DET:
                # Shuffles the hand away: only worth it when the hand is thin,
                # and never while holding the Rare Candy combo.
                if self._rare_candy_ready():
                    return W["lillie_hold_combo"]
                return W["lillie_thin"] if len(self._hand_ids()) <= 3 else W["lillie_wide"]

        # Evolving INTO Grimmsnarl ex triggers Punk Up -- always the best evolve.
        if t == OptionType.EVOLVE and cid == GRIMMSNARL_EX:
            return W["evolve_grimmsnarl"]

        return super()._score(o)

    # ---- Punk Up / manual energy routing ----
    def _score_attach(self, o):
        """Two {D} powers Shadow Bullet. Fill the attacker to exactly two first,
        then bank the rest on a second Marnie's line so a knock-out on our
        Active does not reset the clock. Munkidori needs one {D} of its own for
        Adrena-Brain (Punk Up cannot reach it -- it is not a Marnie's Pokemon).
        """
        try:
            mon = _get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.me)
            if mon is None:
                return super()._score_attach(o)
            have = _dark_count(mon)
            active = (o.inPlayArea == _ACTIVE)

            if mon.id == GRIMMSNARL_EX:
                if have < 2:
                    return W["grimm_needs_energy"] + (W["grimm_active_bonus"] if active else 0.0)
                return W["grimm_surplus_active"] if active else W["grimm_surplus_bench"]
            if mon.id == MORGREM:
                # Pre-loading the backup line: it evolves into the next wall.
                return W["morgrem_needs"] if have < 2 else W["morgrem_surplus"]
            if mon.id == MUNKIDORI:
                return W["munkidori_needs"] if have < 1 else W["munkidori_surplus"]
            if mon.id == IMPIDIMP:
                return W["impidimp_needs"] if have < 2 else W["impidimp_surplus"]
        except Exception:
            pass
        return super()._score_attach(o)

    # ---- targeting: snipe, damage-counter movement, gust ----
    def _score_card(self, o):
        try:
            ctx = self.ctx
            pidx = getattr(o, "playerIndex", self.me)
            mon = _get_card(self.obs, o.area, o.index, pidx)

            # Taking damage counters OFF one of ours: pull them off the wall we
            # most want to keep alive, measured against the biggest hit the
            # opponent can currently make.
            if ctx in (SelectContext.REMOVE_DAMAGE_COUNTER,
                       SelectContext.DETACH_FROM) and pidx == self.me and mon is not None:
                if _missing_hp(mon) <= 0:
                    return -100.0
                score = min(_missing_hp(mon), 30) * W["heal_per_counter"]
                if mon.id == GRIMMSNARL_EX:
                    score += W["heal_grimm_bonus"]
                if o.area == _ACTIVE:
                    score += W["heal_active_bonus"]
                return score

            # Putting damage counters ON the opponent (Adrena-Brain) or picking
            # Shadow Bullet's benched victim: 30 damage that completes a KO is
            # worth a prize; otherwise set up the next one.
            if pidx != self.me and mon is not None and o.area in (_ACTIVE, _BENCH):
                if ctx in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY,
                           SelectContext.DAMAGE, SelectContext.EFFECT_TARGET):
                    hp = mon.hp or 0
                    if 0 < hp <= SHADOW_BULLET_BENCH:
                        return W["snipe_finish"] + _prize(mon.id) * W["snipe_prize_bonus"]
                    score = _prize(mon.id) * W["target_prize"]
                    score += W["target_damaged"] * (1.0 - hp / max(mon.maxHp or hp or 1, 1))
                    # Softening a fresh ex on the bench pays off next turn.
                    if o.area == _BENCH and _prize(mon.id) >= 2:
                        score += W["target_bench_ex"]
                    return score

                # Boss's Orders: drag up something we can knock out now.
                if ctx in (SelectContext.TO_ACTIVE, SelectContext.SWITCH) and o.area == _BENCH:
                    act = self.myp.active[0] if self.myp.active else None
                    if act is not None:
                        dmg = _damage(act.id, len(act.energies or []), mon.id)
                        if dmg >= (mon.hp or 0):
                            return W["gust_kill"] + _prize(mon.id) * W["snipe_prize_bonus"]
                    return _prize(mon.id) * W["gust_prize"] + W["target_prize"] * (
                        1.0 - (mon.hp or 1) / max(mon.maxHp or 1, 1))

            # Opening lead: we want Impidimp, the thing Rare Candy evolves.
            if ctx == SelectContext.SETUP_ACTIVE_POKEMON and mon is not None:
                return {IMPIDIMP: 900.0, SNORUNT: 500.0,
                        MUNKIDORI: 400.0}.get(mon.id, 100.0)
            if ctx == SelectContext.SETUP_BENCH_POKEMON and mon is not None:
                return {IMPIDIMP: 900.0, MUNKIDORI: 700.0,
                        SNORUNT: 400.0}.get(mon.id, 100.0)

            # Searching deck/discard (Poffin, Poke Pad, Spikemuth, Stretcher):
            # take the piece the combo is missing.
            if ctx in (SelectContext.TO_HAND, SelectContext.TO_BENCH,
                       SelectContext.LOOK, SelectContext.TO_FIELD) and pidx == self.me:
                if mon is not None:
                    return self._search_value(mon.id)
        except Exception:
            pass
        return super()._score_card(o)

    def _search_value(self, cid):
        """What the deck most wants to find, given what it already has."""
        hand = self._hand_ids()
        play = self._in_play_ids()
        if cid == GRIMMSNARL_EX:
            return 950.0 if GRIMMSNARL_EX not in hand else 300.0
        if cid == RARE_CANDY:
            return 900.0 if (GRIMMSNARL_EX in hand and RARE_CANDY not in hand) else 350.0
        if cid == IMPIDIMP:
            return 850.0 if play.count(IMPIDIMP) < 2 else 250.0
        if cid == MUNKIDORI:
            return 600.0 if MUNKIDORI not in play else 200.0
        if cid == MORGREM:
            return 500.0 if MORGREM not in hand else 200.0
        if cid == DARK_ENERGY:
            return 450.0
        if cid == SNORUNT:
            return 300.0
        if cid == FROSLASS:
            return 350.0 if SNORUNT in play else 150.0
        return 200.0

    # ---- discard: never pitch the combo ----
    def _score_discard(self, o):
        try:
            card = _get_card(self.obs, o.area, o.index, self.me)
            if card is not None:
                if card.id in (GRIMMSNARL_EX, RARE_CANDY):
                    return -900.0
                if card.id in (IMPIDIMP, MUNKIDORI, MORGREM):
                    return -400.0
        except Exception:
            pass
        return super()._score_discard(o)


def grimmsnarl_agent(obs_dict, deck=None):
    """Entry point: agent(obs_dict, deck) -> list[int]. Never raises."""
    deck = list(deck or GRIMMSNARL_DECK)
    try:
        if obs_dict.get("select") is None:
            return deck
        obs = to_observation_class(obs_dict)
        n = len(obs.select.option)
        if n == 0:
            return []
        mc = obs.select.maxCount or 1
        ranked = [i for i in GrimmsnarlPolicy(obs).choose() if 0 <= i < n]
        return ranked[:mc] if ranked else list(range(min(mc, n)))
    except Exception:
        try:
            sel = obs_dict.get("select")
            if sel is None:
                return deck
            n = len(sel.get("option") or [])
            mc = sel.get("maxCount", 1) or 1
            return list(range(min(mc, n))) if n else []
        except Exception:
            return deck
