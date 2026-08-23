"""Card-specific policy for the elite Mega Lucario ex list (elite_07).

Why this deck: mined from 1,138 games between players rated >= 1150, this list
won 75.9% of 83 games -- the best in the elite pool -- while the Alakazam
archetype our fork plays won 48.8% of 170. The same pilot (Majkel1337) appears on
lists at 75.9%, 52.8% and 36.7%, so the spread is the deck, not just the player.

Why a NEW policy rather than pointing the fork at it: the fork's 69 weights are
keyed to Abra/Kadabra/Alakazam card IDs. Aimed at another archetype it degrades to
a generic agent, and our generic policy measured 4.2% against the fork. A strong
deck in weak hands loses to a middling deck in tuned hands -- that is exactly what
the elite_pilot experiment showed.

Deck engine:
  Mega Lucario ex  340 HP, retreat 2.
      Mega Brave  270 damage, but CANNOT be used again on the following turn.
      Aura Jab    130 damage, and attaches up to 3 Basic {F} Energy from the
                  discard pile to the Bench.
  So the deck alternates: Mega Brave for the knockout, Aura Jab on the off-turn to
  rebuild board energy while still hitting for 130. Aura Jab is therefore worth
  far more than its damage suggests, and the search's damage-only attack ranking
  systematically undervalues it.

  Solrock/Lunatone are a pair: Cosmic Beam (70) does NOTHING without Lunatone
  benched, so Lunatone has value purely as a bench enabler.
  Makuhita -> Hariyama gives Wild Press 210 with 70 recoil: a finisher that trades
  its own HP, sensible only when it wins a prize race.

All weights are tunable; evolution writes overrides to lucario_w.json, matching
the pattern that let us tune the fork without editing its code.
"""
from __future__ import annotations

import json
import os

from domain_policy import (  # noqa: F401
    DomainPolicy, SelectContext, OptionType, AreaType, CardType,
    _CARDS, _get_card, _prize, _ACTIVE, _BENCH, _ADMG, _ACOST,
    evaluate_board, domain_agent,
)

# AreaType.HAND. Options for PLAY/EVOLVE carry no cardId; the engine identifies
# the card by option.index into the hand, so this constant is what makes every
# card-specific branch below reachable.
_HAND = getattr(AreaType, "HAND", 2)

# --- the deck ---
F_ENERGY = 6
MEGA_LUCARIO = 678
RIOLU = 677
SOLROCK = 676
LUNATONE = 675
MAKUHITA = 673
HARIYAMA = 674
ULTRA_BALL = 1121
PREMIUM_POWER_PRO = 1141
FIGHTING_GONG = 1142
POKE_PAD = 1152
JUDGE = 1213
LILLIES = 1227
WALLYS = 1229
SWITCH = 1123
BOSS_ORDERS = 1182
HEROS_CAPE = 1159

ATK_MEGA_BRAVE = "mega brave"
ATK_AURA_JAB = "aura jab"

W = {
    # board building -- Mega Lucario is the whole deck
    "play_riolu": 21000.0,
    "play_lucario": 24000.0,
    "evolve_lucario": 26000.0,
    "play_solrock": 9000.0,
    "play_lunatone": 11000.0,      # enables Cosmic Beam; useless card, useful bench
    "play_makuhita": 7000.0,
    "evolve_hariyama": 12000.0,
    "play_bench_penalty": 3000.0,
    # finders
    "ultra_ball_need": 19000.0,    # discard 2 to fetch the attacker
    "ultra_ball_have": 4000.0,
    "poke_pad": 12000.0,
    "fighting_gong": 15000.0,
    "premium_power_pro": 16000.0,
    "wallys": 13000.0,
    # draw / disruption, conditioned on hand size
    "judge_thin": 12000.0,
    "judge_wide": 2500.0,
    "lillies_thin": 15000.0,
    "lillies_wide": 3000.0,
    # tempo
    "boss_kill": 26000.0,
    "boss_plain": 6000.0,
    "switch_trapped": 14000.0,
    "switch_idle": 2000.0,
    "heros_cape": 12000.0,
    # energy: Mega Brave needs a full board, Aura Jab refills from discard
    "attach_lucario_active": 14000.0,
    "attach_lucario_bench": 9000.0,
    "attach_generic": 6000.0,
    # attacks -- the alternation is the engine
    "mega_brave_kill": 30000.0,
    "mega_brave_plain": 8000.0,
    "aura_jab_accel": 12000.0,     # worth more than 130 damage when energy is short
    "aura_jab_plain": 7000.0,
    "wild_press_kill": 15000.0,
    "wild_press_plain": 1500.0,    # 70 recoil: only when it wins the race
    "cosmic_beam_ready": 6000.0,
    "attack_other": 1000.0,
}

for _p in ("lucario_w.json",
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "lucario_w.json"),
           "/kaggle_simulations/agent/lucario_w.json"):
    try:
        if os.path.exists(_p):
            W.update(json.load(open(_p)))
            break
    except Exception:
        pass


def _atk_name(aid):
    try:
        from cg.api import all_attack
        global _ATKN
        if "_ATKN" not in globals():
            _ATKN = {a.attackId: (a.name or "").strip().lower() for a in all_attack()}
        return _ATKN.get(aid, "")
    except Exception:
        return ""


class LucarioPolicy(DomainPolicy):
    def _hand_ids(self):
        try:
            return [c.id for c in (self.myp.hand or []) if c is not None]
        except Exception:
            return []

    def _board_ids(self):
        return [p.id for p in self._my_board() if p is not None]

    def _my_active(self):
        try:
            return self.myp.active[0] if self.myp.active else None
        except Exception:
            return None

    def _active_energy(self):
        a = self._my_active()
        return len(a.energies or []) if a else 0

    def _f_in_discard(self):
        try:
            return sum(1 for c in (self.myp.discard or [])
                       if c is not None and c.id == F_ENERGY)
        except Exception:
            return 0

    def _have_lucario(self):
        return MEGA_LUCARIO in self._hand_ids() or MEGA_LUCARIO in self._board_ids()

    def _lunatone_benched(self):
        try:
            return any(p is not None and p.id == LUNATONE for p in (self.myp.bench or []))
        except Exception:
            return False

    def _card_id(self, o):
        """Resolve the card an option refers to.

        `Option.cardId` is None for PLAY/EVOLVE — the engine identifies the card
        by `option.index` into the HAND area, which is how the fork does it:
            card = get_card(obs, AreaType.HAND, option.index, state.yourIndex)
        Reading `o.cardId` directly made every card-specific branch below fall
        through to generic scoring, which is why this policy previously produced
        results IDENTICAL to the generic one.
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
        hand = self._hand_ids()
        thin = len(hand) <= 3

        if t == OptionType.EVOLVE:
            if cid == MEGA_LUCARIO:
                return W["evolve_lucario"]
            if cid == HARIYAMA:
                return W["evolve_hariyama"]

        if t == OptionType.PLAY:
            if cid == MEGA_LUCARIO:
                return W["play_lucario"]
            if cid == RIOLU:
                return W["play_riolu"]
            if cid == LUNATONE:
                return W["play_lunatone"]
            if cid == SOLROCK:
                return W["play_solrock"]
            if cid == MAKUHITA:
                return W["play_makuhita"]
            if cid == ULTRA_BALL:
                return W["ultra_ball_have"] if self._have_lucario() else W["ultra_ball_need"]
            if cid == POKE_PAD:
                return W["poke_pad"]
            if cid == FIGHTING_GONG:
                return W["fighting_gong"]
            if cid == PREMIUM_POWER_PRO:
                return W["premium_power_pro"]
            if cid == WALLYS:
                return W["wallys"]
            if cid == HEROS_CAPE:
                return W["heros_cape"]
            if cid == SWITCH:
                a = self._my_active()
                hurt = a is not None and (a.maxHp or 0) - (a.hp or 0) >= 150
                return W["switch_trapped"] if hurt else W["switch_idle"]
            if cid == BOSS_ORDERS:
                return W["boss_kill"] if (self.plan and self.plan[3]) else W["boss_plain"]
            if cid == JUDGE:
                return W["judge_thin"] if thin else W["judge_wide"]
            if cid == LILLIES:
                return W["lillies_thin"] if thin else W["lillies_wide"]

        if t == OptionType.ATTACH:
            try:
                mon = _get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.me)
                if mon is not None and mon.id == MEGA_LUCARIO:
                    return W["attach_lucario_active"] if o.inPlayArea == _ACTIVE \
                        else W["attach_lucario_bench"]
            except Exception:
                pass
            return W["attach_generic"]

        if t == OptionType.ATTACK:
            aid = getattr(o, "attackId", None)
            nm = _atk_name(aid)
            killing = bool(self.plan and self.plan[3] and self.plan[1] == aid)
            if nm == ATK_MEGA_BRAVE:
                return W["mega_brave_kill"] if killing else W["mega_brave_plain"]
            if nm == ATK_AURA_JAB:
                # refilling from a stocked discard is the reason this attack exists
                return W["aura_jab_accel"] if self._f_in_discard() >= 2 else W["aura_jab_plain"]
            if nm == "wild press":
                return W["wild_press_kill"] if killing else W["wild_press_plain"]
            if nm == "cosmic beam":
                return W["cosmic_beam_ready"] if self._lunatone_benched() else 1.0
            return W["attack_other"]

        return super()._score(o)


def rank_options(obs):
    try:
        return LucarioPolicy(obs).choose()
    except Exception:
        n = len(obs.select.option) if obs.select else 0
        return list(range(n))


def lucario_agent(obs_dict, deck):
    from cg.api import to_observation_class
    try:
        if obs_dict.get("select") is None:
            return list(deck)
        obs = to_observation_class(obs_dict)
        n = len(obs.select.option)
        if n == 0:
            return []
        mc = obs.select.maxCount or 1
        r = [i for i in LucarioPolicy(obs).choose() if 0 <= i < n]
        return r[:mc] if r else list(range(min(mc, n)))
    except Exception:
        return domain_agent(obs_dict, deck)
