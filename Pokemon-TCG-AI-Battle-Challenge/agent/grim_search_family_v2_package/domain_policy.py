"""
Our own card-data-driven DOMAIN-KNOWLEDGE policy for the PTCG AI Battle.

This is NOT a copy of anyone's agent. It reads the real card catalog
(all_card_data / all_attack) and scores every legal option from first principles
of the game: prize trade (ex=2 / megaEx=3 prizes), weakness x2 damage, true
per-attack damage/cost, knock-out detection, energy routed to the planned
attacker, evolution/board development, and card-advantage engines. Because it is
data-driven it works for ANY deck (we pair it with a strong meta deck), and it's
our implementation end-to-end.

Entry point: `domain_agent(obs_dict) -> list[int]`. It ranks the legal options
and returns the top `maxCount`. Never crashes: any failure is caught and falls
back to a legal move. Requires cg.api (bundled at eval).
"""

from __future__ import annotations

try:
    from cg.api import (
        AreaType, OptionType, SelectContext, EnergyType, CardType,
        all_card_data, all_attack, to_observation_class,
    )
    _OK = True
except Exception:
    _OK = False

if _OK:
    _CARDS = {c.cardId: c for c in all_card_data()}
    _ADMG = {a.attackId: (a.damage or 0) for a in all_attack()}
    _ACOST = {a.attackId: len(a.energies or []) for a in all_attack()}
else:
    _CARDS, _ADMG, _ACOST = {}, {}, {}

_ACTIVE = getattr(AreaType, "ACTIVE", 4) if _OK else 4
_BENCH = getattr(AreaType, "BENCH", 5) if _OK else 5


# --- card knowledge --------------------------------------------------------
def _prize(cid):
    c = _CARDS.get(cid)
    if c is None:
        return 1
    return 3 if getattr(c, "megaEx", False) else 2 if getattr(c, "ex", False) else 1


def _best_attack(cid, energy):
    """(attackId, damage, cost) of the best attack this card can pay for."""
    c = _CARDS.get(cid)
    if c is None:
        return None
    best = None
    for aid in (getattr(c, "attacks", None) or []):
        cost = _ACOST.get(aid, 99)
        if cost <= energy:
            dmg = _ADMG.get(aid, 0)
            if best is None or dmg > best[1]:
                best = (aid, dmg, cost)
    return best


def _weak(cid):
    c = _CARDS.get(cid)
    return getattr(c, "weakness", None) if c else None


def _etype(cid):
    c = _CARDS.get(cid)
    return getattr(c, "energyType", None) if c else None


def _get_card(obs, area, index, pidx):
    try:
        p = obs.current.players[pidx]
        if area == _ACTIVE:
            return p.active[index]
        if area == _BENCH:
            return p.bench[index]
        if area == getattr(AreaType, "HAND", 2):
            return p.hand[index]
        if area == getattr(AreaType, "DISCARD", 3):
            return p.discard[index]
        if area == getattr(AreaType, "DECK", 1):
            return obs.select.deck[index]
        if area == getattr(AreaType, "STADIUM", 7):
            return obs.current.stadium[index]
    except Exception:
        pass
    return None


def _damage(att_cid, att_energy, def_cid):
    """Best damage att_cid deals to def_cid (weakness-aware)."""
    ba = _best_attack(att_cid, att_energy)
    if ba is None:
        return 0
    dmg = ba[1]
    w = _weak(def_cid)
    if w is not None and w == _etype(att_cid):
        dmg *= 2
    return dmg


class DomainPolicy:
    """Scores every legal option using game-theoretic domain knowledge."""

    def __init__(self, obs):
        self.obs = obs
        self.st = obs.current
        self.sel = obs.select
        self.ctx = self.sel.context
        self.me = self.st.yourIndex
        self.op = 1 - self.me
        self.myp = self.st.players[self.me]
        self.opp = self.st.players[self.op]
        self.plan = None          # (board_index, attackId, target_board_index, ko)
        if self.ctx == SelectContext.MAIN:
            self._plan_attack()

    # ---- board helpers ----
    def _my_board(self):
        return list(self.myp.active) + list(self.myp.bench)

    def _op_board(self):
        return list(self.opp.active) + list(self.opp.bench)

    def _can_retreat_now(self):
        """True if a RETREAT option is legal right now (this decision)."""
        return any(o.type == OptionType.RETREAT for o in self.sel.option)

    # ---- attack planning: best (attacker, attack, target) ----
    # NOTE: only the ACTIVE (index 0) can attack. A benched attacker (ai != 0)
    # is only reachable by retreating into it first -- a SEPARATE decision the
    # engine will ask about later this turn. We still evaluate bench attackers
    # so `_score_attach`/retreat can steer us toward switching in a stronger
    # one when it clearly beats the current active.
    def _plan_attack(self):
        best_score, best = -1e18, None
        my = self._my_board()
        opp = self._op_board()
        for ai, atk in enumerate(my):
            if atk is None:
                continue
            energy = len(atk.energies or [])
            # allow one attach if it's still available
            eff_energy = energy + (0 if self.st.energyAttached else 1)
            ba = _best_attack(atk.id, eff_energy)
            if ba is None:
                continue
            aid = ba[0]
            for ti, tgt in enumerate(opp):
                if tgt is None:
                    continue
                if ti != 0:
                    # can only hit bench with a gust effect; approximate: skip
                    continue
                dmg = _damage(atk.id, eff_energy, tgt.id)
                ko = dmg >= tgt.hp
                score = 0.0
                if ko:
                    score += _prize(tgt.id) * 100000  # taking prizes wins games
                    if len(self.opp.prize) <= _prize(tgt.id):
                        score += 500000              # lethal / game-winning
                else:
                    score += dmg * (tgt.hp and 100.0 / max(tgt.hp, 1))  # % of KO
                score += (200 if ai == 0 else 0)     # prefer active attacker
                score -= ba[2] * 5                    # cheaper attacks slightly better
                if score > best_score:
                    best_score, best = score, (ai, aid, ti, ko)
        self.plan = best

    # ---- opponent threat: can they KO my active next turn? ----
    def _lethal_threat(self):
        act = self.myp.active[0] if self.myp.active else None
        if act is None:
            return False
        mx = 0
        for p in self._op_board():
            if p is None:
                continue
            d = _damage(p.id, len(p.energies or []) + 1, act.id)
            mx = max(mx, d)
        return mx >= act.hp

    # ---- score one option ----
    def _score(self, o):
        t = o.type
        # NUMBER: usually "how many to draw" -> take more
        if t == OptionType.NUMBER:
            return float(getattr(o, "number", 0))
        if t == OptionType.YES:
            return 100.0 if self.ctx == getattr(SelectContext, "IS_FIRST", -99) else 5.0
        if t == OptionType.NO:
            return 0.0
        if t == OptionType.ABILITY:
            return 30000.0                          # free value: draw/search engines
        if t == OptionType.PLAY:
            card = _get_card(self.obs, AreaType.HAND, o.index, self.me)
            if card is not None and _CARDS.get(card.id) and getattr(_CARDS[card.id], "cardType", None) == CardType.POKEMON:
                return 20000.0                       # develop the board (basics/benched)
            return 6000.0                            # trainers (draw/search/boss)
        if t == OptionType.EVOLVE:
            p = _get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.me)
            return 9000.0 + (len(p.energies or []) if p else 0)  # evolve into attackers
        if t == OptionType.ATTACH:
            return self._score_attach(o)
        if t == OptionType.RETREAT:
            # retreat when the active is threatened with lethal, OR the plan
            # picked a stronger benched attacker (switch INTO that attacker).
            if self._lethal_threat() and len(self.myp.bench) > 0:
                return 3000.0
            if self.plan and self.plan[0] != 0:
                return 2500.0
            return -5.0
        if t == OptionType.ATTACK:
            if self.plan and getattr(o, "attackId", None) == self.plan[1]:
                return 1200.0
            return 1000.0
        if t in (OptionType.CARD, OptionType.TOOL_CARD, OptionType.ENERGY_CARD, OptionType.ENERGY):
            return self._score_card(o)
        if t == OptionType.DISCARD:
            return self._score_discard(o)
        if t == OptionType.END:
            return -100000.0
        return 1.0

    def _planned_attacker_index(self):
        return self.plan[0] if self.plan else 0

    def _score_attach(self, o):
        # route energy to the planned attacker (active preferred)
        base = 8000.0
        try:
            area, idx = o.inPlayArea, o.inPlayIndex
            board_index = idx if area == _ACTIVE else idx + len(self.myp.active)
            if board_index == self._planned_attacker_index():
                base += 400.0
            if area == _ACTIVE:
                base += 50.0
        except Exception:
            pass
        return base

    def _readiness_score(self, cid):
        """How good a Pokémon is to lead with: cheapest usable attack wins,
        big HP is a tiebreaker. Deck-agnostic (no hardcoded card names)."""
        c = _CARDS.get(cid)
        if c is None:
            return 0.0
        best = None
        for aid in (getattr(c, "attacks", None) or []):
            cost = _ACOST.get(aid, 99)
            dmg = _ADMG.get(aid, 0)
            if dmg <= 0:
                continue
            eff = dmg / max(cost, 1)
            if best is None or eff > best:
                best = eff
        score = (best or 0.0) * 10.0
        score += (getattr(c, "hp", 0) or 0) * 0.2
        if getattr(c, "ex", False) or getattr(c, "megaEx", False):
            score -= 15.0  # big-prize attackers are juicier gust targets; lead cautiously
        return score

    def _score_card(self, o):
        card = _get_card(self.obs, o.area, o.index, getattr(o, "playerIndex", self.me))
        if card is None:
            return 0.0
        pidx = getattr(o, "playerIndex", self.me)
        # targeting an ENEMY (attack target / gust): prefer the planned target,
        # else the one closest to a KO / highest prize value
        if pidx != self.me and o.area in (_ACTIVE, _BENCH):
            score = _prize(card.id) * 100.0
            score += (200.0 if o.area == _ACTIVE else 0.0)
            hp = getattr(card, "hp", 1) or 1
            score += 300.0 * (1.0 - hp / max(getattr(_CARDS.get(card.id), "hp", hp) or hp, 1))
            return score
        # choosing which Basic Pokémon starts active/bench at game setup:
        # prefer the one that can swing soonest (best damage-per-energy).
        if self.ctx in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON):
            return self._readiness_score(card.id)
        # choosing our own card (e.g. from hand/deck to keep): favor Pokémon/energy
        ct = getattr(_CARDS.get(card.id), "cardType", None)
        return 50.0 if ct is not None else 10.0

    def _score_discard(self, o):
        # discard the least valuable: prefer basic energy / duplicates, keep attackers
        card = _get_card(self.obs, o.area, o.index, self.me)
        if card is None:
            return -50.0
        c = _CARDS.get(card.id)
        # basic energy is the safest discard
        if c is not None and getattr(c, "cardType", None) == CardType.BASIC_ENERGY:
            return -10.0
        return -60.0

    def choose(self):
        opts = self.sel.option
        if not opts:
            return []

        # HARD OVERRIDE 1: a planned lethal attack is legally available THIS
        # decision -> take it immediately. Never let ranking/search risk
        # passing up a game-ending KO (loss analysis found this was the #1
        # cause of losses when only *scored* highly instead of guaranteed).
        if self.ctx == SelectContext.MAIN and self.plan and self.plan[3]:
            for i, o in enumerate(opts):
                if o.type == OptionType.ATTACK and getattr(o, "attackId", None) == self.plan[1]:
                    rest = [j for j in range(len(opts)) if j != i]
                    rest.sort(key=lambda j: self._score(opts[j]), reverse=True)
                    return [i] + rest

        # HARD OVERRIDE 2: my active is under confirmed lethal threat and a
        # RETREAT is legal -> retreat immediately. Never risk losing the
        # attacker to a ranking that got outvoted by setup actions.
        if self.ctx == SelectContext.MAIN and self._lethal_threat():
            for i, o in enumerate(opts):
                if o.type == OptionType.RETREAT:
                    rest = [j for j in range(len(opts)) if j != i]
                    rest.sort(key=lambda j: self._score(opts[j]), reverse=True)
                    return [i] + rest

        scored = sorted(range(len(opts)), key=lambda i: self._score(opts[i]), reverse=True)
        return scored


def rank_options(obs):
    """Rank the legal options of an already-built Observation dataclass (as
    returned by cg's search_step). Used to drive a search rollout with domain
    knowledge instead of generic type-priority."""
    try:
        return DomainPolicy(obs).choose()
    except Exception:
        n = len(obs.select.option) if obs.select else 0
        return list(range(n))


def evaluate_board(cur, me):
    """Domain-knowledge board value from `me`'s perspective, for scoring the
    end of a simulated turn (search leaf evaluation). Higher = better for me."""
    if cur.result >= 0:
        return 1e9 if cur.result == me else (-1e9 if cur.result == (1 - me) else 0.0)
    mp, op = cur.players[me], cur.players[1 - me]
    val = (len(op.prize) - len(mp.prize)) * 10000.0

    def board_hp(p):
        return sum(x.hp for x in p.active if x) + sum(x.hp for x in p.bench if x)

    val += board_hp(mp) * 0.3 - board_hp(op) * 1.0
    for p in list(mp.active) + list(mp.bench):
        if p:
            val += 200.0 * len(p.energies or [])
    val += 300.0 * (len(mp.active) + len(mp.bench))

    # opponent threat: assume +1 energy next turn, weakness-aware
    my_active = mp.active[0] if mp.active else None
    threat = 0
    for p in list(op.active) + list(op.bench):
        if p is None:
            continue
        d = _damage(p.id, len(p.energies or []) + 1, my_active.id) if my_active else 0
        threat = max(threat, d)
    if my_active:
        if threat >= my_active.hp:
            val -= (4000.0 if (_CARDS.get(my_active.id) and
                               (_CARDS[my_active.id].ex or _CARDS[my_active.id].megaEx))
                    else 2000.0)
        else:
            val -= threat * 1.5

    # my KO potential on their active
    op_active = op.active[0] if op.active else None
    if my_active and op_active:
        d = _damage(my_active.id, len(my_active.energies or []), op_active.id)
        if d >= op_active.hp:
            val += _prize(op_active.id) * 3000.0
    return val


def domain_agent(obs_dict, deck):
    """Entry point. `deck` is returned during deck-selection."""
    try:
        if obs_dict.get("select") is None:
            return list(deck)
        obs = to_observation_class(obs_dict)
        n = len(obs.select.option)
        if n == 0:
            return []
        mc = obs.select.maxCount or 1
        ranked = DomainPolicy(obs).choose()
        ranked = [i for i in ranked if 0 <= i < n]
        if not ranked:
            return list(range(min(mc, n)))
        return ranked[:mc]
    except Exception:
        try:
            sel = obs_dict.get("select")
            if sel is None:
                return list(deck)
            n = len(sel.get("option") or [])
            mc = sel.get("maxCount", 1) or 1
            return list(range(min(mc, n))) if n else []
        except Exception:
            return list(deck) if deck else [0]
