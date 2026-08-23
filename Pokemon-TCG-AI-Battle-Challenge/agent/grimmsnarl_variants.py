"""Ablation variants of our Grimmsnarl policy.

Each variant disables exactly one of the three deck-specific overrides so the
arena can attribute the win-rate delta to that component instead of to the
bundle. Research only -- the submission entry point uses the full policy.
"""

from __future__ import annotations

from domain_policy import DomainPolicy
from grimmsnarl_policy import GrimmsnarlPolicy, GRIMMSNARL_DECK

try:
    from cg.api import to_observation_class
except Exception:  # pragma: no cover
    to_observation_class = None


class NoSnipeTargeting(GrimmsnarlPolicy):
    """Drop Shadow Bullet snipe / Adrena-Brain targeting; keep the rest."""
    _score_card = DomainPolicy._score_card


class NoEnergyRouting(GrimmsnarlPolicy):
    """Drop Punk Up energy distribution; keep the rest."""
    _score_attach = DomainPolicy._score_attach


class NoTrainerPriority(GrimmsnarlPolicy):
    """Drop Rare Candy / Poffin / Spikemuth priorities; keep the rest.

    Compound: also drops the Grimmsnarl-ex evolve priority, since both live in
    the same _score override. Use NoTrainerOnly to separate them.
    """
    _score = DomainPolicy._score


class NoTrainerOnly(GrimmsnarlPolicy):
    """Clean ablation: keep the evolve priority, drop only the trainer scores."""

    def _score(self, o):
        from cg.api import OptionType as _OT
        from grimmsnarl_policy import GRIMMSNARL_EX as _GX, W as _W
        if o.type == _OT.EVOLVE and getattr(o, "cardId", None) == _GX:
            return _W["evolve_grimmsnarl"]
        return DomainPolicy._score(self, o)


class NoSetupLead(GrimmsnarlPolicy):
    """Keep snipe/heal targeting, but let the base pick the opening lead."""

    def _score_card(self, o):
        try:
            from cg.api import SelectContext as _SC
            if self.ctx in (_SC.SETUP_ACTIVE_POKEMON, _SC.SETUP_BENCH_POKEMON):
                return DomainPolicy._score_card(self, o)
        except Exception:
            pass
        return GrimmsnarlPolicy._score_card(self, o)


def _make(cls):
    def run(obs_dict, deck=None):
        deck = list(deck or GRIMMSNARL_DECK)
        try:
            if obs_dict.get("select") is None:
                return deck
            obs = to_observation_class(obs_dict)
            n = len(obs.select.option)
            if n == 0:
                return []
            mc = obs.select.maxCount or 1
            ranked = [i for i in cls(obs).choose() if 0 <= i < n]
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
    return run


class EnergyOnly(DomainPolicy):
    """Minimal policy: the deck-agnostic base plus ONLY Punk Up energy routing.

    The clean ablation panel showed energy routing is the single override that
    reliably pays (-5.9 rating when removed), while snipe targeting, setup lead
    and trainer priority were each neutral-to-negative.
    """
    _score_attach = GrimmsnarlPolicy._score_attach
    _hand_ids = GrimmsnarlPolicy._hand_ids
    _in_play_ids = GrimmsnarlPolicy._in_play_ids
    _rare_candy_ready = GrimmsnarlPolicy._rare_candy_ready
    _have_grimmsnarl_in_play = GrimmsnarlPolicy._have_grimmsnarl_in_play


energy_only_agent = _make(EnergyOnly)
no_snipe_agent = _make(NoSnipeTargeting)
no_energy_agent = _make(NoEnergyRouting)
no_trainer_agent = _make(NoTrainerPriority)
no_trainer_only_agent = _make(NoTrainerOnly)
no_setup_lead_agent = _make(NoSetupLead)
