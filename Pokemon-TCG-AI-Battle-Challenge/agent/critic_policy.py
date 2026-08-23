"""domain_policy with its hand-written leaf evaluation replaced by a learned critic.

`hybrid_agent` takes a `policy_module` and calls `policy_module.evaluate_board`
at every search leaf. That function is currently a hand-written formula
(prize_diff*10000 - opponent_hp + own_hp*0.3 + energy + presence) that nobody
ever validated. It is the prime suspect for why 40x more search bought nothing:
extra samples of a biased evaluator converge harder on the wrong answer.

Here the leaf value becomes a win-probability estimate from `value_critic.pt`,
trained on 1,529,127 board positions labelled by game outcome:

    MLP        AUC 0.8124   (early game, turn<=8: 0.7523)
    logistic   AUC 0.7239   (early game: 0.6671)

for reference, the published comparison point is a linear probe at AUC 0.818 and
a neural value head at 0.61 (discussion 713608).

Unlike the behaviour-cloning ranker -- which predicted teacher moves well but
PLAYED worse -- a better leaf compounds through search instead of drifting,
because lookahead corrects its errors rather than accumulating them.

Everything except evaluate_board is re-exported from domain_policy unchanged, so
candidate ranking and the hard overrides behave identically. If the critic fails
to load, evaluate_board falls straight back to the original formula.
"""
from __future__ import annotations

import os

# Re-export the whole domain policy surface hybrid_agent touches.
from domain_policy import (  # noqa: F401
    SelectContext, OptionType, DomainPolicy, rank_options, domain_agent,
    evaluate_board as _formula_eval,
)

_M = None
_STATS = None
_TORCH = None
_READY = False
_SCALE = float(os.environ.get("CRITIC_SCALE", "20000"))


def _load():
    global _M, _STATS, _TORCH, _READY
    if _READY:
        return _M is not None
    _READY = True
    try:
        import torch
        import torch.nn as nn
        _TORCH = torch
        here = os.path.dirname(os.path.abspath(__file__))
        path = None
        for c in (os.path.join(here, "value_critic.pt"),
                  "/kaggle_simulations/agent/value_critic.pt", "value_critic.pt"):
            if os.path.exists(c):
                path = c
                break
        if path is None:
            return False
        ck = torch.load(path, map_location="cpu", weights_only=False)
        d = len(ck["feats"])
        if ck["kind"] == "mlp":
            m = nn.Sequential(nn.Linear(d, 128), nn.ReLU(),
                              nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
            sd = {k.replace("net.", ""): v for k, v in ck["model"].items()}
            m.load_state_dict({f"{i}.{p}": sd[f"{i}.{p}"]
                               for i in (0, 2, 4) for p in ("weight", "bias")})
        else:
            m = nn.Sequential(nn.Linear(d, 1), nn.Flatten(0))
            m.load_state_dict(ck["model"])
        m.eval()
        _M, _STATS = m, ck
        return True
    except Exception:
        _M = None
        return False


def _feats(cur, me):
    """Must match tools/rank/extract_values.py exactly."""
    mp, op = cur.players[me], cur.players[1 - me]

    def mons(p):
        return [m for m in list(p.active or []) + list(p.bench or []) if m]

    my, oo = mons(mp), mons(op)
    a0 = mp.active[0] if mp.active else None
    a1 = op.active[0] if op.active else None
    myhp = sum((m.hp or 0) for m in my)
    ophp = sum((m.hp or 0) for m in oo)
    mye = sum(len(m.energies or []) for m in my)
    ope = sum(len(m.energies or []) for m in oo)
    mypz, oppz = len(mp.prize or []), len(op.prize or [])
    return [
        float(cur.turn), float(mypz), float(oppz), float(oppz - mypz),
        float(mp.handCount or 0), float(op.handCount or 0),
        float(mp.deckCount or 0), float(op.deckCount or 0),
        float(len(my)), float(len(oo)), float(myhp), float(ophp), float(myhp - ophp),
        float(mye), float(ope), float(mye - ope),
        float(a0.hp or 0) if a0 else 0.0, float(a1.hp or 0) if a1 else 0.0,
        float(a0.maxHp or 0) if a0 else 0.0, float(a1.maxHp or 0) if a1 else 0.0,
        float(len([m for m in (mp.bench or []) if m])),
        float(len([m for m in (op.bench or []) if m])),
        float(1 if cur.firstPlayer == me else 0),
        float(len(mp.discard or [])), float(len(op.discard or [])),
    ]


def evaluate_board(cur, me):
    """Learned win-probability leaf value, scaled to the formula's range."""
    try:
        if cur.result >= 0:
            return 1e9 if cur.result == me else (-1e9 if cur.result == (1 - me) else 0.0)
        if not _load():
            return _formula_eval(cur, me)
        f = _feats(cur, me)
        mu, sd = _STATS["mu"], _STATS["sd"]
        x = [(v - mu[i]) / sd[i] for i, v in enumerate(f)]
        with _TORCH.no_grad():
            logit = _M(_TORCH.tensor([x], dtype=_TORCH.float)).item()
        # logit is already monotone in win probability; scaling only puts it on a
        # comparable footing with the terminal +-1e9 sentinels.
        return logit * _SCALE
    except Exception:
        try:
            return _formula_eval(cur, me)
        except Exception:
            return 0.0
