"""Adapter exposing the forked public baseline's weight table for tuning.

The public agent we submitted keeps its 69 card-specific priorities in a module
level WEIGHTS dict (with `W is WEIGHTS`, so runtime mutation works) and also reads
an external ./alak_w.json at import. That means we can evolve its weights with our
own CRN/SPRT/memetic infrastructure and ship the result as a weight FILE, without
editing a line of its code.

Two other knobs it exposes, both conservative:
    K_OPP = 3              opponent branching at ply 2
    TIME_BUDGET_S = 0.8    of a 600 s per-game budget

Its authors tuned these without common random numbers (the published write-ups
say nobody had CRN working). We do, so our search over the same surface is better
directed than theirs was.
"""
from __future__ import annotations

import importlib.util as _ilu
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_FP = os.path.join(_HERE, "fork", "fork_main.py")

_spec = _ilu.spec_from_file_location("fork_main_tunable", _FP)
_fm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fm)

W = _fm.WEIGHTS          # same object the agent reads at decision time
DECK = [int(x) for x in open(os.path.join(_HERE, "fork", "deck.csv")) if x.strip()]


def agent(obs_dict, deck=None):
    return _fm.agent(obs_dict)


def set_knobs(k_opp=None, budget=None):
    if k_opp is not None:
        _fm.K_OPP = int(k_opp)
    if budget is not None:
        _fm.TIME_BUDGET_S = float(budget)
