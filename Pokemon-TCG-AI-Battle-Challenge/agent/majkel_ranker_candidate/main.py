"""Majkel exact-deck replay ranker, gated through shallow forward search."""
from __future__ import annotations

import ranker_agent

DECK = [
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,18,18,96,96,96,96,
    1094,1094,1094,1094,1118,1118,1119,1119,1119,1119,1122,1122,
    1122,1127,1127,1137,1147,1147,1159,1182,1182,1182,1201,1213,
    1213,1213,1213,1221,1223,1223,1227,1227,1227,1227,1251,1251,
]

_run = ranker_agent.make_agent(DECK, mode="prior", topk=3)


def agent(obs_dict):
    return _run(obs_dict)
