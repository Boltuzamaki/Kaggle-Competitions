"""Does restoring the fork's archetype templates make it play better?

The fork determinizes the opponent's hidden deck before every rollout.
`_match_archetype` picks the closest template by Pokemon-ID overlap, and the
sampled opponent deck drives the whole search. With no templates it falls back
to a generic guess.

Our extraction of the public notebook pulled its source and its deck list, but
not the `top20_decks` dataset it attaches -- so the agent sitting at 742.5 on the
ladder has been searching against a blind belief model. Two hardcoded lines
(grimmsnarl, great_tusk) stay active in BOTH arms, so this isolates exactly one
variable: the template pool.

The templates are our own Aug-03 mined top-team decks, not copied content.

Paired on shared seeds: same deck, same policy, same opponents.
"""
from __future__ import annotations

import importlib.util as ilu
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

from paired_eval import play, build, mcnemar  # noqa: E402

FP = os.path.join(ROOT, "agent", "fork", "fork_main.py")
DECK = [int(x) for x in open(os.path.join(ROOT, "agent", "fork", "deck.csv")) if x.strip()]
OPPS = ["public-alakazam", "td-td_00", "meta-dudunsparce", "hybs-grimmsnarl"]


def load(tag):
    s = ilu.spec_from_file_location("fm_" + tag, FP)
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def bind(m):
    def w(o):
        return list(DECK) if o.get("select") is None else m.agent(o)
    return w


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    full = load("full")
    blind = load("blind")
    blind._TEMPLATES = []          # exactly what we have been submitting
    blind._TEMPLATE_SIG = []
    print(f"templates: full={len(full._TEMPLATE_SIG)} blind={len(blind._TEMPLATE_SIG)}",
          flush=True)
    if not full._TEMPLATE_SIG:
        print("ABORT: templated arm has no templates -- cwd is wrong")
        return

    A, B = bind(full), bind(blind)
    import arena  # noqa: F401
    a = b = aw = bw = 0
    for name in OPPS:
        fo, do = build(name)
        for s in [913000 + i for i in range(n)]:
            x = y = 0
            for seat in (0, 1):
                random.seed(s)
                x += int(play(s, A, fo, DECK, do) == 0) if seat == 0 else \
                     int(play(s, fo, A, do, DECK) == 1)
            for seat in (0, 1):
                random.seed(s)
                y += int(play(s, B, fo, DECK, do) == 0) if seat == 0 else \
                     int(play(s, fo, B, do, DECK) == 1)
            aw += x
            bw += y
            if x > y:
                a += 1
            elif y > x:
                b += 1
        print(f"  after {name}: templated {aw} / blind {bw}  (disc {a}/{b})", flush=True)
    p = mcnemar(a, b)
    print(f"\nTEMPLATED {aw} wins vs BLIND {bw}   discordant {a}/{b}   p={p:.4f}")
    print("VERDICT:", "TEMPLATES BETTER" if (a > b and p < 0.05)
          else ("templates worse" if b > a else "no significant difference"))


if __name__ == "__main__":
    main()
