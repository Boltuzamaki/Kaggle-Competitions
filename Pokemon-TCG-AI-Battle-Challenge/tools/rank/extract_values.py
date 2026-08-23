"""Extract (board state -> did this player win) pairs for a value critic.

This is deliberately NOT the failed policy cloning. That task was "predict which
option a strong player picked", which reached 47% accuracy yet played worse than
the search it was meant to guide -- classic covariate shift.

This task is "from this board, who wins?", which is used as the LEAF EVALUATOR
inside search rather than as a policy. Search corrects its own errors by lookahead,
so a better leaf estimate compounds instead of drifting.

The signal is known to be there: a rival team reported a 10-feature linear probe
on public board state reaching AUC 0.818 (0.69 even on turns 1-4) where their
neural value head managed 0.61 (discussion 713608).

Unlike the policy dataset, EVERY board is usable -- winners and losers, all teams --
because the label is the game outcome, not somebody's choice. That makes the
dataset far larger and free of teacher-quality bias.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "agent"))

FEATS = [
    "turn", "my_prize", "op_prize", "prize_diff", "my_hand", "op_hand",
    "my_deck", "op_deck", "my_mons", "op_mons", "my_hp", "op_hp", "hp_diff",
    "my_energy", "op_energy", "energy_diff", "my_active_hp", "op_active_hp",
    "my_active_max", "op_active_max", "my_bench", "op_bench", "is_first",
    "my_discard", "op_discard",
]


def features(cur, me):
    mp = cur["players"][me]
    op = cur["players"][1 - me]

    def mons(p):
        return [m for m in list(p.get("active") or []) + list(p.get("bench") or []) if m]

    my, oo = mons(mp), mons(op)
    a0 = (mp.get("active") or [None])[0]
    a1 = (op.get("active") or [None])[0]
    myhp = sum((m.get("hp") or 0) for m in my)
    ophp = sum((m.get("hp") or 0) for m in oo)
    mye = sum(len(m.get("energies") or []) for m in my)
    ope = sum(len(m.get("energies") or []) for m in oo)
    mypz = len(mp.get("prize") or [])
    oppz = len(op.get("prize") or [])
    return [
        float(cur.get("turn", 0)), float(mypz), float(oppz), float(oppz - mypz),
        float(mp.get("handCount", 0) or 0), float(op.get("handCount", 0) or 0),
        float(mp.get("deckCount", 0) or 0), float(op.get("deckCount", 0) or 0),
        float(len(my)), float(len(oo)), float(myhp), float(ophp), float(myhp - ophp),
        float(mye), float(ope), float(mye - ope),
        float(a0.get("hp") or 0) if a0 else 0.0, float(a1.get("hp") or 0) if a1 else 0.0,
        float(a0.get("maxHp") or 0) if a0 else 0.0, float(a1.get("maxHp") or 0) if a1 else 0.0,
        float(len([m for m in (mp.get("bench") or []) if m])),
        float(len([m for m in (op.get("bench") or []) if m])),
        float(1 if cur.get("firstPlayer") == me else 0),
        float(len(mp.get("discard") or [])), float(len(op.get("discard") or [])),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "value_dataset.pkl"))
    ap.add_argument("--stride", type=int, default=3, help="sample every Nth board")
    ap.add_argument("--limit-games", type=int, default=0)
    a = ap.parse_args()

    dirs = [os.path.join(ROOT, "data", "episodes")] + sorted(
        glob.glob(os.path.join(ROOT, "data", "ep_*")))
    files = []
    for d in dirs:
        files += sorted(glob.glob(os.path.join(d, "*.json")))
    if a.limit_games:
        files = files[:a.limit_games]
    print(f"{len(files)} episode files")

    X, Y = [], []
    stats = Counter()
    for gi, fp in enumerate(files):
        if gi % 1000 == 0:
            print(f"  {gi}/{len(files)} boards={len(X)}", flush=True)
        try:
            d = json.load(open(fp))
            rw = d.get("rewards") or []
            if len(rw) != 2 or rw[0] == rw[1]:
                stats["draw"] += 1
                continue
            winner = 0 if (rw[0] or 0) > (rw[1] or 0) else 1
            steps = d.get("steps") or []
            for si, step in enumerate(steps):
                if si % a.stride:
                    continue
                for p in (0, 1):
                    try:
                        obs = step[p].get("observation") or {}
                        cur = obs.get("current")
                        if not cur or cur.get("result", -1) >= 0:
                            continue
                        me = cur.get("yourIndex")
                        if me is None:
                            continue
                        X.append(features(cur, me))
                        Y.append(1 if me == winner else 0)
                    except Exception:
                        pass
            stats["games"] += 1
        except Exception:
            stats["unreadable"] += 1

    print(f"\n{len(X)} boards, positive rate {sum(Y)/max(len(Y),1):.3f}, {dict(stats)}")
    pickle.dump({"X": X, "Y": Y, "feats": FEATS}, open(a.out, "wb"), protocol=4)
    print("wrote", a.out, f"({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
