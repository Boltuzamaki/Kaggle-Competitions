"""Generate successful on-policy corrections with controlled exploration.

This is deliberately not random self-play.  A strong fixed Grim policy plays a
diverse frozen league; on a small fraction of single-select decisions it explores
another legal action.  Only the candidate's decisions from games it wins are
retained.  Opponents/seeds are recorded so validation can hold them out later.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import pickle
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "agent"),
                os.path.join(ROOT, "tools", "crn")]

from paired_eval import StartData, SerialData, _lib, MAX_STEPS  # noqa: E402
from extract_dataset import board_context, option_features, state_tokens  # noqa: E402

BASE = os.environ.get("TRAJ_BASE", "ours-grimmsnarl")
OPPONENTS = ("public-alakazam", "public-archaludon", "public-garchomp-v28",
             "router-v12", "meta-crustle", "meta-grimmsnarl")
_base = _deck = _opps = None


def init_worker():
    global _base, _deck, _opps
    from paired_eval import build
    _base, _deck = build(BASE)
    _opps = {name: build(name) for name in OPPONENTS}


def play_job(job):
    seed, seat, opponent_name, epsilon, *flags = job
    keep_losses = bool(flags[0]) if flags else False
    opponent, opp_deck = _opps[opponent_name]
    cards = (ctypes.c_int * 120)(*(list(_deck) + list(opp_deck))) \
        if seat == 0 else (ctypes.c_int * 120)(*(list(opp_deck) + list(_deck)))
    start = _lib.CrnBattleStart(cards, seed & 0xFFFFFFFF or 1)
    if start.errorPlayer >= 0:
        return []
    ptr = start.battlePtr
    rows = []
    rng = random.Random(seed ^ int(epsilon * 10000) ^ (seat << 25))
    agents = (_base, opponent) if seat == 0 else (opponent, _base)
    try:
        winner = None
        for _ in range(MAX_STEPS):
            sd = _lib.GetBattleData(ptr)
            if not sd.json:
                break
            obs = json.loads(sd.json.decode("utf-8", "replace"))
            cur = obs.get("current") or {}
            if cur.get("result", -1) >= 0:
                winner = cur["result"]
                break
            sel = obs.get("select") or {}
            opts = sel.get("option") or []
            who = sd.selectPlayer if sd.selectPlayer in (0, 1) else 0
            choice = agents[who](obs) or [0]
            baseline_choice = list(choice)
            explored = False
            if who == seat and len(opts) >= 2 and (sel.get("maxCount", 1) or 1) == 1:
                if rng.random() < epsilon:
                    alternatives = [i for i in range(len(opts)) if i != choice[0]]
                    if alternatives:
                        choice = [rng.choice(alternatives)]
                        explored = choice != baseline_choice
                me = cur.get("yourIndex")
                if me is not None and len(choice) == 1 and 0 <= choice[0] < len(opts):
                    feats = [option_features(o, cur, sel, me) for o in opts]
                    rows.append({"episode": f"self_{seed}_{seat}_{opponent_name}_{epsilon}",
                                 "seed": seed, "seat": seat, "opponent": opponent_name,
                                 "epsilon": epsilon, "teacher_name": "winner_explorer",
                                 "baseline_y": baseline_choice[0] if len(baseline_choice) == 1 else -1,
                                 "explored": explored,
                                 "ctx": board_context(cur, me), "state": state_tokens(cur, me),
                                 "cids": [f[0] for f in feats], "types": [f[1] for f in feats],
                                 "nums": [f[2] for f in feats], "y": choice[0],
                                 "ctxid": int(sel.get("context", 0) or 0)})
            choice = [x for x in choice if isinstance(x, int) and 0 <= x < len(opts)] or [0]
            arr = (ctypes.c_int * len(choice))(*choice)
            if _lib.Select(ptr, arr, len(choice)) != 0:
                winner = 1 - who
                break
        won = winner == seat
        return won, rows if won or keep_losses else []
    except Exception:
        return False, []
    finally:
        _lib.BattleFinish(ptr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=930000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epsilons", default="0.02,0.05,0.10")
    ap.add_argument("--causal-only", action="store_true",
                    help="keep variants only when epsilon=0 loses the matched game")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opponents", default=",".join(OPPONENTS))
    args = ap.parse_args()
    epsilons = tuple(float(value) for value in args.epsilons.split(",") if value.strip())
    opponents = tuple(value.strip() for value in args.opponents.split(",") if value.strip())
    unknown = sorted(set(opponents) - set(OPPONENTS))
    if unknown:
        raise ValueError(f"unknown opponents: {unknown}")
    groups = [(args.seed0 + i, seat, opp)
              for i in range(args.seeds) for seat in (0, 1) for opp in opponents]
    jobs = groups if args.causal_only else [(*group, eps) for group in groups for eps in epsilons]
    rows = []
    with mp.get_context("fork").Pool(args.workers, initializer=init_worker) as pool:
        fn = causal_job if args.causal_only else play_job
        iterator = pool.imap_unordered(fn, jobs, chunksize=2)
        for i, result in enumerate(iterator, 1):
            part = result if args.causal_only else result[1]
            rows.extend(part)
            if i % 200 == 0:
                print(i, len(jobs), len(rows), flush=True)
    with open(args.out, "wb") as fh:
        pickle.dump(rows, fh, protocol=4)
    print(f"wrote {len(rows)} winning decisions from "
          f"{len(set(r['episode'] for r in rows))} won games", flush=True)


def causal_job(group):
    seed, seat, opponent = group
    base_won, _ = play_job((seed, seat, opponent, 0.0))
    if base_won:
        return []
    kept = []
    # Worker receives the configured exploration rates through this constant;
    # the causal round deliberately uses conservative rates.
    for epsilon in (0.01, 0.02, 0.05):
        won, rows = play_job((seed, seat, opponent, epsilon))
        if won:
            kept.extend(rows)
    return kept


if __name__ == "__main__":
    main()
