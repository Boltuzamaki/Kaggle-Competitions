"""Parallel exact-team/deck extraction for elite replay behavior cloning."""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from multiprocessing import Pool

from extract_dataset import (board_context, extract_decks, option_features,
                             state_tokens, load_lb_scores)

_TEAM = None
_DECK = None
_MAX_OPTIONS = 60
_MIN_SCORE = 0.0
_WINNERS_ONLY = False
_LB = {}


def init_worker(team, deck, max_options, min_score, winners_only, lb):
    global _TEAM, _DECK, _MAX_OPTIONS, _MIN_SCORE, _WINNERS_ONLY, _LB
    _TEAM, _DECK, _MAX_OPTIONS = team, deck, max_options
    _MIN_SCORE, _WINNERS_ONLY, _LB = min_score, winners_only, lb


def inspect(fp):
    try:
        d = json.load(open(fp))
        teams = [x.strip() for x in ((d.get("info") or {}).get("TeamNames") or [])]
        decks = extract_decks(d)
        if len(teams) != 2 or not decks:
            return []
        rewards = d.get("rewards") or []
        winner = None
        if len(rewards) == 2 and rewards[0] != rewards[1]:
            winner = 0 if (rewards[0] or 0) > (rewards[1] or 0) else 1
        players = [p for p in (0, 1)
                   if (not _TEAM or teams[p] == _TEAM)
                   and sorted(decks[p]) == _DECK
                   and _LB.get(teams[p], -1) >= _MIN_SCORE
                   and (not _WINNERS_ONLY or winner is None or p == winner)]
        if not players:
            return []
        rows = []
        for p in players:
            for step in d.get("steps") or []:
                rec = step[p]
                obs = rec.get("observation") or {}
                sel = obs.get("select")
                act = rec.get("action")
                opts = (sel or {}).get("option") or []
                if (not act or sel is None or len(opts) < 2
                        or len(opts) > _MAX_OPTIONS
                        or (sel.get("maxCount", 1) or 1) != 1
                        or len(act) != 1 or not isinstance(act[0], int)
                        or not 0 <= act[0] < len(opts)):
                    continue
                cur = obs.get("current")
                me = (cur or {}).get("yourIndex")
                if me is None:
                    continue
                feats = [option_features(o, cur, sel, me) for o in opts]
                rows.append({"episode": os.path.basename(fp).split(".")[0],
                             "player": p, "teacher_name": _TEAM,
                             "ctx": board_context(cur, me),
                             "state": state_tokens(cur, me),
                             "cids": [f[0] for f in feats],
                             "types": [f[1] for f in feats],
                             "nums": [f[2] for f in feats], "y": act[0],
                             "ctxid": int(sel.get("context", 0) or 0)})
        return rows
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", action="append", required=True)
    ap.add_argument("--team", default="",
                    help="optional exact team; empty clones every qualifying pilot")
    ap.add_argument("--deck", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)-1))
    ap.add_argument("--max-options", type=int, default=60)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--winners-only", action="store_true")
    a = ap.parse_args()
    paths = [p for root in a.episodes for p in glob.glob(os.path.join(root, "*.json"))]
    if a.deck.endswith(".csv"):
        deck = sorted(int(x) for x in open(a.deck) if x.strip())
    else:
        deck = sorted(json.load(open(a.deck)))
    lb = load_lb_scores()
    rows = []
    with Pool(a.workers, initializer=init_worker,
              initargs=(a.team, deck, a.max_options, a.min_score,
                        a.winners_only, lb)) as pool:
        for i, part in enumerate(pool.imap_unordered(inspect, paths, chunksize=2)):
            rows.extend(part)
            if i % 1000 == 0:
                print(i, len(rows), flush=True)
    with open(a.out, "wb") as fh:
        pickle.dump(rows, fh, protocol=4)
    print(f"wrote {len(rows)} decisions from {len(set(r['episode'] for r in rows))} games")


if __name__ == "__main__":
    main()
