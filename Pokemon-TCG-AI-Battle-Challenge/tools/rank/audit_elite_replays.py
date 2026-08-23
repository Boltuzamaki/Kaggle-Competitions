"""Summarise action-level replay coverage by elite team and deck signature."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict
from multiprocessing import Pool


def deck_lists(d):
    try:
        for rec in d["steps"][0]:
            for viz in rec.get("visualize") or []:
                lists = [x for x in (viz.get("action") or [])
                         if isinstance(x, list) and len(x) == 60]
                if len(lists) == 2:
                    return lists
    except Exception:
        pass
    return None


def decision_count(d, player):
    n = 0
    for step in d.get("steps") or []:
        try:
            rec = step[player]
            obs = rec.get("observation") or {}
            sel = obs.get("select")
            act = rec.get("action")
            opts = (sel or {}).get("option") or []
            if (sel is not None and len(opts) >= 2 and act and len(act) == 1
                    and isinstance(act[0], int) and 0 <= act[0] < len(opts)):
                n += 1
        except Exception:
            pass
    return n


def inspect_file(fp):
    """Return compact per-player records; suitable for process-pool mapping."""
    try:
        d = json.load(open(fp))
        names = (d.get("info") or {}).get("TeamNames") or []
        dl = deck_lists(d)
        rewards = d.get("rewards") or []
        if len(names) != 2 or not dl:
            return []
        out = []
        for p, raw_name in enumerate(names):
            dec = decision_count(d, p)
            won = len(rewards) == 2 and rewards[p] > rewards[1-p]
            out.append((raw_name.strip(), dec, won, tuple(sorted(dl[p]))))
        return out
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", action="append", required=True)
    ap.add_argument("--leaderboard", required=True)
    ap.add_argument("--min-score", type=float, default=1000)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    lb = {}
    with open(a.leaderboard, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                lb[row["TeamName"].strip()] = float(row["Score"])
            except Exception:
                pass

    stats = defaultdict(Counter)
    decks = defaultdict(Counter)
    paths = []
    for root in a.episodes:
        paths.extend(glob.glob(os.path.join(root, "*.json")))
    with Pool(a.workers) as pool:
        for i, records in enumerate(pool.imap_unordered(inspect_file, paths, chunksize=4)):
            if i % 1000 == 0:
                print(f"{i}/{len(paths)}", flush=True)
            for name, dec, won, deck in records:
                if lb.get(name, -1) < a.min_score:
                    continue
                stats[name]["games"] += 1
                stats[name]["decisions"] += dec
                if won:
                    stats[name]["wins"] += 1
                    stats[name]["win_decisions"] += dec
                decks[name][deck] += 1

    ranked = sorted(stats, key=lambda n: (stats[n]["decisions"], lb[n]), reverse=True)
    print("team\tlb\tgames\twins\twr\tdecisions\twin_decisions\tmodal_deck_games\tmodal_deck")
    for name in ranked[:a.top]:
        s = stats[name]
        deck, freq = decks[name].most_common(1)[0]
        print(f"{name}\t{lb[name]:.1f}\t{s['games']}\t{s['wins']}\t"
              f"{s['wins']/s['games']:.3f}\t{s['decisions']}\t{s['win_decisions']}\t"
              f"{freq}\t{','.join(map(str, deck))}")


if __name__ == "__main__":
    main()
