"""Mine deck lists used by 1200+ rated players, and what they beat.

Rationale for going after decks rather than actions: deck composition is the only
lever in this project that ever produced a large validated gain (+112 from the
Aug-03 meta decks), while every attempt to learn HOW to play from replays -- a
card-identity action ranker, a board-value critic, archetype belief templates --
was measured and rejected. Behaviour cloning also caps you at the level you are
cloning; the existing fork is a ~950-era agent that scores 761 today.

The manifest carries per-episode player ratings, so we can select games where BOTH
players were rated above a threshold and extract the exact 60-card lists they
brought. Each replay's opening action IS the deck submission, so no simulation is
needed.

Also records win/loss per deck signature so a list is judged by how it performed
among elite players, not merely by how often it appears.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "agent"))

MIN_SCORE = float(os.environ.get("ED_MIN", "1150"))


def manifest_scores():
    out = {}
    for f in glob.glob(os.path.join(ROOT, "data", "ep_*", "manifest.csv")):
        for r in csv.DictReader(open(f)):
            try:
                out[r["episode_id"]] = (float(r["min_score"]), float(r["avg_score"]))
            except Exception:
                pass
    return out


def deck_of(step0, seat):
    """The opening action is the 60-card deck submission."""
    try:
        a = step0[seat].get("action")
        if isinstance(a, list) and len(a) == 60 and all(isinstance(x, int) for x in a):
            return a
    except Exception:
        pass
    return None


def main():
    scores = manifest_scores()
    files = glob.glob(os.path.join(ROOT, "data", "ep_*", "*.json"))
    print(f"scanning {len(files)} replays for players rated >= {MIN_SCORE}", flush=True)

    sig_count = Counter()
    sig_wins = Counter()
    sig_games = Counter()
    sig_example = {}
    sig_names = defaultdict(Counter)
    seen = 0

    try:
        from cg.api import all_card_data
        CARD = {c.cardId: c.name for c in all_card_data()}
    except Exception:
        CARD = {}

    for f in files:
        eid = os.path.basename(f)[:-5]
        sc = scores.get(eid)
        if not sc or sc[0] < MIN_SCORE:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        steps = d.get("steps") or []
        if len(steps) < 2:
            continue
        rewards = d.get("rewards") or [0, 0]
        teams = (d.get("info") or {}).get("TeamNames") or ["?", "?"]
        seen += 1
        for seat in (0, 1):
            deck = deck_of(steps[1], seat) or deck_of(steps[0], seat)
            if not deck:
                continue
            sig = tuple(sorted(deck))
            sig_count[sig] += 1
            sig_games[sig] += 1
            try:
                if rewards[seat] > rewards[1 - seat]:
                    sig_wins[sig] += 1
            except Exception:
                pass
            sig_example.setdefault(sig, deck)
            if seat < len(teams):
                sig_names[sig][teams[seat]] += 1

    print(f"  elite episodes matched: {seen}")
    print(f"  distinct deck lists    : {len(sig_count)}\n")

    ranked = sorted(sig_count, key=lambda s: (-sig_games[s], -sig_wins[s]))
    out = {}
    for i, sig in enumerate(ranked[:15]):
        g, w = sig_games[sig], sig_wins[sig]
        who = ", ".join(n for n, _ in sig_names[sig].most_common(2))
        top = Counter(sig_example[sig]).most_common(4)
        desc = " ".join(f"{CARD.get(c, c)}x{n}" for c, n in top)
        print(f"  deck{i:02d}  games {g:4d}  wins {w:4d} ({100*w/max(g,1):4.1f}%)  {who[:34]}")
        print(f"          {desc[:96]}")
        out[f"elite_{i:02d}"] = {"deck": sig_example[sig], "games": g, "wins": w,
                                 "teams": [n for n, _ in sig_names[sig].most_common(3)]}

    p = os.path.join(ROOT, "agent", "elite_decks.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
