"""Extract (observation, options, chosen index) training pairs from official replays.

The daily top-episode datasets are published by the competition hosts explicitly
"to help in reviewing replays as well as training agents" (discussion 709160), so
this is sanctioned use of public game DATA -- not public agent code. The encoding,
model and training loop are ours.

Two filters matter more than anything else here:

  * teacher strength -- only keep decisions made by teams whose leaderboard score
    clears a floor, because we are trying to clone play that is ~500 points
    stronger than our current agent, not the median of the field;
  * outcome -- only keep the WINNER's decisions, which is filtered behaviour
    cloning and reliably beats plain BC on logged data.

Each decision is stored as the option list plus a compact board context. Every
option keeps its CARD IDENTITY (`cardId`): the public post-mortem in discussion
713608 found nine methods plateaued because they encoded only option *type* and
damage, so "the network literally could not see card synergies". Card identity is
the representational lever those methods lacked.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "agent"))


def load_lb_scores():
    """team name -> leaderboard score, for teacher filtering."""
    pats = glob.glob(os.path.join(ROOT, "scratchpad", "**", "*publicleaderboard*.csv"),
                     recursive=True)
    if not pats:
        return {}
    latest = sorted(pats)[-1]
    out = {}
    with open(latest, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                out[r["TeamName"].strip()] = float(r["Score"])
            except Exception:
                pass
    return out


def board_context(cur, me):
    """Small fixed-width numeric summary of the position."""
    try:
        mp = cur["players"][me]
        op = cur["players"][1 - me]

        def mons(p):
            act = p.get("active") or []
            ben = p.get("bench") or []
            return [m for m in list(act) + list(ben) if m]

        my, oo = mons(mp), mons(op)
        f = [
            float(cur.get("turn", 0)),
            float(len(mp.get("prize") or [])),
            float(len(op.get("prize") or [])),
            float(mp.get("handCount", 0) or 0),
            float(op.get("handCount", 0) or 0),
            float(mp.get("deckCount", 0) or 0),
            float(op.get("deckCount", 0) or 0),
            float(len(my)), float(len(oo)),
            float(sum((m.get("hp") or 0) for m in my)),
            float(sum((m.get("hp") or 0) for m in oo)),
            float(sum(len(m.get("energies") or []) for m in my)),
            float(sum(len(m.get("energies") or []) for m in oo)),
            float(1 if cur.get("energyAttached") else 0),
            float(1 if cur.get("supporterPlayed") else 0),
            float(1 if cur.get("retreated") else 0),
            float(1 if cur.get("firstPlayer") == me else 0),
        ]
        # active pokemon identity + hp on both sides
        a0 = (mp.get("active") or [None])[0]
        a1 = (op.get("active") or [None])[0]
        f += [float(a0["id"]) if a0 else 0.0, float(a0.get("hp") or 0) if a0 else 0.0,
              float(a1["id"]) if a1 else 0.0, float(a1.get("hp") or 0) if a1 else 0.0]
        return f
    except Exception:
        return [0.0] * 21


def state_tokens(cur, me):
    """Card-identity tokens for hand/board/discards with compact local numbers."""
    out = []
    try:
        for side, pidx in ((0, me), (1, 1-me)):
            p = cur["players"][pidx]
            zones = ((0, p.get("hand") or []),
                     (1, p.get("active") or []),
                     (2, p.get("bench") or []),
                     (3, p.get("discard") or []))
            for z, cards in zones:
                # Opponent hand is hidden; it is normally counts/placeholders.
                if side == 1 and z == 0:
                    continue
                for card in cards:
                    if not card:
                        continue
                    cid = int(card.get("id", 0) or 0)
                    if cid <= 0:
                        continue
                    hp = float(card.get("hp", 0) or 0)
                    ens = card.get("energies") or card.get("energyCards") or []
                    out.append((cid, side * 4 + z, [hp, float(len(ens))]))
        for card in cur.get("stadium") or []:
            if card and int(card.get("id", 0) or 0) > 0:
                out.append((int(card["id"]), 8, [0.0, 0.0]))
    except Exception:
        pass
    return out


def _zone_card_id(cur, sel, me, area, index, player_index=None):
    """Resolve engine index-based options to card identity (notably PLAY)."""
    try:
        pidx = me if player_index is None else int(player_index)
        if not (0 <= pidx < len(cur.get("players") or [])):
            pidx = me
        p = cur["players"][pidx]
        zones = {1: (sel or {}).get("deck") or [], 2: p.get("hand") or [],
                 3: p.get("discard") or [], 4: p.get("active") or [],
                 5: p.get("bench") or [], 6: p.get("prize") or [],
                 7: cur.get("stadium") or [], 12: cur.get("looking") or []}
        card = zones.get(int(area), [])[int(index)]
        return int((card or {}).get("id", 0) or 0)
    except Exception:
        return 0


def option_features(o, cur=None, sel=None, me=None):
    """(card_id, type_id, numeric[]) for one legal option -- card identity kept."""
    t = int(o.get("type", 0) or 0)
    cid = int(o.get("cardId", 0) or 0)
    if cid == 0 and cur is not None and me is not None:
        # PLAY/EVOLVE/ATTACH refer to the hand by index but commonly omit cardId.
        area = o.get("area")
        if area is None and t in (7, 8, 9):
            area = 2
        cid = _zone_card_id(cur, sel, me, area, o.get("index", 0),
                            o.get("playerIndex"))
    num = [
        float(o.get("number", 0) or 0),
        float(o.get("area", 0) or 0),
        float(o.get("index", 0) or 0),
        float(o.get("playerIndex", -1) if o.get("playerIndex") is not None else -1),
        float(o.get("count", 0) or 0),
        float(o.get("inPlayArea", 0) or 0),
        float(o.get("inPlayIndex", 0) or 0),
        float(o.get("attackId", 0) or 0),
    ]
    return cid, t, num


def extract_decks(d):
    try:
        for rec in d["steps"][0]:
            for viz in rec.get("visualize") or []:
                decks = [x for x in (viz.get("action") or [])
                         if isinstance(x, list) and len(x) == 60]
                if len(decks) == 2:
                    return decks
    except Exception:
        pass
    return None


def extract(paths, lb, min_score, winners_only, max_options, limit_games,
            teams_filter=None, exact_deck=None):
    rows = []
    stats = Counter()
    for gi, fp in enumerate(paths):
        if limit_games and gi >= limit_games:
            break
        if gi % 200 == 0:
            print(f"  {gi}/{len(paths)} games, {len(rows)} decisions", flush=True)
        try:
            d = json.load(open(fp))
        except Exception:
            stats["unreadable"] += 1
            continue
        teams = [t.strip() for t in ((d.get("info") or {}).get("TeamNames") or ["?", "?"])]
        decks = extract_decks(d) if exact_deck is not None else None
        rw = d.get("rewards") or []
        winner = None
        if len(rw) == 2 and rw[0] != rw[1]:
            winner = 0 if (rw[0] or 0) > (rw[1] or 0) else 1
        scores = [lb.get(t, -1.0) for t in teams]

        for step in d.get("steps") or []:
            for p in (0, 1):
                try:
                    rec = step[p]
                except Exception:
                    continue
                act = rec.get("action")
                obs = rec.get("observation") or {}
                sel = obs.get("select")
                if not act or sel is None:
                    continue
                if teams_filter and teams[p] not in teams_filter:
                    stats["skip_team"] += 1
                    continue
                if exact_deck is not None and (not decks or sorted(decks[p]) != exact_deck):
                    stats["skip_deck"] += 1
                    continue
                if winners_only and winner is not None and p != winner:
                    stats["skip_loser"] += 1
                    continue
                if min_score > 0 and scores[p] < min_score:
                    stats["skip_weak_teacher"] += 1
                    continue
                opts = sel.get("option") or []
                n = len(opts)
                if n < 2 or n > max_options:
                    stats["skip_size"] += 1
                    continue
                if (sel.get("maxCount", 1) or 1) != 1:
                    stats["skip_multiselect"] += 1
                    continue
                if not (len(act) == 1 and isinstance(act[0], int) and 0 <= act[0] < n):
                    stats["skip_misaligned"] += 1
                    continue
                cur = obs.get("current")
                if not cur:
                    continue
                me = cur.get("yourIndex")
                if me is None:
                    continue
                feats = [option_features(o, cur, sel, me) for o in opts]
                rows.append({
                    "episode": os.path.basename(fp).split(".")[0],
                    "player": p,
                    "teacher_name": teams[p],
                    "state": state_tokens(cur, me),
                    "ctx": board_context(cur, me),
                    "cids": [f[0] for f in feats],
                    "types": [f[1] for f in feats],
                    "nums": [f[2] for f in feats],
                    "y": act[0],
                    "ctxid": int(sel.get("context", 0) or 0),
                })
                stats["kept"] += 1
        stats["games"] += 1
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", action="append", default=[],
                    help="episode directory; repeatable")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "rank_dataset.pkl"))
    ap.add_argument("--min-score", type=float, default=1000.0,
                    help="only clone teams at or above this leaderboard score")
    ap.add_argument("--winners-only", action="store_true", default=True)
    ap.add_argument("--all-sides", dest="winners_only", action="store_false")
    ap.add_argument("--max-options", type=int, default=60)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--team", action="append", default=[],
                    help="exact teacher team name; repeatable")
    ap.add_argument("--exact-deck", default="",
                    help="JSON file containing the exact 60-card deck")
    a = ap.parse_args()

    lb = load_lb_scores()
    print(f"leaderboard scores loaded: {len(lb)} teams")
    episode_roots = a.episodes or [os.path.join(ROOT, "data", "episodes")]
    paths = sorted(p for root in episode_roots
                   for p in glob.glob(os.path.join(root, "*.json")))
    print(f"{len(paths)} episode files; teacher floor = {a.min_score}, "
          f"winners_only = {a.winners_only}")
    exact_deck = None
    if a.exact_deck:
        exact_deck = sorted(json.load(open(a.exact_deck)))
    rows, stats = extract(paths, lb, a.min_score, a.winners_only,
                          a.max_options, a.limit_games, set(a.team), exact_deck)
    print("\nstats:", dict(stats))
    print(f"kept {len(rows)} decisions")
    if rows:
        sizes = Counter(len(r["cids"]) for r in rows)
        print("option-count distribution (top 8):", sizes.most_common(8))
        with open(a.out, "wb") as fh:
            pickle.dump(rows, fh, protocol=4)
        print("wrote", a.out, f"({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
