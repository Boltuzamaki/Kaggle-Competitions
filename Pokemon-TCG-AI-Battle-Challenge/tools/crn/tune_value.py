"""Tune v3's leaf-evaluation weights using paired common random numbers.

Every candidate is compared against the CURRENT BEST over identical seeds, and
scored by win-rate weighted by live field share -- because an equal-weighted
average across matchups nearly hid a 15-point Crustle/Garchomp gap.

A candidate is promoted only if it beats the incumbent on a fresh, disjoint seed
set (the search seeds are never reused for confirmation), which is the gate that
correctly rejected the earlier Kaggle weight sweep as overfit.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "agent"), os.path.join(ROOT, "tools"),
          os.path.join(ROOT, "references", "top_rankers")):
    if p not in sys.path:
        sys.path.insert(0, p)

from paired_eval import play, build, mcnemar        # noqa: E402
import search_tuned                                  # noqa: E402
import meta_decks                                    # noqa: E402

DECK = meta_decks.CRUSTLE            # the deck we actually submit
# Opponent panel weighted by the Aug-03 live field share.
PANEL = [("meta-grimmsnarl", 37.9), ("meta-dudunsparce", 10.5),
         ("meta-alakazam", 10.0), ("meta-dragapult", 6.0)]
SEARCH_SEEDS = [7000 + i for i in range(int(os.environ.get("TUNE_SEEDS", "24")))]
CONFIRM_SEEDS = [90000 + i for i in range(int(os.environ.get("CONFIRM_SEEDS", "60")))]
ROUNDS = int(os.environ.get("TUNE_ROUNDS", "12"))
OUT = os.path.join(ROOT, "scratchpad", "scrape_20260804", "value_tuning.json")

_opps = {name: build(name) for name, _ in PANEL}
_total_share = sum(w for _, w in PANEL)


def score(genome, seeds):
    """Field-share-weighted win-rate of `genome` over `seeds`, both seats."""
    cand = search_tuned.make_agent(DECK, genome)
    weighted = 0.0
    detail = {}
    for name, share in PANEL:
        fo, deck_o = _opps[name]
        wins = games = 0
        for s in seeds:
            for seat in (0, 1):
                random.seed(s)
                if seat == 0:
                    w = play(s, cand, fo, DECK, deck_o)
                    wins += int(w == 0)
                else:
                    w = play(s, fo, cand, deck_o, DECK)
                    wins += int(w == 1)
                games += 1
        wr = wins / max(games, 1)
        detail[name] = wr
        weighted += (share / _total_share) * wr
    return weighted, detail


def mutate(g, rng, rate=0.4, scale=0.35):
    child = dict(g)
    for k in child:
        if k in ("det", "maxroll"):
            continue                     # keep the compute budget fixed
        if rng.random() < rate:
            child[k] = max(0.0, child[k] * (1.0 + rng.gauss(0, scale)))
    return child


def main():
    rng = random.Random(2026)
    base = dict(search_tuned.PARAMS)
    best, best_score = dict(base), None

    best_score, base_detail = score(base, SEARCH_SEEDS)
    print(f"baseline (stock v3) field-weighted: {best_score:.4f}")
    for k, v in base_detail.items():
        print(f"    {k:18s} {v:.3f}")

    history = [("baseline", best_score)]
    for r in range(ROUNDS):
        cand = mutate(best, rng)
        s, _ = score(cand, SEARCH_SEEDS)
        tag = "accept" if s > best_score else "reject"
        print(f"  round {r+1:2d}: {s:.4f} vs {best_score:.4f}  {tag}", flush=True)
        if s > best_score:
            best, best_score = cand, s
        history.append((f"round{r+1}", s))

    print("\nconfirming on disjoint seeds...")
    cb, cb_detail = score(dict(base), CONFIRM_SEEDS)
    cc, cc_detail = score(best, CONFIRM_SEEDS)
    promote = cc > cb
    print(f"  baseline  {cb:.4f}")
    print(f"  candidate {cc:.4f}")
    print(f"  promote: {promote}")

    json.dump({"baseline_params": base, "best_params": best,
               "search_score": best_score, "confirm_baseline": cb,
               "confirm_candidate": cc, "promote": bool(promote),
               "confirm_detail_baseline": cb_detail,
               "confirm_detail_candidate": cc_detail,
               "history": history}, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    if promote:
        wp = os.path.join(ROOT, "agent", "value_w.json")
        json.dump(best, open(wp, "w"), indent=1)
        print("wrote", wp)


if __name__ == "__main__":
    main()
