"""Identify the opponent's archetype from visible cards and determinize from it.

v3's determinization fills the opponent's deck with a placeholder Basic
(`_SNORLAX`) because it has no idea what they are playing. Every rollout
therefore assumes the opponent draws cards that do nothing, so the search
systematically under-rates their threats -- and averaging more samples of that
model just converges harder on the same wrong answer, which is why more
determinizations bought nothing measurable.

We now have the real field: agent/meta_decks.py holds the current meta lists
mined from the official replay dump. Anything the opponent has revealed (active,
bench, discard, stadium) is matched against those lists to pick the most likely
archetype, and the unseen remainder of that list becomes the deck we roll out
against.

Deck lists are public strategic data; no opponent policy code is used.
"""

from __future__ import annotations

from collections import Counter

from meta_decks import META_DECKS

# Cheap identity: how many copies of each card each meta list runs.
_DECK_COUNTS = {name: Counter(deck) for name, deck in META_DECKS.items()}
_FALLBACK = 1072  # a Basic Pokemon, matching v3's placeholder behaviour


def visible_opponent_cards(opp):
    """Card IDs the opponent has actually revealed to us."""
    seen = []
    try:
        for mon in list(opp.active or []) + list(opp.bench or []):
            if mon is None:
                continue
            seen.append(mon.id)
            # Evolution stacks reveal the whole line underneath.
            pre = getattr(mon, "preEvolution", None) or []
            for p in pre:
                pid = getattr(p, "id", p)
                if isinstance(pid, int):
                    seen.append(pid)
            for e in (getattr(mon, "energyCards", None) or []):
                eid = getattr(e, "id", e)
                if isinstance(eid, int):
                    seen.append(eid)
        for c in (opp.discard or []):
            if c is not None:
                seen.append(c.id)
    except Exception:
        pass
    return [c for c in seen if isinstance(c, int)]


def identify(opp):
    """(archetype_name, score) best explaining the revealed cards, or (None, 0).

    Score is the number of revealed copies the list can account for, so a deck
    that runs 4 of a card we have seen 3 of scores all 3.
    """
    seen = Counter(visible_opponent_cards(opp))
    if not seen:
        return None, 0
    best, best_score = None, 0
    for name, counts in _DECK_COUNTS.items():
        score = sum(min(n, counts.get(cid, 0)) for cid, n in seen.items())
        if score > best_score:
            best, best_score = name, score
    return best, best_score


def opponent_deck_sample(opp, rng, min_evidence=2):
    """A plausible list of the opponent's remaining deck cards.

    Falls back to v3's placeholder when the evidence is too thin to guess -- a
    confident wrong archetype is worse than admitting ignorance.
    """
    n = getattr(opp, "deckCount", 0) or 0
    if n <= 0:
        return []
    name, score = identify(opp)
    if name is None or score < min_evidence:
        return [_FALLBACK] * n

    remaining = Counter(META_DECKS[name])
    for cid in visible_opponent_cards(opp):
        if remaining.get(cid):
            remaining[cid] -= 1
    pool = [cid for cid, k in remaining.items() for _ in range(max(0, k))]
    if not pool:
        return [_FALLBACK] * n
    return [rng.choice(pool) for _ in range(n)]
