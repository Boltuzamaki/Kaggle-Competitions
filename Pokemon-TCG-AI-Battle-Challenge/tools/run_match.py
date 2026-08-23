"""
Local self-play harness for the Pokémon TCG AI Battle Challenge.

Runs our agent against a baseline (or itself) inside the cabt simulator and
reports a win-rate. This is your offline measuring stick — the leaderboard only
gives you 5 submissions/day, but this you can run unlimited times.

Usage
-----
    pip install kaggle-environments==1.30.1
    python tools/run_match.py --games 40
    python tools/run_match.py --games 40 --opponent random --render result.html

Extend this into a full round-robin arena vs. a gauntlet of frozen opponents
(see PLAN.md, Phase 3).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

# Make `agent/` importable regardless of where we're run from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from agent.main import agent as our_agent  # noqa: E402

# Deck used by the baseline random_agent during the deck-selection phase.
# Populated in main() once we know which deck we're testing with.
_DECK: list[int] = []


def random_agent(obs_dict: dict) -> list[int]:
    """Uniform-random legal move — the floor every agent must beat."""
    sel = obs_dict.get("select")
    if sel is None:  # deck-selection phase -> return the deck card IDs
        return list(_DECK)
    opts = sel.get("option") or []
    if not opts:
        return []
    k = min(sel.get("maxCount", 1) or 1, len(opts))
    return random.sample(range(len(opts)), k)


def load_deck(path: str) -> list[int]:
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip() and not line.startswith("#")]


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a win-rate — trust this over raw %."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def main() -> None:
    ap = argparse.ArgumentParser(description="Local cabt self-play win-rate.")
    ap.add_argument("--games", type=int, default=20, help="number of games to play")
    ap.add_argument("--opponent", choices=["random", "self"], default="random")
    ap.add_argument("--deck", default=os.path.join(_ROOT, "agent", "deck.csv"))
    ap.add_argument("--render", default=None, help="write an HTML replay of game 1")
    args = ap.parse_args()

    try:
        from kaggle_environments import make
    except ImportError:
        sys.exit("kaggle-environments not installed. Run: pip install kaggle-environments==1.30.1")

    global _DECK
    deck = load_deck(args.deck)
    _DECK = deck  # so the baseline random_agent can return it during deck-select
    print(f"Deck: {len(deck)} cards  |  opponent: {args.opponent}  |  games: {args.games}")

    opponent = our_agent if args.opponent == "self" else random_agent

    wins = draws = losses = 0
    t0 = time.time()
    for g in range(args.games):
        # Alternate who goes first to remove first-player bias.
        players = [our_agent, opponent] if g % 2 == 0 else [opponent, our_agent]
        our_seat = 0 if g % 2 == 0 else 1

        env = make("cabt", configuration={"decks": [deck, deck]})
        env.run(players)

        # Final rewards live in the last step: env.steps[-1][seat]["reward"].
        final = env.steps[-1]
        try:
            our_r = final[our_seat].get("reward")
            opp_r = final[1 - our_seat].get("reward")
            if our_r is None or opp_r is None:
                draws += 1
            elif our_r > opp_r:
                wins += 1
            elif our_r < opp_r:
                losses += 1
            else:
                draws += 1
        except Exception:
            draws += 1

        if args.render and g == 0:
            with open(args.render, "w") as f:
                f.write(env.render(mode="html"))
            print(f"  wrote replay -> {args.render}")

        print(f"  game {g + 1}/{args.games}: W{wins}-L{losses}-D{draws}", end="\r")

    n = args.games
    wr = wins / n if n else 0.0
    lo, hi = wilson_interval(wins, n)
    dt = time.time() - t0
    print("\n" + "=" * 48)
    print(f"Record:   {wins}W - {losses}L - {draws}D  ({n} games)")
    print(f"Win-rate: {wr:.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"Time:     {dt:.1f}s  ({dt / max(n,1):.1f}s/game)")
    print("=" * 48)
    if lo > 0.5:
        print("[OK] Significantly beats the opponent -- candidate to submit.")
    elif hi < 0.5:
        print("[BAD] Significantly worse -- do not submit.")
    else:
        print("[--] Inconclusive -- run more games.")


if __name__ == "__main__":
    main()
