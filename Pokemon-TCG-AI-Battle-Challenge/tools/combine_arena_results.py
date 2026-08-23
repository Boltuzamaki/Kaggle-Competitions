"""Combine compatible one-pairing arena JSON files into one summary artifact."""

from __future__ import annotations

import argparse
import csv
import json
import math


def wilson(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    margin = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games))
    margin /= denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--competitor-b",
        default=None,
        help="select this opponent when an input contains multiple pairings",
    )
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        pairings = data.get("pairings") or []
        if args.competitor_b is not None:
            pairings = [
                row for row in pairings
                if row.get("competitor_b") == args.competitor_b
            ]
        if len(pairings) != 1:
            raise SystemExit(f"{path} must contain exactly one pairing")
        rows.append(pairings[0])

    identity = (rows[0]["competitor_a"], rows[0]["competitor_b"])
    if any((row["competitor_a"], row["competitor_b"]) != identity for row in rows):
        raise SystemExit("pairing identities do not match")

    games = sum(row["games"] for row in rows)
    wins = sum(row["a_wins"] for row in rows)
    losses = sum(row["a_losses"] for row in rows)
    draws = sum(row["draws"] for row in rows)
    low, high = wilson(wins, games)

    def weighted(field):
        return sum(row[field] * row["games"] for row in rows) / games

    combined = {
        "competitor_a": rows[0]["competitor_a"],
        "competitor_b": rows[0]["competitor_b"],
        "deck_a": rows[0]["deck_a"],
        "deck_b": rows[0]["deck_b"],
        "games": games,
        "a_wins": wins,
        "a_losses": losses,
        "draws": draws,
        "a_winrate": round(wins / games, 6),
        "wilson_low": round(low, 6),
        "wilson_high": round(high, 6),
        "first_player_winrate": round(weighted("first_player_winrate"), 6),
        "second_player_winrate": round(weighted("second_player_winrate"), 6),
        "mean_match_s": round(weighted("mean_match_s"), 6),
        "p95_match_s_conservative": max(row["p95_match_s"] for row in rows),
        "crashes": sum(row["crashes"] for row in rows),
        "invalids": sum(row["invalids"] for row in rows),
        "timeouts": sum(row["timeouts"] for row in rows),
        "game_errors": sum(row["game_errors"] for row in rows),
        "source_files": args.inputs,
    }
    output = {
        "pairings": [combined],
        "ratings": {
            rows[0]["competitor_a"]: combined["a_winrate"],
            rows[0]["competitor_b"]: round(losses / games, 6),
        },
    }
    with open(args.out + ".json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    with open(args.out + ".csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined))
        writer.writeheader()
        writer.writerow(combined)
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
