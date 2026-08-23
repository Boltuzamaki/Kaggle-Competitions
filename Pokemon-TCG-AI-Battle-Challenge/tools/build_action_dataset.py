"""Build a flat state/action dataset from our own trajectory traces."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path


def _chosen_label(choice):
    if not choice:
        return "NONE"
    if choice.get("type") == "PLAY":
        return f"PLAY:{choice.get('card_id')}"
    if choice.get("type") == "ATTACK":
        return f"ATTACK:{choice.get('attack_id')}"
    if choice.get("type") in {"ATTACH", "EVOLVE"}:
        return f"{choice.get('type')}:{choice.get('target_id')}"
    return choice.get("type", "UNKNOWN")


def _row(source, opponent, game, event):
    board = event.get("board") or {}
    active = board.get("active") or {}
    opponent_active = board.get("opponent_active") or {}
    chosen = (event.get("chosen_options") or [{}])[0]
    options = event.get("options") or []
    option_scores = [option.get("score") for option in options if option.get("score") is not None]
    chosen_score = chosen.get("score")
    best_score = max(option_scores) if option_scores else None
    return {
        "source": source,
        "opponent": opponent,
        "game": game.get("game"),
        "seat": game.get("seat"),
        "outcome": game.get("outcome"),
        "turn": event.get("turn"),
        "context": event.get("context"),
        "active_id": active.get("id"),
        "active_hp": active.get("hp"),
        "active_energy": active.get("energy"),
        "bench_count": len(board.get("bench") or []),
        "opponent_active_id": opponent_active.get("id"),
        "opponent_active_hp": opponent_active.get("hp"),
        "opponent_bench_count": len(board.get("opponent_bench") or []),
        "prizes": board.get("prizes"),
        "opponent_prizes": board.get("opponent_prizes"),
        "hand_count": sum((board.get("hand") or {}).values()),
        "discard_count": sum((board.get("discard") or {}).values()),
        "chosen_type": chosen.get("type"),
        "chosen_card_id": chosen.get("card_id"),
        "chosen_attack_id": chosen.get("attack_id"),
        "chosen_target_id": chosen.get("target_id"),
        "chosen_label": _chosen_label(chosen),
        "chosen_score": chosen_score,
        "best_score": best_score,
        "score_gap": (
            best_score - chosen_score
            if best_score is not None and chosen_score is not None
            else None
        ),
        "legal_option_count": len(options),
        "legal_types": "|".join(sorted({option.get("type", "") for option in options})),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--csv", default="scratchpad/action_dataset.csv")
    parser.add_argument("--report", default="scratchpad/action_dataset_report.json")
    args = parser.parse_args()

    rows = []
    games_seen = set()
    for input_path in args.inputs:
        path = Path(input_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        opponent = payload.get("summary", {}).get("opponent", path.stem)
        for game in payload.get("games", []):
            games_seen.add((path.name, game.get("game")))
            for event in game.get("events", []):
                if event.get("context") == "MAIN" and event.get("board"):
                    rows.append(_row(path.name, opponent, game, event))

    output = Path(args.csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    by_opponent = defaultdict(Counter)
    actions = Counter()
    outcomes = Counter()
    for row in rows:
        by_opponent[row["opponent"]][row["outcome"]] += 1
        actions[row["chosen_label"]] += 1
        outcomes[row["outcome"]] += 1
    report = {
        "source_files": args.inputs,
        "games": len(games_seen),
        "main_decisions": len(rows),
        "decision_outcomes": dict(outcomes),
        "decisions_by_opponent_and_outcome": {
            opponent: dict(counts) for opponent, counts in by_opponent.items()
        },
        "top_actions": actions.most_common(25),
        "contains_public_policy_actions": False,
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
