"""Train a tiny original hashed logistic outcome ranker on our decisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import random


def _hash(feature, dimension):
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dimension


def _bucket(value, size=1):
    try:
        return int(float(value or 0)) // size
    except Exception:
        return 0


def features(row, dimension):
    categorical = [
        "bias",
        f"context={row.get('context')}",
        f"active={row.get('active_id')}",
        f"opp_active={row.get('opponent_active_id')}",
        f"action={row.get('chosen_label')}",
        f"type={row.get('chosen_type')}",
        f"turn={_bucket(row.get('turn'), 2)}",
        f"active_hp={_bucket(row.get('active_hp'), 50)}",
        f"active_energy={_bucket(row.get('active_energy'))}",
        f"bench={_bucket(row.get('bench_count'))}",
        f"opp_hp={_bucket(row.get('opponent_active_hp'), 50)}",
        f"opp_bench={_bucket(row.get('opponent_bench_count'))}",
        f"prizes={_bucket(row.get('prizes'))}",
        f"opp_prizes={_bucket(row.get('opponent_prizes'))}",
        f"hand={_bucket(row.get('hand_count'), 2)}",
        f"legal={_bucket(row.get('legal_option_count'), 3)}",
        f"action_active={row.get('chosen_label')}|{row.get('active_id')}",
        f"action_opp={row.get('chosen_label')}|{row.get('opponent_active_id')}",
    ]
    result = Counter(_hash(feature, dimension) for feature in categorical)
    return dict(result)


def _sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _predict(weights, encoded):
    return _sigmoid(sum(weights[index] * value for index, value in encoded.items()))


def _metrics(rows, weights, dimension):
    correct = positive_correct = negative_correct = 0
    positives = negatives = 0
    logloss = 0.0
    for row in rows:
        target = 1 if row["outcome"] == "win" else 0
        probability = _predict(weights, features(row, dimension))
        prediction = probability >= 0.5
        correct += prediction == bool(target)
        positives += target
        negatives += 1 - target
        positive_correct += bool(target) and prediction
        negative_correct += (not target) and (not prediction)
        probability = min(max(probability, 1e-6), 1 - 1e-6)
        logloss -= target * math.log(probability) + (1 - target) * math.log(1 - probability)
    count = max(len(rows), 1)
    tpr = positive_correct / positives if positives else 0.0
    tnr = negative_correct / negatives if negatives else 0.0
    return {
        "rows": len(rows),
        "accuracy": correct / count,
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "logloss": logloss / count,
        "positive_rate": positives / count,
        "majority_baseline": max(positives, negatives) / count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="scratchpad/action_dataset.csv")
    parser.add_argument("--holdout", default="v3-alakazam")
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--model", default="scratchpad/context_ranker.json")
    parser.add_argument("--report", default="scratchpad/context_ranker_report.json")
    args = parser.parse_args()

    with open(args.data, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row["opponent"] != args.holdout]
    holdout = [row for row in rows if row["opponent"] == args.holdout]
    if not train or not holdout:
        raise RuntimeError("both train and holdout rows are required")

    per_game = Counter((row["source"], row["game"]) for row in train)
    weights = [0.0] * args.dimension
    rng = random.Random(7)
    for epoch in range(args.epochs):
        rng.shuffle(train)
        rate = 0.12 / math.sqrt(1.0 + epoch * 0.08)
        for row in train:
            target = 1.0 if row["outcome"] == "win" else 0.0
            encoded = features(row, args.dimension)
            probability = _predict(weights, encoded)
            game_weight = 1.0 / per_game[(row["source"], row["game"])]
            error = (target - probability) * game_weight
            for index, value in encoded.items():
                weights[index] += rate * (error * value - 0.0005 * weights[index])

    model = {
        "kind": "hashed_logistic_outcome_ranker",
        "dimension": args.dimension,
        "weights": weights,
        "feature_version": 1,
        "trained_on_public_policy_actions": False,
    }
    report = {
        "train_opponents": sorted({row["opponent"] for row in train}),
        "holdout_opponent": args.holdout,
        "train": _metrics(train, weights, args.dimension),
        "holdout": _metrics(holdout, weights, args.dimension),
    }
    Path(args.model).write_text(json.dumps(model), encoding="utf-8")
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
