"""Nested audit of model routing by decisive-feature missingness pattern."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLASSES = np.array(["at-risk", "fit", "unhealthy"])
LABELS = {label: index for index, label in enumerate(CLASSES)}

ARTIFACTS = {
    "realmlp": (
        ROOT / "kaggle_kernels/realmlp_seed2027_gpu/output/oof_preds.csv",
        ROOT / "kaggle_kernels/realmlp_seed2027_gpu/output/test_preds.csv",
    ),
    "realmlp_31415": (
        ROOT
        / "kaggle_kernels/monitor_downloads/boltuzamaki"
        / "health-risk-realmlp-seed-31415-gpu/oof_preds.csv",
        ROOT
        / "kaggle_kernels/monitor_downloads/boltuzamaki"
        / "health-risk-realmlp-seed-31415-gpu/test_preds.csv",
    ),
    "xgb": (
        ROOT / "kaggle_kernels/rule_xgb_gpu/output/oof_preds.csv",
        ROOT / "kaggle_kernels/rule_xgb_gpu/output/test_preds.csv",
    ),
    "te_hgb": (
        ROOT / "kaggle_kernels/hgbc_seedset_b_cpu/output/tehgbc_oof_preds.csv",
        ROOT / "kaggle_kernels/hgbc_seedset_b_cpu/output/tehgbc_test_preds.csv",
    ),
    "pattern_hgb": (
        ROOT / "kaggle_kernels/pattern_hgb_cpu/output/oof_preds.csv",
        ROOT / "kaggle_kernels/pattern_hgb_cpu/output/test_preds.csv",
    ),
}


def load(path: Path, ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path).sort_values("id")
    if not np.array_equal(frame["id"].to_numpy(), ids):
        raise ValueError(f"ID mismatch: {path}")
    values = frame[list(CLASSES)].to_numpy(float)
    return values / values.sum(axis=1, keepdims=True)


def score(y: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> float:
    pred = probability.argmax(1)
    return float(np.mean([
        np.mean(pred[mask & (y == cls)] == cls) for cls in range(3)
    ]))


def fit_base(
    y: np.ndarray,
    probabilities: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, float]:
    """Greedy three-family blend fitted only on the supplied rows."""
    weights = {"realmlp": 1.0}
    blend = probabilities["realmlp"].copy()
    current = score(y, blend, mask)
    remaining = {"xgb", "te_hgb"}
    for _ in range(2):
        best = (current, "", 0.0, blend)
        for name in sorted(remaining):
            for alpha in np.arange(0.025, 0.5001, 0.025):
                candidate = (1 - alpha) * blend + alpha * probabilities[name]
                value = score(y, candidate, mask)
                if value > best[0] + 1e-12:
                    best = (value, name, float(alpha), candidate)
        if not best[1]:
            break
        current, name, alpha, blend = best
        weights = {key: value * (1 - alpha) for key, value in weights.items()}
        weights[name] = alpha
        remaining.remove(name)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-seed-bag", action="store_true")
    args = parser.parse_args()
    train = pd.read_csv(ROOT / "data/train.csv").sort_values("id")
    test = pd.read_csv(ROOT / "data/test.csv").sort_values("id")
    train_ids = train["id"].to_numpy()
    test_ids = test["id"].to_numpy()
    y = train["health_condition"].map(LABELS).to_numpy()
    key = ["sleep_duration", "stress_level", "physical_activity_level"]
    train_pattern = (
        train[key].isna().astype("int8").astype(str).agg("".join, axis=1).to_numpy()
    )
    test_pattern = (
        test[key].isna().astype("int8").astype(str).agg("".join, axis=1).to_numpy()
    )
    oof = {name: load(paths[0], train_ids) for name, paths in ARTIFACTS.items()}
    test_probability = {
        name: load(paths[1], test_ids) for name, paths in ARTIFACTS.items()
    }

    meta_fold = train_ids % 7
    nested = np.zeros_like(oof["realmlp"])
    nested_base = np.zeros_like(nested)
    selections: list[dict] = []
    base_weights: list[dict[str, float]] = []
    seed_weights: list[float] = []
    grid = np.arange(0.0, 0.5001, 0.05)

    for fold in range(7):
        fit = meta_fold != fold
        heldout = ~fit
        if args.use_seed_bag:
            seed_grid = np.arange(0.0, 1.0001, 0.05)
            seed_scores = [
                score(
                    y,
                    (1 - alpha) * oof["realmlp"]
                    + alpha * oof["realmlp_31415"],
                    fit,
                )
                for alpha in seed_grid
            ]
            seed_weight = float(seed_grid[int(np.argmax(seed_scores))])
        else:
            seed_weight = 0.0
        seed_weights.append(seed_weight)
        fold_oof = dict(oof)
        fold_oof["realmlp"] = (
            (1 - seed_weight) * oof["realmlp"]
            + seed_weight * oof["realmlp_31415"]
        )
        fold_weights = fit_base(y, fold_oof, fit)
        base = sum(
            fold_weights[name] * fold_oof[name] for name in fold_weights
        )
        fold_prediction = base.copy()
        nested_base[heldout] = base[heldout]
        base_weights.append(fold_weights)
        fold_selection = {}
        for pattern in sorted(np.unique(train_pattern)):
            pattern_fit = fit & (train_pattern == pattern)
            best = (score(y, fold_prediction, fit), "base", 0.0)
            for name, probability in fold_oof.items():
                if name == "realmlp_31415":
                    continue
                for alpha in grid[1:]:
                    candidate = fold_prediction.copy()
                    candidate[train_pattern == pattern] = (
                        (1 - alpha) * base[train_pattern == pattern]
                        + alpha * probability[train_pattern == pattern]
                    )
                    value = score(y, candidate, fit)
                    if value > best[0] + 1e-12:
                        best = (value, name, float(alpha))
            _, name, alpha = best
            if name != "base":
                rows = train_pattern == pattern
                fold_prediction[rows] = (
                    (1 - alpha) * base[rows] + alpha * oof[name][rows]
                )
            fold_selection[pattern] = {"model": name, "alpha": alpha,
                                       "fit_rows": int(pattern_fit.sum())}
        nested[heldout] = fold_prediction[heldout]
        selections.append({
            "fold": fold,
            "realmlp_seed31415_weight": seed_weight,
            "base_weights": fold_weights,
            "selection": fold_selection,
            "heldout_score": score(y, fold_prediction, heldout),
            "base_heldout_score": score(y, base, heldout),
        })

    # Median fold-selected RealMLP seed weight and base-family weights.
    seed_weight = float(np.median(seed_weights))
    oof["realmlp"] = (
        (1 - seed_weight) * oof["realmlp"]
        + seed_weight * oof["realmlp_31415"]
    )
    test_probability["realmlp"] = (
        (1 - seed_weight) * test_probability["realmlp"]
        + seed_weight * test_probability["realmlp_31415"]
    )
    base_names = ["realmlp", "xgb", "te_hgb"]
    weights = {
        name: float(np.median([item.get(name, 0.0) for item in base_weights]))
        for name in base_names
    }
    weights = {name: value for name, value in weights.items() if value > 0}
    total_weight = sum(weights.values())
    weights = {name: value / total_weight for name, value in weights.items()}
    base = sum(weights[name] * oof[name] for name in weights)
    base_test = sum(weights[name] * test_probability[name] for name in weights)

    # Only routes selected identically by a majority of meta-folds are eligible.
    consensus = {}
    for pattern in sorted(np.unique(train_pattern)):
        choices = [
            (entry["selection"][pattern]["model"],
             entry["selection"][pattern]["alpha"])
            for entry in selections
        ]
        counts = {choice: choices.count(choice) for choice in set(choices)}
        choice, count = max(counts.items(), key=lambda item: (item[1], item[0]))
        if count >= 4 and choice[0] != "base":
            consensus[pattern] = {
                "model": choice[0], "alpha": choice[1], "votes": count
            }

    all_rows = np.ones(len(train), bool)
    nested_delta = (
        score(y, nested, all_rows) - score(y, nested_base, all_rows)
    )
    candidate_consensus = consensus
    consensus = candidate_consensus if nested_delta > 1e-5 else {}

    final_oof = base.copy()
    final_test = base_test.copy()
    for pattern, route in consensus.items():
        name, alpha = route["model"], route["alpha"]
        train_rows = train_pattern == pattern
        test_rows = test_pattern == pattern
        final_oof[train_rows] = (
            (1 - alpha) * base[train_rows] + alpha * oof[name][train_rows]
        )
        final_test[test_rows] = (
            (1 - alpha) * base_test[test_rows]
            + alpha * test_probability[name][test_rows]
        )

    report = {
        "consensus_realmlp_seed31415_weight": seed_weight,
        "consensus_base_weights": weights,
        "base_score": score(y, base, all_rows),
        "nested_base_score": score(y, nested_base, all_rows),
        "nested_router_score": score(y, nested, all_rows),
        "nested_delta": nested_delta,
        "folds": selections,
        "candidate_consensus_routes": candidate_consensus,
        "consensus_routes": consensus,
        "consensus_score": score(y, final_oof, all_rows),
    }
    suffix = "_seedbag" if args.use_seed_bag else ""
    output = ROOT / f"artifacts/missing_pattern_router_audit{suffix}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    submission = pd.DataFrame({
        "id": test_ids,
        "health_condition": CLASSES[final_test.argmax(1)],
    })
    submission.to_csv(
        ROOT / f"submissions/missing_pattern_router{suffix}.csv", index=False
    )
    print(json.dumps({
        key: report[key] for key in (
            "consensus_realmlp_seed31415_weight",
            "consensus_base_weights", "base_score", "nested_base_score",
            "nested_router_score", "nested_delta",
            "consensus_routes", "consensus_score"
        )
    }, indent=2))


if __name__ == "__main__":
    main()
