"""Nested meta-fold audit for aligned OOF/test probability artifacts.

Model inclusion and blend weights are chosen on six deterministic meta-folds
and evaluated on the untouched seventh. This is intentionally stricter than
optimizing weights on all OOF rows and reporting that same score.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLASSES = np.array(["at-risk", "fit", "unhealthy"])
CLASS_TO_INT = {value: index for index, value in enumerate(CLASSES)}


@dataclass
class Artifact:
    name: str
    oof: Path
    test: Path


BASE_ARTIFACTS = [
    Artifact(
        "realmlp",
        ROOT / "kaggle_kernels/realmlp_seed2027_gpu/output/oof_preds.csv",
        ROOT / "kaggle_kernels/realmlp_seed2027_gpu/output/test_preds.csv",
    ),
    Artifact(
        "xgb_rule",
        ROOT / "kaggle_kernels/rule_xgb_gpu/output/oof_preds.csv",
        ROOT / "kaggle_kernels/rule_xgb_gpu/output/test_preds.csv",
    ),
    Artifact(
        "hgb_rule",
        ROOT / "kaggle_kernels/rule_hgb_cpu/output/oof_preds.csv",
        ROOT / "kaggle_kernels/rule_hgb_cpu/output/test_preds.csv",
    ),
    Artifact(
        "tabnet",
        ROOT / "kaggle_kernels/monitor_downloads/boltuzamaki"
        / "health-risk-rule-tabnet-gpu/oof_preds.csv",
        ROOT / "kaggle_kernels/monitor_downloads/boltuzamaki"
        / "health-risk-rule-tabnet-gpu/test_preds.csv",
    ),
]


def score(y: np.ndarray, probabilities: np.ndarray, mask: np.ndarray) -> float:
    prediction = probabilities.argmax(axis=1)
    recalls = [
        np.mean(prediction[mask & (y == cls)] == cls)
        for cls in range(len(CLASSES))
    ]
    return float(np.mean(recalls))


def load_probabilities(path: Path, ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path).sort_values("id")
    if not np.array_equal(frame["id"].to_numpy(), ids):
        raise ValueError(f"ID mismatch: {path}")
    values = frame[list(CLASSES)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite probabilities: {path}")
    return values / values.sum(axis=1, keepdims=True)


def greedy_fit(
    y: np.ndarray,
    models: dict[str, np.ndarray],
    mask: np.ndarray,
    anchor: str = "realmlp",
    max_models: int = 5,
) -> tuple[dict[str, float], np.ndarray]:
    weights = {anchor: 1.0}
    blend = models[anchor].copy()
    current = score(y, blend, mask)
    remaining = set(models) - {anchor}
    grid = np.linspace(0.025, 0.50, 20)
    while remaining and len(weights) < max_models:
        best: tuple[float, str, float, np.ndarray] | None = None
        for name in sorted(remaining):
            for incoming_weight in grid:
                candidate = (
                    (1.0 - incoming_weight) * blend
                    + incoming_weight * models[name]
                )
                candidate_score = score(y, candidate, mask)
                record = (candidate_score, name, float(incoming_weight), candidate)
                if best is None or record[0] > best[0]:
                    best = record
        assert best is not None
        if best[0] <= current + 1e-5:
            break
        current, name, incoming_weight, blend = best
        weights = {
            model_name: value * (1.0 - incoming_weight)
            for model_name, value in weights.items()
        }
        weights[name] = incoming_weight
        remaining.remove(name)
    return weights, blend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/stable_ensemble_audit.json")
    args = parser.parse_args()

    artifacts = list(BASE_ARTIFACTS)
    if args.extra_manifest and args.extra_manifest.exists():
        for item in json.loads(args.extra_manifest.read_text()):
            artifacts.append(
                Artifact(item["name"], Path(item["oof"]), Path(item["test"]))
            )
    artifacts = [
        item
        for item in artifacts
        if item.oof.exists()
        and item.test.exists()
        and item.oof.stat().st_size > 0
        and item.test.stat().st_size > 0
    ]
    if not any(item.name == "realmlp" for item in artifacts):
        raise RuntimeError("RealMLP anchor artifact is required")

    train = pd.read_csv(ROOT / "data/train.csv", usecols=["id", "health_condition"]).sort_values("id")
    test = pd.read_csv(ROOT / "data/test.csv", usecols=["id"]).sort_values("id")
    train_ids = train["id"].to_numpy()
    test_ids = test["id"].to_numpy()
    y = train["health_condition"].map(CLASS_TO_INT).to_numpy()
    oof = {item.name: load_probabilities(item.oof, train_ids) for item in artifacts}
    test_prob = {item.name: load_probabilities(item.test, test_ids) for item in artifacts}

    meta_fold = train_ids % 7
    fold_results = []
    selected_weights = []
    heldout_prediction = np.zeros((len(train), len(CLASSES)), dtype=np.float64)
    for fold in range(7):
        fit_mask = meta_fold != fold
        validation_mask = ~fit_mask
        weights, _ = greedy_fit(y, oof, fit_mask)
        validation_blend = sum(weights[name] * oof[name] for name in weights)
        heldout_prediction[validation_mask] = validation_blend[validation_mask]
        result = {
            "fold": fold,
            "weights": weights,
            "fit_score": score(y, validation_blend, fit_mask),
            "heldout_score": score(y, validation_blend, validation_mask),
            "anchor_heldout_score": score(y, oof["realmlp"], validation_mask),
        }
        result["heldout_delta"] = (
            result["heldout_score"] - result["anchor_heldout_score"]
        )
        fold_results.append(result)
        selected_weights.append(weights)

    nested_score = score(y, heldout_prediction, np.ones(len(y), dtype=bool))
    anchor_score = score(y, oof["realmlp"], np.ones(len(y), dtype=bool))

    # Consensus final weights: median per model, then normalize. Models absent
    # from a fold count as zero, penalizing unstable selections.
    names = sorted(oof)
    consensus = {
        name: float(np.median([weights.get(name, 0.0) for weights in selected_weights]))
        for name in names
    }
    consensus = {name: value for name, value in consensus.items() if value > 0}
    total = sum(consensus.values())
    consensus = {name: value / total for name, value in consensus.items()}
    final_oof = sum(consensus[name] * oof[name] for name in consensus)
    final_test = sum(consensus[name] * test_prob[name] for name in consensus)
    full_oof_score = score(y, final_oof, np.ones(len(y), dtype=bool))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "models": names,
        "solo_scores": {
            name: score(y, values, np.ones(len(y), dtype=bool))
            for name, values in oof.items()
        },
        "fold_results": fold_results,
        "nested_meta_oof_score": nested_score,
        "realmlp_anchor_score": anchor_score,
        "nested_delta": nested_score - anchor_score,
        "consensus_weights": consensus,
        "consensus_full_oof_score": full_oof_score,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    submission = pd.DataFrame(
        {"id": test_ids, "health_condition": CLASSES[final_test.argmax(axis=1)]}
    )
    submission_path = ROOT / "submissions/stable_nested_ensemble.csv"
    submission.to_csv(submission_path, index=False)
    print(json.dumps(report, indent=2))
    print(submission_path)


if __name__ == "__main__":
    main()
