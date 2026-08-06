"""Build conservative external-anchor correction candidates from strong models.

This script intentionally uses only model outputs that cleared roughly 0.9495
OOF balanced accuracy.  It keeps the high-scoring external anchor everywhere
except on rows where a broad, architecture-diverse consensus disagrees.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "reference_notebooks/today_is_a_new_day/anchor/submission.csv"
HYBRID = ROOT / "reference_notebooks/today_is_a_new_day/output/submission.csv"
OUT_DIR = ROOT / "submissions"
AUDIT_PATH = ROOT / "artifacts/anchor_consensus_audit.json"
CLASSES = ("at-risk", "fit", "unhealthy")

# One vote per independently trained pipeline. Seed variants are deliberately
# retained for stability, but architecture families are balanced through the
# weights below so that RealMLP cannot dominate by seed count alone.
MODELS = {
    "ftt2027": (
        "kaggle_kernels/monitor_downloads/boltuzamaki/"
        "health-risk-ftt-balanced-seed-2027-gpu/testpred_ftt.csv",
        1.5,
    ),
    "realmlp2027": ("kaggle_kernels/realmlp_seed2027_gpu/output/test_preds.csv", 0.5),
    "realmlp31415": (
        "kaggle_kernels/monitor_downloads/boltuzamaki/"
        "health-risk-realmlp-seed-31415-gpu/test_preds.csv",
        0.5,
    ),
    "realmlp7777": (
        "kaggle_kernels/monitor_downloads/boltuzamaki/"
        "health-risk-realmlp-variant-7777-gpu/test_preds.csv",
        0.5,
    ),
    "realmlp4242": (
        "kaggle_kernels/monitor_downloads/divyanshuboltuzamaki/"
        "health-risk-realmlp-seed-4242-gpu/test_preds.csv",
        0.5,
    ),
    "hgb_seedbag_b": ("kaggle_kernels/hgbc_seedset_b_cpu/output/tehgbc_test_preds.csv", 0.75),
    "hgb_seedbag_c": (
        "kaggle_kernels/monitor_downloads/divyanshuboltuzamaki/"
        "health-risk-te-hgbc-seedset-c-cpu/tehgbc_test_preds.csv",
        0.75,
    ),
    "xgb_rule": ("kaggle_kernels/rule_xgb_gpu/output/test_preds.csv", 0.75),
    "xgb_te": (
        "kaggle_kernels/monitor_downloads/boltuzamaki/"
        "health-risk-te-xgb-cpu/test_preds.csv",
        0.75,
    ),
    "catboost_te": (
        "kaggle_kernels/monitor_downloads/divyanshuboltuzamaki/"
        "health-risk-te-catboost-cpu/test_preds.csv",
        0.75,
    ),
    "ebm": (
        "kaggle_kernels/monitor_downloads/boltuzamaki/"
        "health-risk-rule-ebm-cpu/test_preds.csv",
        0.75,
    ),
    "ydf": (
        "kaggle_kernels/monitor_downloads/boltuzamaki/"
        "health-risk-ydf-cpu/test_preds.csv",
        0.75,
    ),
}


def read_labels(path: Path) -> tuple[list[str], list[str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [r["id"] for r in rows], [r["health_condition"] for r in rows]


def read_probs(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [r["id"] for r in rows]
    probs = [[float(r[c]) for c in CLASSES] for r in rows]
    return ids, probs


def write_submission(path: Path, ids: list[str], labels: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id", "health_condition"))
        writer.writerows(zip(ids, labels))


def main() -> None:
    ids, anchor = read_labels(ANCHOR)
    hybrid_ids, hybrid = read_labels(HYBRID)
    if ids != hybrid_ids:
        raise ValueError("Anchor and hybrid IDs are not aligned")

    loaded: dict[str, tuple[list[list[float]], float]] = {}
    for name, (relative, weight) in MODELS.items():
        path = ROOT / relative
        if not path.exists():
            print(f"Skipping missing model: {name}: {path}")
            continue
        model_ids, probs = read_probs(path)
        if model_ids != ids:
            raise ValueError(f"IDs are not aligned for {name}")
        loaded[name] = (probs, weight)

    total_weight = sum(weight for _, weight in loaded.values())
    consensus_label: list[str] = []
    agreement: list[float] = []
    mean_margin: list[float] = []
    raw_votes: list[int] = []
    for row_idx in range(len(ids)):
        weighted_votes = [0.0, 0.0, 0.0]
        weighted_probs = [0.0, 0.0, 0.0]
        class_votes = [0, 0, 0]
        for probs, weight in loaded.values():
            row = probs[row_idx]
            pred = max(range(3), key=row.__getitem__)
            weighted_votes[pred] += weight
            class_votes[pred] += 1
            for class_idx in range(3):
                weighted_probs[class_idx] += weight * row[class_idx]
        pred = max(range(3), key=weighted_votes.__getitem__)
        consensus_label.append(CLASSES[pred])
        agreement.append(weighted_votes[pred] / total_weight)
        raw_votes.append(class_votes[pred])
        anchor_idx = CLASSES.index(anchor[row_idx])
        mean_margin.append(
            (weighted_probs[pred] - weighted_probs[anchor_idx]) / total_weight
        )

    hybrid_changed = {i for i, (a, h) in enumerate(zip(anchor, hybrid)) if a != h}
    candidates = {
        # Preserve the proven 190 changes, then add only near-unanimous,
        # high-margin disagreements from the stronger OOF model panel.
        "anchor_hybrid_plus_unanimous": (0.999, 0.20),
        "anchor_hybrid_plus_consensus95": (0.95, 0.25),
        "anchor_hybrid_plus_consensus90": (0.90, 0.30),
        # Independent conservative candidate, useful for measuring whether the
        # original 190 changes or our OOF panel generalizes better.
        "anchor_strong_consensus_only": (0.95, 0.25),
    }

    audit: dict[str, object] = {
        "models": {k: {"path": MODELS[k][0], "weight": v[1]} for k, v in loaded.items()},
        "anchor_rows": len(ids),
        "hybrid_changes": len(hybrid_changed),
        "hybrid_change_consensus": {},
        "candidates": {},
    }
    for threshold in (0.5, 0.75, 0.9, 0.95, 0.999):
        matching = sum(
            consensus_label[i] == hybrid[i] and agreement[i] >= threshold
            for i in hybrid_changed
        )
        audit["hybrid_change_consensus"][str(threshold)] = matching

    for name, (min_agreement, min_margin) in candidates.items():
        additions = {
            i
            for i in range(len(ids))
            if consensus_label[i] != anchor[i]
            and agreement[i] >= min_agreement
            and mean_margin[i] >= min_margin
        }
        if name == "anchor_strong_consensus_only":
            changed = additions
        else:
            changed = hybrid_changed | additions
        labels = anchor.copy()
        for i in changed:
            labels[i] = hybrid[i] if i in hybrid_changed else consensus_label[i]
        out_path = OUT_DIR / f"{name}.csv"
        write_submission(out_path, ids, labels)
        audit["candidates"][name] = {
            "agreement_threshold": min_agreement,
            "margin_threshold": min_margin,
            "total_changes": len(changed),
            "consensus_changes": len(additions),
            "new_beyond_hybrid": len(additions - hybrid_changed),
            "hybrid_changes_replaced_by_consensus": sum(
                i in additions and consensus_label[i] != hybrid[i] for i in hybrid_changed
            ),
            "path": str(out_path.relative_to(ROOT)),
        }

    AUDIT_PATH.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
