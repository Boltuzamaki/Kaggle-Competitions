"""Build the nested-CV-approved source-rule blend without submitting it."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("at-risk", "fit", "unhealthy")
CORE = ("sleep_duration", "stress_level", "physical_activity_level")
ALPHA = 0.075
SOURCE = ROOT / "data/original_source/student_health_dataset_50k.csv"
FTT = ROOT / "kaggle_kernels/monitor_downloads/boltuzamaki/health-risk-ftt-balanced-seed-2027-gpu/testpred_ftt.csv"
REALMLP = ROOT / "kaggle_kernels/monitor_downloads/boltuzamaki/health-risk-realmlp-seed-31415-gpu/test_preds.csv"
TEST = ROOT / "data/test.csv"
OUT = ROOT / "submissions/source_rule_nested_blend.csv"
AUDIT = ROOT / "artifacts/source_rule_nested_blend_audit.json"


def main() -> None:
    class_index = {label: idx for idx, label in enumerate(CLASSES)}
    counts: dict[tuple[str, ...], list[int]] = defaultdict(lambda: [0, 0, 0])
    totals = [0, 0, 0]
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            label = class_index[row["health_condition"]]
            counts[tuple(row[col] for col in CORE)][label] += 1
            totals[label] += 1

    changed_vs_base = 0
    matched = 0
    transitions: dict[str, int] = {}
    with (
        TEST.open(newline="") as test_handle,
        FTT.open(newline="") as ftt_handle,
        REALMLP.open(newline="") as real_handle,
        OUT.open("w", newline="") as out_handle,
    ):
        test_rows = csv.DictReader(test_handle)
        ftt_rows = csv.DictReader(ftt_handle)
        real_rows = csv.DictReader(real_handle)
        writer = csv.writer(out_handle)
        writer.writerow(("id", "health_condition"))
        for row, ftt, real in zip(test_rows, ftt_rows, real_rows):
            if row["id"] != ftt["id"] or row["id"] != real["id"]:
                raise ValueError(f"ID alignment failed at {row['id']}")
            base = [
                0.60 * float(ftt[label]) + 0.40 * float(real[label])
                for label in CLASSES
            ]
            base_idx = max(range(3), key=base.__getitem__)
            key = tuple(row[col] for col in CORE)
            source_counts = counts.get(key) if all(key) else None
            if source_counts:
                matched += 1
                # Balanced source posterior; additive smoothing prevents zeros.
                source_prob = [
                    (source_counts[idx] + 0.2) / (totals[idx] + 0.6)
                    for idx in range(3)
                ]
                normalizer = sum(source_prob)
                source_prob = [value / normalizer for value in source_prob]
                blend = [
                    (1 - ALPHA) * base[idx] + ALPHA * source_prob[idx]
                    for idx in range(3)
                ]
            else:
                blend = base
            pred_idx = max(range(3), key=blend.__getitem__)
            if pred_idx != base_idx:
                changed_vs_base += 1
                transition = f"{CLASSES[base_idx]}->{CLASSES[pred_idx]}"
                transitions[transition] = transitions.get(transition, 0) + 1
            writer.writerow((row["id"], CLASSES[pred_idx]))

    audit = {
        "alpha": ALPHA,
        "base": "0.60 FTT seed2027 + 0.40 RealMLP seed31415",
        "base_oof_balanced_accuracy": 0.9508466169098009,
        "nested_source_blend_balanced_accuracy": 0.950864050773533,
        "nested_baseline_on_audit_folds": 0.9508527885089949,
        "nested_delta": 1.126226453810172e-05,
        "matched_test_rows": matched,
        "changes_vs_base": changed_vs_base,
        "transitions": transitions,
        "submission": str(OUT.relative_to(ROOT)),
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
