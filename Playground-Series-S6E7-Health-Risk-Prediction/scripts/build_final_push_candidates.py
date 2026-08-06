"""Build conservative final-week candidates around the best public-LB anchor.

The base is the 0.95238 external anchor submission. New corrections require agreement
between two independently seeded FT-Transformers and a five-seed RealMLP bag.
This is intentionally much stricter than the broad model-panel correction that
scored 0.95167.
"""

from __future__ import annotations

import csv
import json
from contextlib import ExitStack
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "reference_notebooks/today_is_a_new_day/anchor/submission.csv"
PUBLIC_BASE = ROOT / "reference_notebooks/students_be_healthy/output/submission.csv"
HYBRID = ROOT / "reference_notebooks/today_is_a_new_day/output/submission.csv"
OUT = ROOT / "submissions"
AUDIT = ROOT / "artifacts/final_push_candidates_audit.json"
CLASSES = ("at-risk", "fit", "unhealthy")

MODELS = {
    "ftt2027": ROOT / "kaggle_kernels/monitor_downloads/boltuzamaki/health-risk-ftt-balanced-seed-2027-gpu/testpred_ftt.csv",
    "ftt4242": ROOT / "kaggle_kernels/monitor_downloads/boltuzamaki/health-risk-ftt-balanced-seed-4242-gpu/testpred_ftt.csv",
    "realmlp2027": ROOT / "kaggle_kernels/realmlp_seed2027_gpu/output/test_preds.csv",
    "realmlp31415": ROOT / "kaggle_kernels/monitor_downloads/boltuzamaki/health-risk-realmlp-seed-31415-gpu/test_preds.csv",
    "realmlp4242": ROOT / "kaggle_kernels/monitor_downloads/divyanshuboltuzamaki/health-risk-realmlp-seed-4242-gpu/test_preds.csv",
    "realmlp7777": ROOT / "kaggle_kernels/monitor_downloads/boltuzamaki/health-risk-realmlp-variant-7777-gpu/test_preds.csv",
    "realmlp9001": ROOT / "kaggle_kernels/monitor_downloads/boltuzamaki/health-risk-realmlp-capacity-9001-gpu/test_preds.csv",
}


def main() -> None:
    with PUBLIC_BASE.open(newline="") as handle:
        anchor_rows = list(csv.DictReader(handle))
    ids = [row["id"] for row in anchor_rows]
    anchor = [row["health_condition"] for row in anchor_rows]
    with HYBRID.open(newline="") as handle:
        hybrid_rows = list(csv.DictReader(handle))
    if [row["id"] for row in hybrid_rows] != ids:
        raise ValueError("Hybrid IDs are not aligned")

    # FTT gets 60% architecture weight; the RealMLP seed bag gets 40%.
    weights = {
        "ftt2027": 0.30,
        "ftt4242": 0.30,
        "realmlp2027": 0.08,
        "realmlp31415": 0.08,
        "realmlp4242": 0.08,
        "realmlp7777": 0.08,
        "realmlp9001": 0.08,
    }
    thresholds = (0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)
    candidates = {threshold: anchor.copy() for threshold in thresholds}
    changed = {threshold: [] for threshold in thresholds}

    with ExitStack() as stack:
        readers = {
            name: csv.DictReader(stack.enter_context(path.open(newline="")))
            for name, path in MODELS.items()
        }
        for row_idx, rows_tuple in enumerate(zip(*readers.values())):
            rows = dict(zip(readers, rows_tuple))
            if any(rows[name]["id"] != ids[row_idx] for name in readers):
                raise ValueError(f"ID alignment failed at row {row_idx}")
            probs = {
                name: [float(rows[name][label]) for label in CLASSES]
                for name in readers
            }
            votes = {
                name: max(range(3), key=prob.__getitem__)
                for name, prob in probs.items()
            }
            # Both FTT seeds must agree; at least four of five RealMLPs must
            # agree with them. This prevents seed multiplicity from dominating.
            ftt_pred = votes["ftt2027"]
            if votes["ftt4242"] != ftt_pred:
                continue
            real_votes = sum(
                votes[name] == ftt_pred for name in readers if name.startswith("realmlp")
            )
            if real_votes < 4:
                continue
            proposed = CLASSES[ftt_pred]
            if proposed == anchor[row_idx]:
                continue
            anchor_idx = CLASSES.index(anchor[row_idx])
            blend = [
                sum(weights[name] * probs[name][class_idx] for name in readers)
                for class_idx in range(3)
            ]
            margin = blend[ftt_pred] - blend[anchor_idx]
            for threshold in thresholds:
                if margin >= threshold:
                    candidates[threshold][row_idx] = proposed
                    changed[threshold].append(
                        {"row": row_idx, "id": ids[row_idx], "from": anchor[row_idx],
                         "to": proposed, "margin": margin, "realmlp_votes": real_votes}
                    )

    audit = {
        "base": str(PUBLIC_BASE.relative_to(ROOT)),
        "weights": weights,
        "candidates": {},
    }
    for threshold, labels in candidates.items():
        tag = str(threshold).replace(".", "")
        path = OUT / f"anchor_neural_consensus_m{tag}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("id", "health_condition"))
            writer.writerows(zip(ids, labels))
        transitions: dict[str, int] = {}
        for item in changed[threshold]:
            key = f"{item['from']}->{item['to']}"
            transitions[key] = transitions.get(key, 0) + 1
        audit["candidates"][path.name] = {
            "margin_threshold": threshold,
            "changes_vs_anchor": len(changed[threshold]),
            "transitions": transitions,
            "changes": changed[threshold],
        }

    # Two interpretable ablations of the best, strictest set. Balanced accuracy
    # can react differently to corrections involving each minority class.
    strict = changed[0.70]
    for source_class in ("unhealthy", "fit"):
        labels = anchor.copy()
        selected = [item for item in strict if item["from"] == source_class]
        for item in selected:
            labels[item["row"]] = item["to"]
        path = OUT / f"anchor_neural_m07_from_{source_class.replace('-', '_')}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("id", "health_condition"))
            writer.writerows(zip(ids, labels))
        audit["candidates"][path.name] = {
            "margin_threshold": 0.70,
            "source_class": source_class,
            "changes_vs_anchor": len(selected),
            "changes": selected,
        }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({
        name: {k: v for k, v in details.items() if k != "changes"}
        for name, details in audit["candidates"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
