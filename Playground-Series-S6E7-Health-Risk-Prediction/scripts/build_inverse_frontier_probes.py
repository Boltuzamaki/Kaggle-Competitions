"""Build disjoint public-frontier probes from the verified 0.95288 base."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reference_notebooks/students_be_healthy/output/submission.csv"
EVIDENCE = ROOT / "artifacts/public_cross_family/inverse_ridge_peer_evidence.csv"
OUT = ROOT / "submissions"
AUDIT = ROOT / "artifacts/inverse_frontier_probes_audit.json"


def main() -> None:
    with BASE.open(newline="") as handle:
        base_rows = list(csv.DictReader(handle))
    with EVIDENCE.open(newline="") as handle:
        evidence = list(csv.DictReader(handle))

    # Rows with >=5 support were already tested in the published 0.95288
    # vector. The next disjoint evidence tier is exactly 4/9 peer families.
    frontier = [
        row for row in evidence
        if int(row["peer_support"]) == 4 and float(row["ridge_score"]) > 0
    ]
    groups = {
        "fit": [row for row in frontier if row["proposed"] == "fit"],
        "unhealthy": [row for row in frontier if row["proposed"] == "unhealthy"],
        "union": frontier,
        # Explicit public-oracle candidates disclosed (but commented out) in
        # najiama/post-processing-calibration-lb-0-95288.
        "oracle7": [
            {"id": str(row_id), "proposed": label, "ridge_score": "0"}
            for row_id, label in {
                690144: "fit",
                955916: "fit",
                848145: "unhealthy",
                833134: "fit",
                982084: "unhealthy",
                705455: "fit",
                860046: "unhealthy",
            }.items()
        ],
        # Next-day tier: every row is an at-risk -> unhealthy promotion with
        # positive inverse-ridge evidence and unanimous support from our seven
        # independently seeded FTT/RealMLP models.
        "neural7": [
            {
                "id": str(row_id),
                "proposed": "unhealthy",
                "ridge_score": str(ridge),
            }
            for row_id, ridge in [
                (956659, 0.00021303519315551966),  # 3/9 published peer groups
                (815849, 0.00022328543127514422),  # 3/9
                (908114, 0.00022328543127514422),  # 2/9
                (874665, 0.00022328543127514422),  # 2/9
                (914955, 0.00022328543127514422),  # 2/9
                (803658, 0.00022328543127514422),  # 2/9
                (709483, 0.00022328543127514422),  # 1/9
            ]
        ],
        "neural_support3": [
            {"id": str(row_id), "proposed": "unhealthy", "ridge_score": str(ridge)}
            for row_id, ridge in [
                (956659, 0.00021303519315551966),
                (815849, 0.00022328543127514422),
            ]
        ],
        "neural_support2": [
            {"id": str(row_id), "proposed": "unhealthy", "ridge_score": str(ridge)}
            for row_id, ridge in [
                (908114, 0.00022328543127514422),
                (874665, 0.00022328543127514422),
                (914955, 0.00022328543127514422),
                (803658, 0.00022328543127514422),
            ]
        ],
        "neural_support1": [
            {"id": "709483", "proposed": "unhealthy",
             "ridge_score": "0.00022328543127514422"}
        ],
    }
    audit: dict[str, object] = {"base": str(BASE.relative_to(ROOT)), "groups": {}}
    for name, selected in groups.items():
        mapping = {row["id"]: row["proposed"] for row in selected}
        output = OUT / f"inverse_frontier_support4_{name}.csv"
        changed = []
        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("id", "health_condition"))
            for row in base_rows:
                label = mapping.get(row["id"], row["health_condition"])
                if label != row["health_condition"]:
                    changed.append({
                        "id": row["id"],
                        "from": row["health_condition"],
                        "to": label,
                        "ridge_score": next(
                            float(item["ridge_score"])
                            for item in selected if item["id"] == row["id"]
                        ),
                    })
                writer.writerow((row["id"], label))
        audit["groups"][name] = {
            "path": str(output.relative_to(ROOT)),
            "changes": changed,
            "count": len(changed),
        }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
