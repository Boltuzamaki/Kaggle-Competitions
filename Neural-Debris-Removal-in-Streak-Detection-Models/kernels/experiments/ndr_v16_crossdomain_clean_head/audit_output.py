from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_v1"
RUN = OUTPUT / "ndr_v16"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_submission(path: Path) -> dict:
    frame = pd.read_csv(path, dtype={"image_id": str})
    assert list(frame.columns) in (["image_id", "prediction_string"], ["id", "image_id", "prediction_string"])
    assert len(frame) == 2000 and frame.image_id.nunique() == 2000
    boxes = 0
    confidence_mass = 0.0
    for value in frame.prediction_string:
        text = str(value).strip()
        if not text or text == "nan":
            continue
        values = list(map(float, text.split()))
        assert len(values) % 5 == 0
        for index in range(0, len(values), 5):
            confidence, x, y, width, height = values[index : index + 5]
            assert 0 < confidence <= 1
            assert 0 <= x <= 1024 and 0 <= y <= 1024
            assert width > 0 and height > 0
            assert x + width <= 1024.05 and y + height <= 1024.05
            boxes += 1
            confidence_mass += confidence
    return {
        "rows": len(frame),
        "unique_ids": int(frame.image_id.nunique()),
        "boxes": boxes,
        "confidence_mass": confidence_mass,
        "sha256": sha256(path),
    }


def main() -> None:
    required = [
        RUN / "selection_lock.json",
        RUN / "data_manifest.json",
        RUN / "cross_domain_audit.json",
        RUN / "per_box_diagnostics.csv",
        RUN / "final_report.json",
        OUTPUT / "submission.csv",
    ] + [RUN / f"clean_head_seed_{seed}.pth" for seed in (160721, 160722, 160723)]
    assert all(path.exists() for path in required), [str(path) for path in required if not path.exists()]

    lock = json.loads((RUN / "selection_lock.json").read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "data_manifest.json").read_text(encoding="utf-8"))
    cross = json.loads((RUN / "cross_domain_audit.json").read_text(encoding="utf-8"))
    report = json.loads((RUN / "final_report.json").read_text(encoding="utf-8"))
    assert lock["status"] == "frozen_before_test_enumeration"
    assert lock["inference"] == {"box_bank": "exact V12/M1 only", "boxes_added": 0, "boxes_moved": 0, "confidence_increases": 0}
    assert manifest["test_data_used"] is False
    assert cross["head_enabled"] is False
    assert cross["external_to_synthetic_auc"] < cross["minimum_required_auc"]
    assert cross["synthetic_to_external_auc"] < cross["minimum_required_auc"]
    assert report["anchor_exact"] is True
    assert report["rule_7a_guard_passed"] is True
    assert report["test_used_for_training_or_selection"] is False
    assert report["competition_submission_created"] is False

    checkpoints = {}
    for seed in (160721, 160722, 160723):
        path = RUN / f"clean_head_seed_{seed}.pth"
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state["seed"] == seed and state["input_dimension"] > 0 and "model" in state
        checkpoints[str(seed)] = {"sha256": sha256(path), "input_dimension": state["input_dimension"]}

    diagnostics = pd.read_csv(RUN / "per_box_diagnostics.csv", dtype={"image_id": str})
    assert len(diagnostics) == 3995
    assert diagnostics.image_id.nunique() <= 2000

    submissions = {}
    for path in sorted(OUTPUT.glob("submission*.csv")):
        submissions[path.name] = validate_submission(path)
    hashes = {value["sha256"] for value in submissions.values()}
    assert len(hashes) == 1
    assert next(iter(hashes)) == report["anchor_sha256"] == report["anchor_reproduced_sha256"]
    for variant in report["variants"].values():
        assert variant["boxes_added"] == 0 and variant["confidence_increases"] == 0
        assert variant["changed_boxes"] == 0 and variant["removed_confidence_mass"] == 0

    audit = {
        "status": "audited_complete",
        "head_enabled": cross["head_enabled"],
        "gate": cross,
        "anchor_exact": report["anchor_exact"],
        "diagnostic_boxes": len(diagnostics),
        "all_submission_hashes_identical": len(hashes) == 1,
        "checkpoints": checkpoints,
        "submissions": submissions,
        "zero_boxes_added_or_moved": True,
        "zero_confidence_increases": True,
        "rule_7a_guard_passed": True,
        "competition_submission_created": False,
        "recommendation": "Do not submit any V16 CSV; all are the already-scored V12 anchor.",
    }
    (ROOT / "local_audit_v1.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
