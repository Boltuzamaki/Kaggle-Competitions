from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_v3"
RUN = OUTPUT / "ndr_v18"
V15B_SHA = "4345387b72aecd55dd3856b1607ed836ec9f400365fcf3f3b89f8947f1ff2412"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse(value: object) -> np.ndarray:
    text = str(value).strip()
    if not text or text == "nan":
        return np.zeros((0, 5), np.float64)
    values = np.asarray(list(map(float, text.split())), np.float64)
    assert len(values) % 5 == 0
    return values.reshape(-1, 5)


def validate_submission(path: Path) -> dict:
    frame = pd.read_csv(path, dtype={"image_id": str})
    assert list(frame.columns) == ["id", "image_id", "prediction_string"]
    assert len(frame) == 2000 and frame.image_id.nunique() == 2000
    boxes, mass = 0, 0.0
    for value in frame.prediction_string:
        parsed = parse(value)
        for confidence, x, y, width, height in parsed:
            assert 0 < confidence <= 1
            assert 0 <= x <= 1024 and 0 <= y <= 1024
            assert width > 0 and height > 0
            assert x + width <= 1024.05 and y + height <= 1024.05
        boxes += len(parsed)
        mass += float(parsed[:, 0].sum())
    assert boxes == 3995
    return {"rows": 2000, "unique_ids": 2000, "boxes": boxes, "confidence_mass": mass, "sha256": sha256(path)}


def main() -> None:
    variants = [
        "V18_0_exact_v15b", "V18_A_high_to21", "V18_B_high_restore45",
        "V18_C_two_tier", "V18_D_high_restore70", "V18_E_broad_restore35",
    ]
    required = [
        RUN / "selection_lock.json", RUN / "data_manifest.json", RUN / "cross_domain_audit.json",
        RUN / "training_history.csv", RUN / "per_box_diagnostics.csv", RUN / "final_report.json",
        OUTPUT / "submission.csv",
    ] + [RUN / f"canonical_bundle_{seed}.pth" for seed in (180721, 180722, 180723)]
    required += [OUTPUT / f"submission_{name}.csv" for name in variants]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    assert not missing, missing

    lock = json.loads((RUN / "selection_lock.json").read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "data_manifest.json").read_text(encoding="utf-8"))
    cross = json.loads((RUN / "cross_domain_audit.json").read_text(encoding="utf-8"))
    report = json.loads((RUN / "final_report.json").read_text(encoding="utf-8"))
    assert lock["status"] == "frozen_before_test_enumeration"
    assert lock["rule_7a"]["test_used_for_training_or_selection"] is False
    assert lock["rule_7a"]["test_labels_or_pseudo_labels_created"] is False
    assert lock["rule_7a"]["competition_submission_created"] is False
    assert manifest["test_data_used"] is False
    assert cross["gate_enabled"] is True and cross["test_data_used"] is False
    assert min(cross["external_to_synthetic_auc"], cross["synthetic_to_external_auc"]) >= cross["required_ensemble_auc"]
    assert cross["selected_experts"] == ["profile_mlp", "physics_mlp"]
    assert cross["high_validation_precision"] >= 0.98
    assert cross["low_validation_precision"] >= 0.95
    assert cross["flip_decision_stability"] >= 0.90
    assert report["v15b_exact"] is True and report["v15b_reproduced_sha256"] == V15B_SHA
    assert report["rule_7a_guard_passed"] is True
    assert report["test_used_for_training_or_selection"] is False
    assert report["competition_submission_created"] is False

    checkpoints = {}
    for seed in (180721, 180722, 180723):
        path = RUN / f"canonical_bundle_{seed}.pth"
        assert zipfile.is_zipfile(path)
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            names = archive.namelist()
            assert any(name.endswith("data.pkl") for name in names) and len(names) > 10
        checkpoints[str(seed)] = {"sha256": sha256(path), "valid_torch_zip": True}

    history = pd.read_csv(RUN / "training_history.csv")
    assert set(history.seed.unique()) == {180752, 180753, 180721, 180722, 180723}
    assert np.isfinite(history.loss).all()
    diagnostics = pd.read_csv(RUN / "per_box_diagnostics.csv", dtype={"image_id": str})
    assert len(diagnostics) == 3995
    assert diagnostics.eligible_epsilon.sum() == 3007
    original_epsilon = diagnostics.anchor <= 0.020001
    incumbent_vetoes = (diagnostics.anchor > 0.020001) & diagnostics.eligible_epsilon
    assert int(original_epsilon.sum()) == 2936 and int(incumbent_vetoes.sum()) == 71
    assert (diagnostics.original >= diagnostics.incumbent_v15b - 1e-6).where(diagnostics.eligible_epsilon, True).all()

    submissions = {path.name: validate_submission(path) for path in sorted(OUTPUT.glob("submission*.csv"))}
    assert len(submissions) == 7
    assert submissions["submission_V18_0_exact_v15b.csv"]["sha256"] == V15B_SHA
    assert submissions["submission.csv"]["sha256"] == submissions["submission_V18_A_high_to21.csv"]["sha256"]

    baseline = pd.read_csv(OUTPUT / "submission_V18_0_exact_v15b.csv", dtype={"image_id": str})
    originals = {(str(row.image_id), int(row.candidate)): float(row.original) for row in diagnostics.itertuples(index=False)}
    comparisons = {}
    for name in variants[1:]:
        candidate = pd.read_csv(OUTPUT / f"submission_{name}.csv", dtype={"image_id": str})
        changed, added_mass = 0, 0.0
        for base_row, new_row in zip(baseline.itertuples(index=False), candidate.itertuples(index=False)):
            base, new = parse(base_row.prediction_string), parse(new_row.prediction_string)
            assert base.shape == new.shape
            assert np.allclose(base[:, 1:], new[:, 1:], atol=1e-8)
            assert np.all(new[:, 0] >= base[:, 0] - 1e-8)
            for index, (before, after) in enumerate(zip(base[:, 0], new[:, 0])):
                if after > before + 1e-8:
                    assert before <= 0.020001
                    assert after <= originals[(str(base_row.image_id), index)] + 1e-6
                    changed += 1
                    added_mass += after - before
        expected = report["variants"][name]
        assert changed == expected["changed_boxes"] == expected["epsilon_promotions"]
        assert abs(added_mass - expected["added_confidence_mass"]) < 2e-3
        comparisons[name] = {"changed_boxes": changed, "added_confidence_mass": added_mass}

    audit = {
        "status": "audited_complete",
        "gate": cross,
        "exact_v15b_hash": True,
        "diagnostic_boxes": len(diagnostics),
        "eligible_incumbent_floor_boxes": int(diagnostics.eligible_epsilon.sum()),
        "original_epsilon_boxes": int(original_epsilon.sum()),
        "v15b_veto_boxes": int(incumbent_vetoes.sum()),
        "v18a_repromoted_v15b_vetoes": int((incumbent_vetoes & (diagnostics.clean_probability >= cross["high_threshold"])).sum()),
        "checkpoints": checkpoints,
        "submissions": submissions,
        "variant_comparisons": comparisons,
        "zero_boxes_added_or_moved": True,
        "zero_promotions_above_original": True,
        "rule_7a_guard_passed": True,
        "competition_submission_created": False,
        "finalists": ["V18_A_high_to21", "V18_B_high_restore45"],
        "recommendation": "V18_A is the safest first score probe; V18_B is the higher-upside second finalist only if V18_A confirms transfer.",
    }
    (ROOT / "local_audit_v3.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
