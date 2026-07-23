import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parent / "audit_files_v3"
RUN = ROOT / "ndr_v14"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_csv(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2000, (path.name, len(rows))
    assert set(rows[0]) == {"id", "image_id", "prediction_string"}, path.name
    ids = [int(row["id"]) for row in rows]
    image_ids = [int(row["image_id"]) for row in rows]
    assert set(ids) == set(image_ids) == set(range(2000)), path.name
    boxes = nonempty = 0
    score_min, score_max, score_mass = 1.0, 0.0, 0.0
    for row in rows:
        values = row["prediction_string"].strip().split()
        assert len(values) % 5 == 0, (path.name, row["id"], len(values))
        nonempty += bool(values)
        for offset in range(0, len(values), 5):
            score, x, y, width, height = map(float, values[offset : offset + 5])
            assert 0.0 < score <= 1.0
            assert x >= 0.0 and y >= 0.0 and width > 0.0 and height > 0.0
            assert x + width <= 1024.05 and y + height <= 1024.05
            score_min = min(score_min, score)
            score_max = max(score_max, score)
            score_mass += score
            boxes += 1
    return {
        "rows": len(rows), "unique_ids": len(set(ids)), "boxes": boxes,
        "nonempty": nonempty, "score_min": score_min, "score_max": score_max,
        "score_mass": score_mass, "sha256": sha256(path),
    }


report = json.loads((RUN / "final_report.json").read_text(encoding="utf-8"))
lock = json.loads((RUN / "selection_lock.json").read_text(encoding="utf-8"))
manifest = json.loads((RUN / "external_data_manifest.json").read_text(encoding="utf-8"))
assert report["status"] == "complete"
assert report["test_used_for_selection"] is False
assert report["competition_submission_created"] is False
assert report["rule_7a_guard_passed"] is True
assert manifest["source"] == "sanidhyavijay24/streaksyolodataset"
assert manifest["test_data_used"] is False
assert manifest["train_labelled_images"] == 1193
assert manifest["validation_labelled_images"] == 225
assert lock["rule_7a"]["test_used_for_training_or_selection"] is False
assert lock["rule_7a"]["test_pseudo_labels"] is False
assert lock["rule_7a"]["competition_submission_created"] is False

audited = {}
for variant, declared in report["variants"].items():
    path = ROOT / Path(declared["path"]).name
    result = audit_csv(path)
    assert result["boxes"] == declared["boxes"]
    assert result["nonempty"] == declared["nonempty"]
    assert result["sha256"] == declared["sha256"]
    audited[variant] = result

alias = audit_csv(ROOT / "submission.csv")
assert alias["sha256"] == report["alias_sha256"]
assert alias["sha256"] == audited[report["alias"]]["sha256"]

events = [json.loads(line) for line in (RUN / "run.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
messages = [event.get("message") for event in events]
assert "SELECTION_LOCK_WRITTEN" in messages
test_events = [i for i, event in enumerate(events) if "TEST" in str(event.get("message", "")).upper()]
assert test_events and messages.index("SELECTION_LOCK_WRITTEN") < min(test_events)

summary = {
    "status": "passed", "rule_7a_guard": True,
    "selection_frozen_before_test": True, "external_data": manifest,
    "ranker": report["ranker"], "pcgrad": report["pcgrad"],
    "variants": audited, "alias": {"name": report["alias"], **alias},
}
(ROOT / "local_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
