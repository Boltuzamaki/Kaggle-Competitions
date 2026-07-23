import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parent / "audit_files_v2"
RUN = ROOT / "ndr_v13"


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
    assert len(set(ids)) == len(set(image_ids)) == 2000, path.name
    assert set(ids) == set(image_ids) == set(range(2000)), path.name
    boxes = 0
    nonempty = 0
    score_min, score_max = 1.0, 0.0
    for row in rows:
        values = row["prediction_string"].strip().split()
        assert len(values) % 5 == 0, (path.name, row["id"], len(values))
        if values:
            nonempty += 1
        for offset in range(0, len(values), 5):
            score, x, y, width, height = map(float, values[offset : offset + 5])
            assert 0.0 <= score <= 1.0, (path.name, row["id"], score)
            assert x >= 0.0 and y >= 0.0 and width > 0.0 and height > 0.0
            assert x + width <= 1024.05 and y + height <= 1024.05
            score_min = min(score_min, score)
            score_max = max(score_max, score)
            boxes += 1
    return {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "boxes": boxes,
        "nonempty": nonempty,
        "score_min": score_min,
        "score_max": score_max,
        "sha256": sha256(path),
    }


report = json.loads((RUN / "final_report.json").read_text(encoding="utf-8"))
lock = json.loads((RUN / "selection_lock.json").read_text(encoding="utf-8"))
assert report["status"] == "complete"
assert report["test_images"] == 2000
assert report["test_used_for_selection"] is False
assert report["competition_submission_created"] is False
assert report["rule_7a_guard_passed"] is True
assert lock["rule_7a"]["test_pixels_used_for_training_or_selection"] is False
assert lock["rule_7a"]["competition_submission_created"] is False

audited = {}
for variant, declared in report["variants"].items():
    path = ROOT / Path(declared["path"]).name
    result = audit_csv(path)
    assert result["boxes"] == declared["boxes"], (variant, result["boxes"], declared["boxes"])
    assert result["nonempty"] == declared["nonempty"], variant
    assert result["sha256"] == declared["sha256"], variant
    audited[variant] = result

alias = audit_csv(ROOT / "submission.csv")
assert alias["sha256"] == report["alias_sha256"]
assert alias["sha256"] == audited[report["alias"]]["sha256"]

run_log = RUN / "run.jsonl"
if not run_log.exists():
    run_log = Path(__file__).parent / "output_v2" / "ndr_v13" / "run.jsonl"
events = [json.loads(line) for line in run_log.read_text(encoding="utf-8").splitlines() if line.strip()]
messages = [event.get("message") for event in events]
assert "SELECTION_LOCK_WRITTEN" in messages
test_events = [i for i, event in enumerate(events) if "TEST" in str(event.get("message", "")).upper()]
assert not test_events or messages.index("SELECTION_LOCK_WRITTEN") < min(test_events)

summary = {
    "status": "passed",
    "rule_7a_guard": True,
    "selection_frozen_before_test": True,
    "ranker": report["ranker"],
    "task_vector_selected": report["task_vector_selected"],
    "projection_selected": report["projection_selected"],
    "variants": audited,
    "alias": {"name": report["alias"], **alias},
}
(ROOT / "local_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
