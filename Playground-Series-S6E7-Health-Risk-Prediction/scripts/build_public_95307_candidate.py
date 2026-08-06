"""Reconstruct and validate the public 0.95307 post-processing candidate."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "public_research/datasets/yw_pack/V98_anchor_0.95303.csv"
CLEANUP_NOTEBOOK = (
    ROOT
    / "public_research/yw8837_95306/public-lb-0-95306-reproducible-29-row-cleanup.py"
)
CALIBRATION_NOTEBOOK = (
    ROOT / "public_research/najiama_95307/post-processing-calibration-lb-0-95307.ipynb"
)
SAMPLE = ROOT / "data/sample_submission.csv"
OUTPUT = ROOT / "submissions/public_reconstructed_95307.csv"
AUDIT = ROOT / "artifacts/public_reconstructed_95307_audit.json"
TARGET = "health_condition"
EXPECTED_ANCHOR_HASH = "ef93d5fff231341b8ddff364974edb4732a5cbc227365a0d063ae3b36adeeb35"
EXPECTED_V114_HASH = "4171d46147b36cea16af087fed7be36bfd9b4bd48098b3d5d20a129123afbb81"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def notebook_source(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        if isinstance(cell.get("source"), list)
        else cell.get("source", "")
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    )


def literal_assignment(source: str, name: str) -> dict:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise TypeError(f"{name} is not a dictionary")
            return value
    raise KeyError(f"Could not find {name}")


def validate(frame: pd.DataFrame, sample: pd.DataFrame) -> None:
    assert frame.columns.tolist() == ["id", TARGET]
    assert len(frame) == len(sample) == 295_753
    assert frame["id"].equals(sample["id"])
    assert frame["id"].is_unique and not frame.isna().any().any()
    assert set(frame[TARGET]) == {"fit", "at-risk", "unhealthy"}


def main() -> None:
    assert sha256(ANCHOR) == EXPECTED_ANCHOR_HASH
    sample = pd.read_csv(SAMPLE)
    anchor = pd.read_csv(ANCHOR)
    validate(anchor, sample)

    cleanup = literal_assignment(notebook_source(CLEANUP_NOTEBOOK), "CORRECTIONS")
    v114 = anchor.copy()
    indexed = v114.set_index("id")[TARGET]
    for row_id, (before, after) in cleanup.items():
        assert indexed.loc[int(row_id)] == before
        v114.loc[v114["id"].eq(int(row_id)), TARGET] = after

    temporary = OUTPUT.with_name(".public_reconstructed_v114.csv")
    v114.to_csv(temporary, index=False, lineterminator="\n")
    assert sha256(temporary) == EXPECTED_V114_HASH
    temporary.unlink()

    overrides = literal_assignment(notebook_source(CALIBRATION_NOTEBOOK), "OVERRIDES")
    candidate = v114.copy()
    mapped = candidate["id"].map({int(k): v for k, v in overrides.items()})
    candidate[TARGET] = mapped.fillna(candidate[TARGET])
    validate(candidate, sample)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(OUTPUT, index=False, lineterminator="\n")

    changed = candidate[TARGET].ne(v114[TARGET])
    transitions = (
        pd.DataFrame({"from": v114.loc[changed, TARGET], "to": candidate.loc[changed, TARGET]})
        .value_counts()
        .rename("rows")
    )
    audit = {
        "output": str(OUTPUT.relative_to(ROOT)),
        "rows": len(candidate),
        "sha256": sha256(OUTPUT),
        "v114_sha256": EXPECTED_V114_HASH,
        "cleanup_rows": len(cleanup),
        "override_manifest_rows": len(overrides),
        "effective_changes_vs_95306": int(changed.sum()),
        "transitions": {
            f"{before}->{after}": int(count)
            for (before, after), count in transitions.items()
        },
        "class_counts": candidate[TARGET].value_counts().to_dict(),
    }
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
