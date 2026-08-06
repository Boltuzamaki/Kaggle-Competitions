"""Build a validated micro-ablation from two aligned submissions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--ids", required=True, help="Comma-separated IDs to restore")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_path = Path(args.base)
    reference_path = Path(args.reference)
    output_path = Path(args.output)
    base = pd.read_csv(base_path)
    reference = pd.read_csv(reference_path)
    ids = {int(value) for value in args.ids.split(",") if value}

    assert base.columns.tolist() == reference.columns.tolist() == ["id", "health_condition"]
    assert len(base) == len(reference) == 295_753
    assert base["id"].equals(reference["id"])
    assert ids <= set(base["id"])

    candidate = base.copy()
    ref_labels = reference.set_index("id")["health_condition"]
    mask = candidate["id"].isin(ids)
    candidate.loc[mask, "health_condition"] = candidate.loc[mask, "id"].map(ref_labels)
    assert int(mask.sum()) == len(ids)
    assert not candidate.isna().any().any()
    assert set(candidate["health_condition"]) == {"at-risk", "fit", "unhealthy"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(output_path, index=False, lineterminator="\n")
    print(json.dumps({
        "output": str(output_path),
        "restored_ids": sorted(ids),
        "effective_changes_vs_base": int(
            candidate["health_condition"].ne(base["health_condition"]).sum()
        ),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
