"""Create a second leak-free TE-HGBC seed ensemble for error diversity."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle_kernels/hgbc_seedset_b_cpu/hgbc_seedset_b_cpu.ipynb"
DEST = ROOT / "kaggle_kernels/hgbc_seedset_c_cpu"


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        source = source.replace(
            "FOLD_SEEDS = [63, 2027, 7777]",
            "FOLD_SEEDS = [19, 314, 9001]",
        )
        source = source.replace(
            "hgbc_seedset_b_63_2027_7777",
            "hgbc_seedset_c_19_314_9001",
        )
        cell["source"] = source.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "hgbc_seedset_c_cpu.ipynb").write_text(
        json.dumps(notebook, indent=1) + "\n"
    )
    metadata = {
        "id": "divyanshuboltuzamaki/health-risk-hgbc-seedset-c-cpu",
        "title": "Health Risk TE-HGBC Seedset C CPU",
        "code_file": "hgbc_seedset_c_cpu.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["playground-series-s6e7"],
    }
    (DEST / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
