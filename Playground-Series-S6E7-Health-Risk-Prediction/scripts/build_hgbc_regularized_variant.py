"""Create a more regularized TE-HGBC variant for CPU evaluation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle_kernels/hgbc_seedset_b_cpu/hgbc_seedset_b_cpu.ipynb"
DEST = ROOT / "kaggle_kernels/hgbc_regularized_cpu"


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    replacements = {
        "FOLD_SEEDS = [63, 2027, 7777]": "FOLD_SEEDS = [101, 2029]",
        "learning_rate=0.0627037115235577": "learning_rate=0.045",
        "max_iter=300": "max_iter=420",
        "max_leaf_nodes=33": "max_leaf_nodes=27",
        "min_samples_leaf=298": "min_samples_leaf=180",
        "l2_regularization=0.028912644384523085": "l2_regularization=0.20",
        "hgbc_seedset_b_63_2027_7777": "hgbc_regularized_101_2029",
    }
    for cell in notebook["cells"]:
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else str(source)
        for old, new in replacements.items():
            text = text.replace(old, new)
        cell["source"] = text.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "hgbc_regularized_cpu.ipynb").write_text(
        json.dumps(notebook, indent=1) + "\n"
    )
    metadata = {
        "id": "boltuzamaki/health-risk-te-hgbc-regularized-cpu",
        "title": "Health Risk TE HGBC Regularized CPU",
        "code_file": "hgbc_regularized_cpu.ipynb",
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
