"""Build a second balanced FT-Transformer seed for diversity."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle_kernels/ftt_balanced_seed2027_gpu/ftt_balanced_seed2027_gpu.ipynb"
OUT = ROOT / "kaggle_kernels/ftt_balanced_seed4242_gpu"


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    replacements = {
        "SEED = 2027": "SEED = 4242",
        'EXPERIMENT_ID = "ftt_balanced_seed2027_7fold"': (
            'EXPERIMENT_ID = "ftt_balanced_seed4242_7fold"'
        ),
    }
    for old, new in replacements.items():
        hits = 0
        for cell in notebook["cells"]:
            source = cell.get("source", [])
            text = "".join(source) if isinstance(source, list) else source
            count = text.count(old)
            if count:
                text = text.replace(old, new)
                cell["source"] = (
                    text.splitlines(keepends=True) if isinstance(source, list) else text
                )
                hits += count
        if hits != 1:
            raise RuntimeError(f"Expected one occurrence of {old!r}, found {hits}")

    OUT.mkdir(parents=True, exist_ok=True)
    notebook_name = "ftt_balanced_seed4242_gpu.ipynb"
    (OUT / notebook_name).write_text(json.dumps(notebook, ensure_ascii=True, indent=1))
    metadata = {
        "id": "boltuzamaki/health-risk-ftt-balanced-seed-4242-gpu",
        "title": "Health Risk FTT Balanced Seed 4242 GPU",
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["playground-series-s6e7"],
        "docker_image": (
            "gcr.io/kaggle-private-byod/python@sha256:"
            "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
        ),
        "machine_shape": "NvidiaTeslaT4",
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Built {OUT / notebook_name}")


if __name__ == "__main__":
    main()
