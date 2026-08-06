"""Build a pure-seed replica of the proven RealMLP anchor configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle_kernels/realmlp_seed2027_gpu/realmlp_seed2027_gpu.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()
    seed = args.seed
    destination = ROOT / f"kaggle_kernels/realmlp_seed{seed}_gpu"
    notebook = json.loads(SOURCE.read_text())
    replacements = {
        "seed_everything(2027)": f"seed_everything({seed})",
        "torch.cuda.manual_seed_all(2027)": f"torch.cuda.manual_seed_all({seed})",
        '"random_state": 2027': f'"random_state": {seed}',
        "SEED = 2027": f"SEED = {seed}",
        'EXPERIMENT_ID = "realmlp_seed2027_gpu_7fold_3epoch"':
            f'EXPERIMENT_ID = "realmlp_seed{seed}_gpu_7fold_3epoch"',
    }
    for old, new in replacements.items():
        hits = 0
        for cell in notebook["cells"]:
            source = cell.get("source", [])
            text = "".join(source) if isinstance(source, list) else str(source)
            count = text.count(old)
            if count:
                text = text.replace(old, new)
                cell["source"] = text.splitlines(keepends=True)
                hits += count
            cell["outputs"] = []
            cell["execution_count"] = None
        if hits != 1:
            raise RuntimeError(f"Expected one replacement for {old!r}, got {hits}")

    destination.mkdir(parents=True, exist_ok=True)
    notebook_name = f"realmlp_seed{seed}_gpu.ipynb"
    (destination / notebook_name).write_text(
        json.dumps(notebook, indent=1) + "\n"
    )
    metadata = {
        "id": f"boltuzamaki/health-risk-realmlp-seed-{seed}-gpu",
        "title": f"Health Risk RealMLP Seed {seed} GPU",
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["playground-series-s6e7"],
        "machine_shape": "NvidiaTeslaT4",
    }
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
