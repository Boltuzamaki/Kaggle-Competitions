"""Create a regularized GPU target-encoded XGBoost variant."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle_kernels/te_xgb_cpu/te_xgb_cpu.ipynb"
DEST = ROOT / "kaggle_kernels/te_xgb_gpu_regularized"


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    replacements = {
        "n_estimators=1100": "n_estimators=1500",
        "learning_rate=0.025": "learning_rate=0.018",
        "max_depth=7": "max_depth=6",
        "min_child_weight=25": "min_child_weight=40",
        "gamma=3.7047462188482423": "gamma=2.5",
        "subsample=0.7550078804514493": "subsample=0.84",
        "colsample_bytree=0.789613735687402": "colsample_bytree=0.84",
        "reg_alpha=1.3031542131096254": "reg_alpha=0.8",
        "reg_lambda=0.007621004297428309": "reg_lambda=0.10",
        "n_jobs=5,": "n_jobs=2,\n    device='cuda',",
        'EXPERIMENT_ID = "te_xgb_cpu_seed2027"':
            'EXPERIMENT_ID = "te_xgb_gpu_regularized_seed2027"',
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
    (DEST / "te_xgb_gpu_regularized.ipynb").write_text(
        json.dumps(notebook, indent=1) + "\n"
    )
    metadata = {
        "id": "divyanshuboltuzamaki/health-risk-te-xgb-regularized-gpu",
        "title": "Health Risk TE XGB Regularized GPU",
        "code_file": "te_xgb_gpu_regularized.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["playground-series-s6e7"],
        "machine_shape": "NvidiaTeslaT4",
    }
    (DEST / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
