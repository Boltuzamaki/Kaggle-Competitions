"""Build a competition-only target-encoded XGBoost CPU experiment."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle_kernels/hgbc_seedset_b_cpu/hgbc_seedset_b_cpu.ipynb"
DEST = ROOT / "kaggle_kernels/te_xgb_cpu"


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    for cell in notebook["cells"]:
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else str(source)
        text = text.replace(
            "from sklearn.ensemble import HistGradientBoostingClassifier",
            "from xgboost import XGBClassifier",
        )
        text = re.sub(
            r"CONFIG = dict\(.*?\n\)",
            """CONFIG = dict(
    objective='multi:softprob',
    eval_metric='mlogloss',
    tree_method='hist',
    n_estimators=1100,
    learning_rate=0.025,
    max_depth=7,
    min_child_weight=25,
    gamma=3.7047462188482423,
    subsample=0.7550078804514493,
    colsample_bytree=0.789613735687402,
    reg_alpha=1.3031542131096254,
    reg_lambda=0.007621004297428309,
    max_delta_step=3,
    n_jobs=5,
    enable_categorical=True,
)""",
            text,
            flags=re.DOTALL,
        )
        text = text.replace(
            "FOLD_SEEDS = [63, 2027, 7777]",
            "FOLD_SEEDS = [2027]",
        )
        text = text.replace(
            'EXPERIMENT_ID = "hgbc_seedset_b_63_2027_7777"',
            'EXPERIMENT_ID = "te_xgb_cpu_seed2027"',
        )
        text = text.replace(
            "HistGradientBoostingClassifier(**{**CONFIG, 'random_state': seed})",
            "XGBClassifier(**CONFIG, random_state=seed)",
        )
        text = re.sub(
            r"\n        joblib\.dump\(\n.*?\n        \)",
            "",
            text,
            flags=re.DOTALL,
        )
        text = text.replace("hgbc_seed_", "te_xgb_seed_")
        text = text.replace("tehgbc_oof_preds.csv", "oof_preds.csv")
        text = text.replace("tehgbc_test_preds.csv", "test_preds.csv")
        cell["source"] = text.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "te_xgb_cpu.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
    metadata = {
        "id": "boltuzamaki/health-risk-te-xgb-cpu",
        "title": "Health Risk TE XGB CPU",
        "code_file": "te_xgb_cpu.ipynb",
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
