"""Build an independent competition-data-only HGBC Kaggle experiment.

This keeps the validated HGBC hyperparameters but trains a new seed set. Each
seed receives its own honest five-fold OOF/test predictions and submission.
Every fitted fold model is saved; the final submission averages only models
trained in this experiment, never public notebook predictions.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "reference_notebooks"
    / "hgbc"
    / "ps-s6e7-hgbc-baseline-lb-0-95034-cv-0-95026.ipynb"
)
OUT_DIR = ROOT / "kaggle_kernels" / "hgbc_seedset_b_cpu"
OUT_NOTEBOOK = OUT_DIR / "hgbc_seedset_b_cpu.ipynb"
KERNEL_ID = "boltuzamaki/health-risk-hgbc-seedset-b-cpu"
EXPERIMENT_ID = "hgbc_seedset_b_63_2027_7777"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence, found {count}: {old!r}")
    return text.replace(old, new)


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))

    imports_cell = notebook["cells"][2]
    imports = "".join(imports_cell["source"])
    imports += (
        "\nimport json\n"
        "import joblib\n"
        "from pathlib import Path\n"
    )
    imports_cell["source"] = imports.splitlines(keepends=True)

    config_cell = notebook["cells"][8]
    config = "".join(config_cell["source"])
    config = replace_once(
        config,
        "FOLD_SEEDS = [42, 2026, 7]",
        "FOLD_SEEDS = [63, 2027, 7777]",
    )
    config += f'\nEXPERIMENT_ID = "{EXPERIMENT_ID}"\n'
    config_cell["source"] = config.splitlines(keepends=True)

    train_cell = notebook["cells"][10]
    code = "".join(train_cell["source"])
    code = replace_once(
        code,
        "te_names = [f'te_{c}_{k}' for c in RAW for k in range(3)]\n",
        (
            "te_names = [f'te_{c}_{k}' for c in RAW for k in range(3)]\n"
            "MODEL_DIR = Path('model_checkpoints')\n"
            "MODEL_DIR.mkdir(exist_ok=True)\n"
            "seed_scores = {}\n"
        ),
    )
    code = replace_once(
        code,
        "    oof_s = np.zeros((len(X), 3))\n",
        (
            "    oof_s = np.zeros((len(X), 3))\n"
            "    test_s = np.zeros((len(X_test), 3))\n"
        ),
    )
    code = replace_once(
        code,
        "        test_preds += model.predict_proba(A_te) / (FOLDS * len(FOLD_SEEDS))\n",
        (
            "        test_s += model.predict_proba(A_te) / FOLDS\n"
            "        joblib.dump(\n"
            "            model,\n"
            "            MODEL_DIR / f'hgbc_seed_{seed}_fold_{fold}.joblib',\n"
            "            compress=3,\n"
            "        )\n"
        ),
    )
    code = replace_once(
        code,
        "    print(f'seed {seed} | 5-fold OOF: {s:.5f}')\n"
        "    oof_preds += oof_s / len(FOLD_SEEDS)\n",
        (
            "    print(f'seed {seed} | 5-fold OOF: {s:.5f}')\n"
            "    seed_scores[str(seed)] = float(s)\n"
            "    oof_preds += oof_s / len(FOLD_SEEDS)\n"
            "    test_preds += test_s / len(FOLD_SEEDS)\n"
            "    seed_oof_df = pd.DataFrame({ID: train[ID]})\n"
            "    seed_test_df = pd.DataFrame({ID: test[ID]})\n"
            "    for i, cls in enumerate(CLASS_NAMES):\n"
            "        seed_oof_df[cls] = oof_s[:, i]\n"
            "        seed_test_df[cls] = test_s[:, i]\n"
            "    seed_oof_df.to_csv(f'oof_seed_{seed}.csv', index=False)\n"
            "    seed_test_df.to_csv(f'test_preds_seed_{seed}.csv', index=False)\n"
            "    seed_sub = pd.DataFrame({ID: test[ID], TARGET: np.argmax(test_s, axis=1)})\n"
            "    seed_sub[TARGET] = seed_sub[TARGET].map(dict(enumerate(CLASS_NAMES)))\n"
            "    seed_sub.to_csv(f'submission_seed_{seed}.csv', index=False)\n"
        ),
    )
    code = replace_once(
        code,
        "print('=' * 26)\n"
        "print(f'Seed-averaged OOF Score: \\033[1m{balanced_accuracy_score(y, oof_preds.argmax(1)):.5f}\\033[0m')",
        (
            "overall_oof = float(balanced_accuracy_score(y, oof_preds.argmax(1)))\n"
            "summary = {\n"
            "    'experiment_id': EXPERIMENT_ID,\n"
            "    'uses_only_competition_data': True,\n"
            "    'folds': FOLDS,\n"
            "    'seeds': FOLD_SEEDS,\n"
            "    'seed_scores': seed_scores,\n"
            "    'overall_oof_balanced_accuracy': overall_oof,\n"
            "}\n"
            "Path('training_summary.json').write_text(json.dumps(summary, indent=2))\n"
            "print('=' * 26)\n"
            "print(f'Seed-averaged OOF Score: \\033[1m{balanced_accuracy_score(y, oof_preds.argmax(1)):.5f}\\033[0m')"
        ),
    )
    train_cell["source"] = code.splitlines(keepends=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    metadata = {
        "id": KERNEL_ID,
        "title": "Health Risk HGBC Seedset B CPU",
        "code_file": OUT_NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["playground-series-s6e7"],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Built {OUT_NOTEBOOK}")
    print(f"Kernel id: {KERNEL_ID}")


if __name__ == "__main__":
    main()
