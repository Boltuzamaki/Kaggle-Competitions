"""Build a class-balanced FT-Transformer experiment for Kaggle GPU.

Unlike the public reference, this experiment optimizes class-balanced loss
directly and ships raw argmax predictions. It does not tune a public-output
blend or use external data. Every fold's native masamlp model is saved.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "reference_notebooks"
    / "fttransformer"
    / "s6e7-ft-transformer-v2-cv-0-95063.ipynb"
)
OUT_DIR = ROOT / "kaggle_kernels" / "ftt_balanced_seed2027_gpu"
OUT_NOTEBOOK = OUT_DIR / "ftt_balanced_seed2027_gpu.ipynb"
KERNEL_ID = "boltuzamaki/health-risk-ftt-balanced-seed-2027-gpu"
EXPERIMENT_ID = "ftt_balanced_seed2027_7fold"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence, found {count}: {old!r}")
    return text.replace(old, new)


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))

    imports_cell = notebook["cells"][5]
    imports = "".join(imports_cell["source"])
    imports += "\nimport json\nfrom pathlib import Path\n"
    imports_cell["source"] = imports.splitlines(keepends=True)

    config_cell = notebook["cells"][7]
    config = "".join(config_cell["source"])
    config = replace_once(config, "SEED = 42", "SEED = 2027")
    config = replace_once(
        config,
        "BETA_GRID = [0.0, 0.5, 0.75, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5]",
        "BETA_GRID = [0.0]  # no post-hoc threshold tuning; raw argmax only",
    )
    config = replace_once(config, "class_weight=None,", "class_weight='balanced',")
    config += (
        f'\nEXPERIMENT_ID = "{EXPERIMENT_ID}"\n'
        "MODEL_ROOT = Path('model_checkpoints')\n"
        "MODEL_ROOT.mkdir(exist_ok=True)\n"
    )
    config_cell["source"] = config.splitlines(keepends=True)

    cv_cell = notebook["cells"][19]
    cv_code = "".join(cv_cell["source"])
    cv_code = replace_once(
        cv_code,
        "    test_proba = np.zeros((len(X_test), len(classes)))\n",
        (
            "    test_proba = np.zeros((len(X_test), len(classes)))\n"
            "    fold_scores = []\n"
        ),
    )
    cv_code = replace_once(
        cv_code,
        "        model.fit(Xtr, y.iloc[tr_idx], eval_set=[(Xva, y.iloc[va])])\n",
        (
            "        model.fit(Xtr, y.iloc[tr_idx], eval_set=[(Xva, y.iloc[va])])\n"
            "        model.save_model(str(MODEL_ROOT / f'fold_{fold}'))\n"
        ),
    )
    cv_code = replace_once(
        cv_code,
        '        print(f"  fold {fold}: raw balanced accuracy = {fold_ba:.5f}")\n',
        (
            '        print(f"  fold {fold}: raw balanced accuracy = {fold_ba:.5f}")\n'
            "        fold_scores.append(float(fold_ba))\n"
        ),
    )
    cv_code = replace_once(
        cv_code,
        "    return oof, test_proba, model\n",
        "    return oof, test_proba, model, fold_scores\n",
    )
    cv_code = replace_once(
        cv_code,
        "oof_proba, test_proba, model = cross_validate(X, y, X_test)",
        "oof_proba, test_proba, model, fold_scores = cross_validate(X, y, X_test)",
    )
    cv_cell["source"] = cv_code.splitlines(keepends=True)

    output_cell = notebook["cells"][23]
    output_code = "".join(output_cell["source"])
    output_code = replace_once(
        output_code,
        'print("wrote: submission.csv, oof_ftt.csv, testpred_ftt.csv")\n',
        (
            'print("wrote: submission.csv, oof_ftt.csv, testpred_ftt.csv")\n'
            "summary = {\n"
            "    'experiment_id': EXPERIMENT_ID,\n"
            "    'uses_only_competition_data': True,\n"
            "    'seed': SEED,\n"
            "    'folds': N_SPLITS,\n"
            "    'class_weight': 'balanced',\n"
            "    'decision_rule': 'raw_argmax',\n"
            "    'raw_oof_balanced_accuracy': float(raw_ba),\n"
            "    'fold_scores': fold_scores,\n"
            "}\n"
            "Path('training_summary.json').write_text(json.dumps(summary, indent=2))\n"
        ),
    )
    output_cell["source"] = output_code.splitlines(keepends=True)

    bag_cell = notebook["cells"][25]
    bag_code = "".join(bag_cell["source"])
    bag_code = replace_once(
        bag_code,
        "    o, t, _ = cross_validate(X, y, X_test, seed=s)",
        "    o, t, _, _ = cross_validate(X, y, X_test, seed=s)",
    )
    bag_cell["source"] = bag_code.splitlines(keepends=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NOTEBOOK.write_text(
        # Kaggle CLI opens the code file with the Windows locale encoding.
        # Escaping non-ASCII keeps the notebook portable through that client.
        json.dumps(notebook, ensure_ascii=True, indent=1),
        encoding="utf-8",
    )
    metadata = {
        "id": KERNEL_ID,
        "title": "Health Risk FTT Balanced Seed 2027 GPU",
        "code_file": OUT_NOTEBOOK.name,
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
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Built {OUT_NOTEBOOK}")
    print(f"Kernel id: {KERNEL_ID}")


if __name__ == "__main__":
    main()
