"""Build a private Kaggle RealMLP experiment from the audited CPU baseline.

The generated notebook:
- reads only the playground-series-s6e7 competition train/test files;
- uses seven outer stratified folds for honest OOF predictions;
- trains a new seed on GPU;
- saves OOF/test probabilities, a standalone submission, target encoders,
  feature preprocessing metadata, and one PyTorch checkpoint per fold.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "reference_notebooks"
    / "kernel_2_realmlp_cpu"
    / "kernel-2-realmlp-pytorch-implementation-cpu.ipynb"
)
OUT_DIR = ROOT / "kaggle_kernels" / "realmlp_seed2027_gpu"
OUT_NOTEBOOK = OUT_DIR / "realmlp_seed2027_gpu.ipynb"

EXPERIMENT_ID = "realmlp_seed2027_gpu_7fold_3epoch"
KERNEL_ID = "boltuzamaki/health-risk-realmlp-seed-2027-gpu"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence, found {count}: {old!r}")
    return text.replace(old, new)


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))

    setup_cell = notebook["cells"][1]
    setup = "".join(setup_cell["source"])
    setup = replace_once(setup, "seed_everything(42)", "seed_everything(2027)")
    setup += (
        "\n# Reproducible GPU execution.\n"
        "if torch.cuda.is_available():\n"
        "    torch.cuda.manual_seed_all(2027)\n"
        "    torch.backends.cudnn.deterministic = True\n"
        "    torch.backends.cudnn.benchmark = False\n"
    )
    setup_cell["source"] = setup.splitlines(keepends=True)

    config_cell = notebook["cells"][9]
    config = "".join(config_cell["source"])
    config = replace_once(config, '"epochs":    2,', '"epochs":    3,')
    config = replace_once(
        config,
        '"device":       "cpu",  # Changed from "cuda" to "cpu" to avoid CUDA compatibility issues',
        '"device":       "cuda",',
    )
    config = replace_once(config, '"random_state": 42,', '"random_state": 2027,')
    config = replace_once(config, "SEED = 42", "SEED = 2027")
    config += f'\nEXPERIMENT_ID = "{EXPERIMENT_ID}"\n'
    config_cell["source"] = config.splitlines(keepends=True)

    train_cell = notebook["cells"][11]
    train_code = "".join(train_cell["source"])
    train_code = replace_once(
        train_code,
        "test_preds = np.zeros((len(X_test), y.nunique()))\n",
        (
            "test_preds = np.zeros((len(X_test), y.nunique()))\n"
            "\n"
            "from pathlib import Path\n"
            "import json\n"
            "import joblib\n"
            "MODEL_DIR = Path('model_checkpoints')\n"
            "MODEL_DIR.mkdir(exist_ok=True)\n"
            "joblib.dump(category_map, MODEL_DIR / 'feature_category_map.joblib')\n"
            "fold_scores = []\n"
        ),
    )
    train_code = replace_once(
        train_code,
        '    print(f"\\nFold {fold} | Score: {fold_score:.5f}\\n")\n',
        (
            '    print(f"\\nFold {fold} | Score: {fold_score:.5f}\\n")\n'
            "    fold_scores.append(float(fold_score))\n"
            "    checkpoint = {\n"
            "        'experiment_id': EXPERIMENT_ID,\n"
            "        'fold': fold,\n"
            "        'seed': SEED,\n"
            "        'model_state_dict': {\n"
            "            key: value.detach().cpu()\n"
            "            for key, value in model.model_.state_dict().items()\n"
            "        },\n"
            "        'classes': model.classes_.tolist(),\n"
            "        'cat_dims': model.cat_dims_,\n"
            "        'cat_col_names': model.cat_col_names_,\n"
            "        'num_col_names': model.num_col_names_,\n"
            "        'preprocessor_tfms': model.preprocessor_._tfms,\n"
            "        'preprocessor_median': getattr(model.preprocessor_, '_median', None),\n"
            "        'preprocessor_iqr_factors': getattr(model.preprocessor_, '_iqr_factors', None),\n"
            "        'best_score': float(model.best_score_),\n"
            "    }\n"
            "    torch.save(checkpoint, MODEL_DIR / f'fold_{fold}.pt')\n"
            "    if TE:\n"
            "        joblib.dump(encoder, MODEL_DIR / f'target_encoder_fold_{fold}.joblib')\n"
        ),
    )
    train_code = replace_once(
        train_code,
        'print("="*26, "\\n")',
        (
            'print("="*26, "\\n")\n'
            "overall_oof = float(metric(y, oof_preds))\n"
            "summary = {\n"
            "    'experiment_id': EXPERIMENT_ID,\n"
            "    'seed': SEED,\n"
            "    'folds': FOLDS,\n"
            "    'epochs': CONFIG['epochs'],\n"
            "    'uses_only_competition_data': True,\n"
            "    'overall_oof_balanced_accuracy': overall_oof,\n"
            "    'fold_scores': fold_scores,\n"
            "}\n"
            "Path('training_summary.json').write_text(json.dumps(summary, indent=2))\n"
        ),
    )
    train_cell["source"] = train_code.splitlines(keepends=True)

    notebook.setdefault("metadata", {}).setdefault("kernelspec", {}).update(
        {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    metadata = {
        "id": KERNEL_ID,
        "title": "Health Risk RealMLP Seed 2027 GPU",
        "code_file": OUT_NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["playground-series-s6e7"],
        # Pin the same T4-compatible Kaggle image used by the audited
        # high-scoring RealMLP reference. Kaggle's newest PyTorch 2.10/cu128
        # image currently raises cudaErrorNoKernelImageForDevice on T4.
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
