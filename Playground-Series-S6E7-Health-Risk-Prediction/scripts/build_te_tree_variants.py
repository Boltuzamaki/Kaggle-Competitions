"""Build target-encoded ExtraTrees and CatBoost CPU experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "kaggle_kernels/hgbc_seedset_b_cpu/hgbc_seedset_b_cpu.ipynb"


def transform(kind: str, owner: str) -> Path:
    notebook = json.loads(SOURCE.read_text())
    if kind == "extra":
        slug = "health-risk-te-extratrees-cpu"
        folder = "te_extratrees_cpu"
        import_line = "from sklearn.ensemble import ExtraTreesClassifier"
        config = """CONFIG = dict(
    n_estimators=500,
    min_samples_leaf=2,
    max_features=0.80,
    bootstrap=True,
    max_samples=0.80,
    n_jobs=5,
    class_weight=None,
)"""
        constructor = "ExtraTreesClassifier(**CONFIG, random_state=seed)"
        prep_old = """for c in STR_CATS:
    X[c] = X[c].astype('category')
    X_test[c] = X_test[c].astype('category').cat.set_categories(X[c].cat.categories)"""
        prep_new = """for c in STR_CATS:
    both = pd.concat([X[c], X_test[c]], ignore_index=True).fillna('NA').astype(str)
    categories = pd.Index(both.unique())
    X[c] = pd.Categorical(X[c].fillna('NA').astype(str), categories=categories).codes
    X_test[c] = pd.Categorical(X_test[c].fillna('NA').astype(str), categories=categories).codes"""
        title = "Health Risk TE ExtraTrees CPU"
    else:
        slug = "health-risk-te-catboost-cpu"
        folder = "te_catboost_cpu"
        import_line = "from catboost import CatBoostClassifier"
        config = """CONFIG = dict(
    iterations=900,
    learning_rate=0.04,
    depth=7,
    l2_leaf_reg=5.0,
    loss_function='MultiClass',
    eval_metric='MultiClass',
    verbose=False,
    thread_count=5,
    cat_features=STR_CATS,
)"""
        constructor = "CatBoostClassifier(**CONFIG, random_seed=seed)"
        prep_old = """for c in STR_CATS:
    X[c] = X[c].astype('category')
    X_test[c] = X_test[c].astype('category').cat.set_categories(X[c].cat.categories)"""
        prep_new = """for c in STR_CATS:
    X[c] = X[c].fillna('NA').astype(str)
    X_test[c] = X_test[c].fillna('NA').astype(str)"""
        title = "Health Risk TE CatBoost CPU"

    for cell in notebook["cells"]:
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else str(source)
        text = text.replace(
            "from sklearn.ensemble import HistGradientBoostingClassifier",
            import_line,
        )
        text = re.sub(r"CONFIG = dict\(.*?\n\)", config, text, flags=re.DOTALL)
        text = text.replace("FOLD_SEEDS = [63, 2027, 7777]", "FOLD_SEEDS = [2027]")
        text = text.replace(
            'EXPERIMENT_ID = "hgbc_seedset_b_63_2027_7777"',
            f'EXPERIMENT_ID = "te_{kind}_cpu_seed2027"',
        )
        text = text.replace(prep_old, prep_new)
        text = text.replace(
            "HistGradientBoostingClassifier(**{**CONFIG, 'random_state': seed})",
            constructor,
        )
        text = re.sub(
            r"\n        joblib\.dump\(\n.*?\n        \)",
            "",
            text,
            flags=re.DOTALL,
        )
        text = text.replace("tehgbc_oof_preds.csv", "oof_preds.csv")
        text = text.replace("tehgbc_test_preds.csv", "test_preds.csv")
        cell["source"] = text.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None

    destination = ROOT / "kaggle_kernels" / folder
    destination.mkdir(parents=True, exist_ok=True)
    notebook_name = f"{folder}.ipynb"
    (destination / notebook_name).write_text(json.dumps(notebook, indent=1) + "\n")
    metadata = {
        "id": f"{owner}/{slug}",
        "title": title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["playground-series-s6e7"],
    }
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return destination


def main() -> None:
    print(transform("extra", "boltuzamaki"))
    print(transform("catboost", "divyanshuboltuzamaki"))


if __name__ == "__main__":
    main()
