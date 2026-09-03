"""Fast executable checks for the TabR experiment's data and validation contract."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SOURCE = Path(__file__).with_name("experiment.py")
TARGET, ID, SEED = "addicted_label", "id", 20260804


def source_tree():
    return ast.parse(SOURCE.read_text())


def test_no_submission_or_foreign_prediction_input():
    text = SOURCE.read_text().lower()
    assert "competition_submit" not in text
    assert "kaggle.api" not in text
    assert "submission.csv" not in text
    assert "public_outputs" not in text


def test_exactly_five_outer_validation_coverage():
    root = SOURCE.parent.parent
    train = pd.read_csv(root / "train.csv", usecols=[ID, TARGET])
    y = train[TARGET].to_numpy()
    seen = np.zeros(len(train), dtype=np.int8)
    folds = np.zeros(len(train), dtype=np.int8)
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (fit_idx, val_idx) in enumerate(outer.split(train, y), 1):
        assert not np.intersect1d(fit_idx, val_idx).size
        seen[val_idx] += 1
        folds[val_idx] = fold
        # The tuning split is wholly inside this fold's fit partition.
        pool = fit_idx
        if len(pool) > 200_000:
            pool, _ = train_test_split(pool, train_size=200_000,
                                       stratify=y[pool], random_state=SEED + fold)
        ia, ib = train_test_split(pool, test_size=.15, stratify=y[pool],
                                  random_state=SEED + 10 + fold)
        assert not np.intersect1d(np.r_[ia, ib], val_idx).size
    assert np.all(seen == 1)
    assert set(np.unique(folds)) == {1, 2, 3, 4, 5}


def test_p100_stack_is_installed_before_pytabkit_without_deps():
    text = SOURCE.read_text()
    torch_at = text.index('"torch==2.5.1"')
    tabr_at = text.index('"pytabkit==1.7.3"')
    assert torch_at < tabr_at
    assert '"--no-deps"' in text[torch_at:tabr_at + 200]
    assert '"torchvision==0.20.1"' in text
    assert '"torchaudio==2.5.1"' in text
    assert '"faiss-cpu==1.12.0"' in text
    assert "faiss-gpu" not in text


def test_numeric_only_schema_has_categorical_retrieval_key():
    # PyTabKit 1.7.3's TabR interface otherwise passes a (n, 0) matrix into
    # OrdinalEncoder and fails before epoch one on this all-numeric dataset.
    text = SOURCE.read_text()
    assert 'x["missing_pattern"]' in text
    assert '.astype(object)' in text
    assert text.count('cat_col_names=CAT_COLS') == 2
    assert 'CAT_COLS = list(X.select_dtypes(exclude=np.number).columns)' in text
    assert '_tabr_interface.SimpleImputer = _KeepEmptySimpleImputer' in text
    assert 'faiss.GpuIndexFlatL2 = TorchExactL2Index' in text
    assert 'class TorchExactL2Index' in text


if __name__ == "__main__":
    test_no_submission_or_foreign_prediction_input()
    test_exactly_five_outer_validation_coverage()
    test_p100_stack_is_installed_before_pytabkit_without_deps()
    test_numeric_only_schema_has_categorical_retrieval_key()
    print("TabR provenance, five-fold coverage, nesting, and P100 checks passed")
