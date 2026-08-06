"""Fixed CV fold assignment, generated once and reused by every model in the
zoo. This is what makes OOF predictions from 300+ independently-trained
models directly comparable and blend-able: they were all evaluated on the
exact same row->fold mapping.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from src.data import encode_target, load_raw


def get_or_create_folds(cfg: dict) -> np.ndarray:
    """Returns an int array of shape (n_train,) with fold id per row.
    Cached to disk at cfg.paths.folds_file so every model training run
    (across process restarts) uses the identical split.
    """
    path = cfg["paths"]["folds_file"]
    if os.path.exists(path):
        return np.load(path)

    train, _ = load_raw(cfg)
    y = encode_target(train, cfg)

    n_folds = cfg["cv"]["n_folds"]
    strategy = cfg["cv"].get("strategy", "stratified")
    seed = cfg["seed"]

    if strategy == "stratified":
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(y)), y)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(y)))

    folds = np.full(len(y), -1, dtype=np.int8)
    for fold_id, (_, val_idx) in enumerate(split_iter):
        folds[val_idx] = fold_id

    assert (folds >= 0).all(), "every row must be assigned to a fold"
    np.save(path, folds)
    return folds


if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config()
    folds = get_or_create_folds(cfg)
    print("folds shape:", folds.shape)
    print("fold sizes:", np.bincount(folds))
