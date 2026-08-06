"""Raw data loading. Every other module (features, folds, train, ensemble)
goes through here so there is exactly one place that knows about file paths,
dtypes, and the fast_dev_run subsampling switch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = cfg["data"]
    train = pd.read_csv(f"{d['dir']}/{d['train_file']}")
    test = pd.read_csv(f"{d['dir']}/{d['test_file']}")

    if cfg["run"].get("fast_dev_run"):
        frac = cfg["run"].get("fast_dev_frac", 0.03)
        rng = np.random.RandomState(cfg["seed"])
        train = train.sample(frac=frac, random_state=rng).reset_index(drop=True)
        test = test.sample(frac=frac, random_state=rng).reset_index(drop=True)

    return train, test


def load_sample_submission(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    return pd.read_csv(f"{d['dir']}/{d['sample_submission_file']}")


def encode_target(train: pd.DataFrame, cfg: dict) -> np.ndarray:
    classes = cfg["data"]["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = train[cfg["data"]["target_col"]].map(class_to_idx)
    if y.isnull().any():
        bad = train.loc[y.isnull(), cfg["data"]["target_col"]].unique()
        raise ValueError(f"Unexpected target labels not in config classes: {bad}")
    return y.to_numpy(dtype=np.int64)


def decode_target(y_idx: np.ndarray, cfg: dict) -> np.ndarray:
    classes = np.array(cfg["data"]["classes"])
    return classes[y_idx]
