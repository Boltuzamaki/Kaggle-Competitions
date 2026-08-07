"""Schema discovery and submission helpers for the robust tabular skill."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BAD_TOKENS = ("solution", "answer", "truth", "ground", "labelled_test")


def runtime_workdir() -> Path:
    """Return the harness data directory, with a current-directory fallback for offline tests."""
    override = os.environ.get("ROBUST_TABULAR_WORKDIR")
    if override:
        candidate = Path(override).expanduser().resolve()
        if candidate.is_dir():
            return candidate
    harness = Path("/work")
    if harness.is_dir():
        names = {path.name.lower() for path in harness.glob("*.csv")}
        if any("train" in name for name in names) and any("test" in name for name in names):
            return harness
    return Path.cwd()


def _safe_csvs(task_dir: str | os.PathLike[str]) -> list[str]:
    paths = glob.glob(os.path.join(str(task_dir), "**", "*.csv"), recursive=True)
    return [
        path
        for path in paths
        if not any(token in os.path.basename(path).lower() for token in BAD_TOKENS)
    ]


def discover_files(task_dir: str | os.PathLike[str] = ".") -> tuple[str, str, str | None]:
    csvs = _safe_csvs(task_dir)

    def pick(required: tuple[str, ...], forbidden: tuple[str, ...] = ()) -> str | None:
        for path in sorted(csvs, key=lambda value: (len(Path(value).parts), len(value))):
            name = os.path.basename(path).lower()
            if any(token in name for token in required) and not any(token in name for token in forbidden):
                return path
        return None

    train_path = pick(("train",), ("sample", "submission"))
    test_path = pick(("test",), ("sample", "submission"))
    sample_path = pick(("sample_submission", "samplesubmission", "sample-submission"))
    if train_path is None or test_path is None:
        raise FileNotFoundError(f"train/test CSV not found; visible CSV count={len(csvs)}")
    return train_path, test_path, sample_path


def infer_id(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame | None) -> str | None:
    if sample is not None and len(sample.columns) and sample.columns[0] in test.columns:
        return str(sample.columns[0])
    preferred = ("row_id", "id", "sample_id", "record_id", "uid", "index")
    for column in test.columns:
        low = str(column).lower()
        if low in preferred or low.endswith("_id"):
            return str(column)
    for column in test.columns:
        if column in train.columns and train[column].is_unique and test[column].is_unique:
            return str(column)
    return None


def infer_target(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame | None, id_col: str | None) -> str:
    train_only = [column for column in train.columns if column not in test.columns and column != id_col]
    preferred = ("target", "label", "class", "outcome", "response", "y")
    for column in train_only:
        if str(column).lower() in preferred:
            return str(column)
    if len(train_only) == 1:
        return str(train_only[0])
    for column in train_only:
        if train[column].nunique(dropna=True) == 2:
            return str(column)
    if sample is not None:
        for column in sample.columns[1:]:
            if column in train.columns:
                return str(column)
    raise ValueError(f"binary target could not be inferred from train-only columns={train_only}")


def normalize_binary(y: pd.Series) -> np.ndarray:
    values = list(pd.unique(y.dropna()))
    if len(values) != 2:
        raise ValueError(f"expected binary target, found {len(values)} values")
    if pd.api.types.is_numeric_dtype(y) and set(values).issubset({0, 1, 0.0, 1.0}):
        return y.astype(int).to_numpy()
    positive = sorted(values, key=lambda value: str(value))[-1]
    return (y == positive).astype(int).to_numpy()


def load_task(task_dir: str | os.PathLike[str] = "."):
    train_path, test_path, sample_path = discover_files(task_dir)
    train = pd.read_csv(train_path, engine="pyarrow")
    test = pd.read_csv(test_path, engine="pyarrow")
    sample = pd.read_csv(sample_path, engine="pyarrow") if sample_path else None
    id_col = infer_id(train, test, sample)
    target_col = infer_target(train, test, sample, id_col)
    features = [column for column in test.columns if column in train.columns and column != id_col]
    if not features:
        raise ValueError("no shared modeling features")
    y = normalize_binary(train[target_col])
    return train, test, sample, id_col, target_col, features, y


def categorical_columns(frame: pd.DataFrame, features: list[str]) -> list[str]:
    """Anything that is not numeric or datetime is categorical.

    Stated this way round the detection survives the container's pandas version:
    under pandas 3 string columns arrive as StringDtype rather than object, and an
    is_object_dtype test silently returns zero categoricals and hands raw strings
    to the numeric path.
    """
    selected = []
    for column in features:
        dtype = frame[column].dtype
        if pd.api.types.is_bool_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            selected.append(column)
        elif pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_datetime64_any_dtype(dtype):
            continue
        else:
            selected.append(column)
    return selected


def native_frames(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    x_train = train[features].copy()
    x_test = test[features].copy()
    cat_cols = categorical_columns(x_train, features)
    for column in cat_cols:
        x_train[column] = x_train[column].fillna("__MISSING__").astype(str)
        x_test[column] = x_test[column].fillna("__MISSING__").astype(str)
    for column in (item for item in features if item not in cat_cols):
        x_train[column] = pd.to_numeric(x_train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        x_test[column] = pd.to_numeric(x_test[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x_train, x_test, cat_cols


def encoded_frames(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    x_train = train[features].copy()
    x_test = test[features].copy()
    cat_cols = categorical_columns(x_train, features)
    for column in features:
        if column in cat_cols:
            combined = pd.concat([x_train[column], x_test[column]], ignore_index=True).fillna("__MISSING__").astype(str)
            codes, _ = pd.factorize(combined, sort=True)
            x_train[column] = codes[: len(x_train)]
            x_test[column] = codes[len(x_train) :]
        else:
            tr = pd.to_numeric(x_train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            te = pd.to_numeric(x_test[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            median = float(tr.median()) if tr.notna().any() else 0.0
            x_train[column] = tr.fillna(median)
            x_test[column] = te.fillna(median)
    return x_train.astype(float), x_test.astype(float), cat_cols


def write_submission(
    path: str | os.PathLike[str],
    predictions: np.ndarray,
    test: pd.DataFrame,
    sample: pd.DataFrame | None,
    id_col: str | None,
    target_col: str,
) -> str:
    predictions = np.asarray(predictions, dtype=float)
    predictions = np.nan_to_num(predictions, nan=0.5, posinf=1.0, neginf=0.0)
    predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
    if len(predictions) != len(test):
        raise ValueError(f"prediction rows={len(predictions)} test rows={len(test)}")
    if sample is not None:
        output = sample.copy()
        if len(output) != len(test):
            raise ValueError("sample-submission row count does not match test")
        prediction_cols = [column for column in output.columns if column != id_col]
        if not prediction_cols:
            prediction_cols = [output.columns[-1]]
        output[prediction_cols[-1]] = predictions
    else:
        ids = test[id_col] if id_col and id_col in test.columns else np.arange(len(test))
        output = pd.DataFrame({id_col or "row_id": ids, target_col: predictions})
    if not np.isfinite(output.iloc[:, -1].to_numpy(dtype=float)).all():
        raise ValueError("non-finite predictions after sanitization")
    output.to_csv(path, index=False)
    return str(path)


def rank_unit(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average", pct=True).to_numpy()


def emit_manifest(payload: dict) -> None:
    print("PORTFOLIO_MANIFEST=" + json.dumps(payload, sort_keys=True, separators=(",", ":")))