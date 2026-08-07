#!/usr/bin/env python3
"""Deterministic CPU AutoML for the Autonomous Agent Prediction competition.

The script deliberately has no LLM-facing configuration surface. It infers the schema,
creates leakage-safe out-of-fold predictions, trains diverse models, greedily blends them,
and emits a small set of leaderboard candidates in the sample-submission schema.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

SEED = 20260717


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (rankdata(values, method="average") - 0.5) / len(values)


def _safe_auc(y: np.ndarray, pred: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, pred))
    except Exception:
        return 0.5


def _infer_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer conservative categorical columns from encoded CSV values."""
    categorical: list[str] = []
    numeric: list[str] = []
    n = max(len(frame), 1)
    cardinality_limit = min(80, max(12, int(np.sqrt(n))))
    for col in frame.columns:
        series = frame[col]
        nonnull = series.dropna()
        nunique = int(nonnull.nunique())
        is_integerish = False
        if len(nonnull):
            vals = pd.to_numeric(nonnull, errors="coerce").to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            is_integerish = len(finite) == len(vals) and bool(
                np.all(np.abs(finite - np.round(finite)) < 1e-8)
            )
        if not pd.api.types.is_numeric_dtype(series):
            categorical.append(col)
        elif is_integerish and 2 <= nunique <= cardinality_limit and nunique / n <= 0.08:
            categorical.append(col)
        else:
            numeric.append(col)
    return categorical, numeric


def _tree_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    blocks = []
    if numeric:
        blocks.append(("num", SimpleImputer(strategy="median", add_indicator=True), numeric))
    if categorical:
        blocks.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2
            )),
        ]), categorical))
    return ColumnTransformer(blocks, remainder="drop")


def _prepare_catboost(
    train: pd.DataFrame, test: pd.DataFrame, categorical: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    train_cb = train.copy()
    test_cb = test.copy()
    for col in categorical:
        train_cb[col] = train_cb[col].fillna("__MISSING__").astype(str)
        test_cb[col] = test_cb[col].fillna("__MISSING__").astype(str)
    return train_cb, test_cb, [train_cb.columns.get_loc(c) for c in categorical]


def _fit_predict_candidates(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    quick: bool,
    regime: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    """Return OOF/test predictions; individual failures do not sink the run."""
    n = len(X)
    if quick:
        folds = 3
        rounds = 260
    else:
        folds = 7 if n < 1500 else (5 if n < 3000 else (4 if n < 20000 else 3))
        rounds = 650 if n < 30000 else 500
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}

    def register(name: str, fold_oof: np.ndarray, fold_test: list[np.ndarray]) -> None:
        if np.isfinite(fold_oof).all() and fold_test:
            oof[name] = fold_oof
            test_preds[name] = np.mean(np.vstack(fold_test), axis=0)

    # Regime-specialized CatBoost allocation: seeds for categorical, shallow bagging for small.
    try:
        from catboost import CatBoostClassifier

        X_cb, T_cb, cat_idx = _prepare_catboost(X, X_test, categorical)
        if regime == "categorical":
            cb_configs = {
                "cat_native_seed1": (7, 0.045, 7.0, SEED),
                "cat_native_seed2": (7, 0.045, 7.0, SEED + 7919),
                "cat_native_shallow": (5, 0.065, 11.0, SEED + 1543),
            }
        elif regime == "small":
            cb_configs = {
                "small_cat_shallow": (5, 0.045, 12.0, SEED),
                "small_cat_seed2": (6, 0.035, 15.0, SEED + 7919),
            }
        else:
            cb_configs = {"numeric_catboost": (7, 0.045, 6.0, SEED)}
        for config_name, (depth, lr, l2, seed_base) in cb_configs.items():
            pred_oof = np.zeros(n)
            pred_test: list[np.ndarray] = []
            for fold, (tr, va) in enumerate(cv.split(X_cb, y)):
                model = CatBoostClassifier(
                    iterations=rounds,
                    depth=depth,
                    learning_rate=lr if not quick else max(lr, 0.065),
                    loss_function="Logloss",
                    eval_metric="AUC",
                    l2_leaf_reg=l2,
                    random_strength=0.6 if regime != "numeric" else 0.4,
                    random_seed=seed_base + fold,
                    thread_count=4,
                    verbose=False,
                    allow_writing_files=False,
                )
                model.fit(
                    X_cb.iloc[tr], y[tr],
                    cat_features=cat_idx,
                    eval_set=(X_cb.iloc[va], y[va]),
                    early_stopping_rounds=100,
                    verbose=False,
                )
                pred_oof[va] = model.predict_proba(X_cb.iloc[va])[:, 1]
                pred_test.append(model.predict_proba(T_cb)[:, 1])
            register(config_name, pred_oof, pred_test)
    except Exception as exc:
        errors["catboost"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    # LightGBM gets extra capacity only in the numeric regime.
    try:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        leaves = [15, 31, 63] if regime == "numeric" else ([15] if regime == "small" else [31])
        for num_leaves in leaves:
            config_name = f"lgb_regime_{num_leaves}"
            pred_oof = np.zeros(n)
            pred_test = []
            for fold, (tr, va) in enumerate(cv.split(X, y)):
                prep = _tree_preprocessor(numeric, categorical)
                X_tr = prep.fit_transform(X.iloc[tr])
                X_va = prep.transform(X.iloc[va])
                T = prep.transform(X_test)
                model = LGBMClassifier(
                    n_estimators=rounds + 250,
                    learning_rate=0.035 if not quick else 0.06,
                    num_leaves=num_leaves,
                    max_depth=-1,
                    min_child_samples=max(20, min(180, n // (55 if regime == "small" else 90))),
                    subsample=0.85,
                    colsample_bytree=0.8,
                    reg_alpha=0.8 if regime == "small" else 0.35,
                    reg_lambda=5.0 if regime == "small" else 2.5,
                    objective="binary",
                    random_state=SEED + fold,
                    n_jobs=4,
                    verbosity=-1,
                )
                model.fit(
                    X_tr, y[tr], eval_set=[(X_va, y[va])], eval_metric="auc",
                    callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
                )
                pred_oof[va] = model.predict_proba(X_va)[:, 1]
                pred_test.append(model.predict_proba(T)[:, 1])
            register(config_name, pred_oof, pred_test)
    except Exception as exc:
        errors["lightgbm"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    # XGBoost provides a second histogram-boosting inductive bias.
    try:
        if regime == "categorical":
            raise RuntimeError("intentionally skipped for categorical specialist regime")
        from xgboost import XGBClassifier

        pred_oof = np.zeros(n)
        pred_test = []
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            prep = _tree_preprocessor(numeric, categorical)
            X_tr = prep.fit_transform(X.iloc[tr])
            X_va = prep.transform(X.iloc[va])
            T = prep.transform(X_test)
            model = XGBClassifier(
                n_estimators=rounds,
                max_depth=5 if n >= 1500 else 4,
                learning_rate=0.04 if not quick else 0.07,
                min_child_weight=max(2, min(12, n // 1200)),
                subsample=0.85,
                colsample_bytree=0.8,
                reg_alpha=0.2,
                reg_lambda=3.0,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                random_state=SEED + fold,
                n_jobs=4,
            )
            model.fit(X_tr, y[tr], eval_set=[(X_va, y[va])], verbose=False)
            pred_oof[va] = model.predict_proba(X_va)[:, 1]
            pred_test.append(model.predict_proba(T)[:, 1])
        register("xgboost", pred_oof, pred_test)
    except Exception as exc:
        errors["xgboost"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    # Histogram gradient boosting is a stable sklearn-only fallback and useful blend member.
    try:
        if regime == "categorical":
            raise RuntimeError("intentionally skipped for categorical specialist regime")
        pred_oof = np.zeros(n)
        pred_test = []
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            prep = _tree_preprocessor(numeric, categorical)
            X_tr = prep.fit_transform(X.iloc[tr])
            X_va = prep.transform(X.iloc[va])
            T = prep.transform(X_test)
            model = HistGradientBoostingClassifier(
                max_iter=260 if quick else 420,
                learning_rate=0.065,
                max_leaf_nodes=31 if n >= 1500 else 15,
                min_samples_leaf=max(12, min(80, n // 150)),
                l2_regularization=2.0,
                early_stopping=True,
                validation_fraction=0.12,
                random_state=SEED + fold,
            )
            model.fit(X_tr, y[tr])
            pred_oof[va] = model.predict_proba(X_va)[:, 1]
            pred_test.append(model.predict_proba(T)[:, 1])
        register("histgb", pred_oof, pred_test)
    except Exception as exc:
        errors["histgb"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    # ExtraTrees is deliberately less correlated with the boosting family.
    try:
        pred_oof = np.zeros(n)
        pred_test = []
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            prep = _tree_preprocessor(numeric, categorical)
            X_tr = prep.fit_transform(X.iloc[tr])
            X_va = prep.transform(X.iloc[va])
            T = prep.transform(X_test)
            model = ExtraTreesClassifier(
                n_estimators=350 if quick else 650,
                max_features=0.8,
                min_samples_leaf=max(2, min(12, n // 2500)),
                class_weight="balanced",
                random_state=SEED + fold,
                n_jobs=4,
            )
            model.fit(X_tr, y[tr])
            pred_oof[va] = model.predict_proba(X_va)[:, 1]
            pred_test.append(model.predict_proba(T)[:, 1])
        register("extratrees", pred_oof, pred_test)
    except Exception as exc:
        errors["extratrees"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    # Small-sample specialist: bagged trees reduce fold and seed variance.
    if regime == "small":
        try:
            pred_oof = np.zeros(n)
            pred_test = []
            for fold, (tr, va) in enumerate(cv.split(X, y)):
                prep = _tree_preprocessor(numeric, categorical)
                X_tr = prep.fit_transform(X.iloc[tr])
                X_va = prep.transform(X.iloc[va])
                T = prep.transform(X_test)
                model = RandomForestClassifier(
                    n_estimators=450 if quick else 800,
                    max_features="sqrt",
                    min_samples_leaf=max(3, n // 250),
                    class_weight="balanced_subsample",
                    random_state=SEED + fold,
                    n_jobs=4,
                )
                model.fit(X_tr, y[tr])
                pred_oof[va] = model.predict_proba(X_va)[:, 1]
                pred_test.append(model.predict_proba(T)[:, 1])
            register("small_randomforest", pred_oof, pred_test)
        except Exception as exc:
            errors["small_randomforest"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    # Regularized linear/OHE model catches additive and high-cardinality effects cheaply.
    try:
        transformers = []
        if numeric:
            transformers.append(("num", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), numeric))
        if categorical:
            transformers.append(("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
            ]), categorical))
        prep = ColumnTransformer(transformers, remainder="drop")
        template = Pipeline([
            ("prep", prep),
            ("model", LogisticRegression(C=0.35, max_iter=1200, class_weight="balanced")),
        ])
        pred_oof = np.zeros(n)
        pred_test = []
        for tr, va in cv.split(X, y):
            model = clone(template)
            model.fit(X.iloc[tr], y[tr])
            pred_oof[va] = model.predict_proba(X.iloc[va])[:, 1]
            pred_test.append(model.predict_proba(X_test)[:, 1])
        register("linear", pred_oof, pred_test)
    except Exception as exc:
        errors["linear"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    return oof, test_preds, errors


def _build_blends(
    y: np.ndarray, oof: dict[str, np.ndarray], test_preds: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, float], str]:
    scores = {name: _safe_auc(y, pred) for name, pred in oof.items()}
    ranked_oof = {name: _rank01(pred) for name, pred in oof.items()}
    ranked_test = {name: _rank01(pred) for name, pred in test_preds.items()}
    candidates = dict(test_preds)

    ordered = sorted(scores, key=scores.get, reverse=True)
    if not ordered:
        raise RuntimeError("Every model failed; no candidate predictions were created")

    # Greedy equal-weight OOF blend: add a model only when it improves validation AUC.
    selected = [ordered[0]]
    blend_oof = ranked_oof[ordered[0]].copy()
    best = _safe_auc(y, blend_oof)
    remaining = ordered[1:]
    while remaining and len(selected) < 4:
        trials = []
        for name in remaining:
            trial = (blend_oof * len(selected) + ranked_oof[name]) / (len(selected) + 1)
            trials.append((_safe_auc(y, trial), name, trial))
        score, name, trial = max(trials, key=lambda item: item[0])
        if score <= best + 0.00015:
            break
        selected.append(name)
        remaining.remove(name)
        blend_oof, best = trial, score

    greedy_test = np.mean(np.vstack([ranked_test[n] for n in selected]), axis=0)
    candidates["blend_greedy"] = greedy_test
    scores["blend_greedy"] = best

    # Conservative diversity blend guards against noisy CV on the 500-row tasks.
    top = ordered[: min(4, len(ordered))]
    conservative_oof = np.mean(np.vstack([ranked_oof[n] for n in top]), axis=0)
    candidates["blend_diverse"] = np.mean(np.vstack([ranked_test[n] for n in top]), axis=0)
    scores["blend_diverse"] = _safe_auc(y, conservative_oof)

    robust_name = max(("blend_greedy", "blend_diverse"), key=scores.get)
    return candidates, scores, robust_name


def _resolve_runtime_data_dir(requested: Path) -> tuple[Path, bool]:
    """Find persistent sandbox data when ADK runs the skill from a temp directory."""
    required = ("train.csv", "test.csv", "sample_submission.csv")
    requested = requested.resolve()
    if all((requested / name).is_file() for name in required):
        return requested, False

    candidates = []
    if os.environ.get("PWD"):
        candidates.append(Path(os.environ["PWD"]))
    candidates.extend([
        Path("/kaggle/working"),
        Path("/workspace"),
        Path("/app"),
        Path("/root"),
    ])
    seen = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if all((candidate / name).is_file() for name in required):
            return candidate, True
    raise FileNotFoundError(
        "Could not locate persistent train.csv, test.csv, and sample_submission.csv; "
        f"temporary cwd={Path.cwd()} PWD={os.environ.get('PWD')}"
    )


def run(
    data_dir: Path,
    output_dir: Path | None,
    quick: bool = False,
    relative_output_paths: bool = False,
) -> dict:
    start = time.time()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if len(sample.columns) != 2:
        raise ValueError("Expected a two-column sample submission")

    id_col, target_col = sample.columns.tolist()
    if target_col not in train.columns:
        raise ValueError(f"Training target {target_col!r} is missing")
    feature_cols = [c for c in test.columns if c != id_col]
    if not feature_cols:
        raise ValueError("No feature columns found")
    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    for frame in (X, X_test):
        for col in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[col]):
                frame[col] = pd.to_numeric(frame[col], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
            else:
                frame[col] = frame[col].astype(object)
                frame.loc[pd.isna(frame[col]), col] = np.nan
    y_raw = train[target_col]
    classes = sorted(pd.unique(y_raw.dropna()).tolist())
    if len(classes) != 2:
        raise ValueError(f"Expected binary target, found {classes}")
    y = (y_raw.to_numpy() == classes[-1]).astype(int)

    categorical, numeric = _infer_columns(X)
    categorical_fraction = len(categorical) / max(1, len(feature_cols))
    if len(train) < 2000:
        regime = "small"
    elif categorical_fraction >= 0.35:
        regime = "categorical"
    else:
        regime = "numeric"
    oof, test_preds, errors = _fit_predict_candidates(
        X, y, X_test, categorical, numeric, quick, regime
    )
    candidates, scores, robust_name = _build_blends(y, oof, test_preds)

    output_dir = data_dir if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # Submit a compact, diverse slate: strongest CV individuals plus both blends.
    individuals = sorted(oof, key=lambda n: scores[n], reverse=True)
    chosen = individuals[: min(4, len(individuals))]
    for blend in ("blend_greedy", "blend_diverse"):
        if blend not in chosen:
            chosen.append(blend)
    files = []
    file_by_name = {}
    for name in chosen:
        path = output_dir / f"candidate_{name}.csv"
        submission = sample.copy()
        submission[id_col] = test[id_col].to_numpy()
        submission[target_col] = np.clip(candidates[name], 1e-6, 1 - 1e-6)
        submission.to_csv(path, index=False)
        reported = path.name if relative_output_paths else str(path).replace("\\", "/")
        files.append(reported)
        file_by_name[name] = reported

    result = {
        "profile": "regime-specialist",
        "regime": regime,
        "rows": int(len(train)),
        "features": int(len(feature_cols)),
        "inferred_categorical": int(len(categorical)),
        "cv_auc": {k: round(float(v), 6) for k, v in sorted(scores.items())},
        "errors": errors,
        "recommended_files": files,
        "robust_file": file_by_name[robust_name],
        "elapsed_seconds": round(time.time() - start, 1),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="Short local research run")
    args = parser.parse_args()
    data_dir, discovered_persistent_root = _resolve_runtime_data_dir(args.data_dir)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = data_dir
    result = run(
        data_dir,
        output_dir,
        args.quick,
        relative_output_paths=discovered_persistent_root,
    )
    print("RESULTS_JSON=" + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
