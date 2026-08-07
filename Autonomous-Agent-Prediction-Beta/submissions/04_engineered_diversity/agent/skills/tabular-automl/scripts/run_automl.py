#!/usr/bin/env python3
"""Engineered-diversity AutoML for unseen binary tabular competitions.

The pipeline combines native-categorical models with a separately engineered branch using
frequency and cross-fitted target encoding, duplicate-aware validation, seed diversity,
and correlation-aware rank blends. It writes a compact slate for public-score selection.
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
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

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


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Fold-fitted relative-frequency encoding for high-cardinality columns."""

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.maps_ = [
            frame.iloc[:, j].value_counts(normalize=True, dropna=False)
            for j in range(frame.shape[1])
        ]
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        out = np.zeros(frame.shape, dtype=float)
        for j, mapping in enumerate(self.maps_):
            out[:, j] = frame.iloc[:, j].map(mapping).fillna(0.0).to_numpy()
        return out


def _engineer_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add deterministic, target-free features for the encoded model branch."""
    train = train.copy()
    test = test.copy()
    numeric = train.select_dtypes(include=[np.number]).columns.tolist()
    if numeric:
        for frame in (train, test):
            frame["__row_missing"] = frame.isna().sum(axis=1).astype(float)
            frame["__num_mean"] = frame[numeric].mean(axis=1)
            frame["__num_std"] = frame[numeric].std(axis=1).fillna(0.0)
            frame["__num_min"] = frame[numeric].min(axis=1)
            frame["__num_max"] = frame[numeric].max(axis=1)
        skewed = []
        for col in numeric:
            nonnull = train[col].dropna()
            if len(nonnull) and nonnull.min() >= 0 and abs(float(nonnull.skew())) >= 2.0:
                skewed.append(col)
        for col in skewed[:8]:
            train[f"{col}__log1p"] = np.log1p(train[col].clip(lower=0))
            test[f"{col}__log1p"] = np.log1p(test[col].clip(lower=0))

    date_columns = []
    for col in train.select_dtypes(exclude=[np.number]).columns:
        nonnull = train[col].dropna()
        if nonnull.nunique() < 20:
            continue
        probe = pd.to_datetime(nonnull.head(2000), errors="coerce")
        if len(probe) and probe.notna().mean() >= 0.95:
            date_columns.append(col)
    for col in date_columns:
        for frame in (train, test):
            parsed = pd.to_datetime(frame[col], errors="coerce")
            frame[f"{col}__epoch"] = (parsed - pd.Timestamp("1970-01-01")).dt.days
            frame[f"{col}__year"] = parsed.dt.year
            frame[f"{col}__month"] = parsed.dt.month
            frame[f"{col}__day"] = parsed.dt.day
            frame[f"{col}__dow"] = parsed.dt.dayofweek
            frame.drop(columns=[col], inplace=True)
    return train, test


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


def _tree_preprocessor(
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    seed: int,
) -> ColumnTransformer:
    """Encoded tree branch with extra fold-safe signal for high-card categoricals."""
    blocks = []
    if numeric:
        blocks.append(("num", SimpleImputer(strategy="median", add_indicator=True), numeric))
    if categorical:
        blocks.append(("cat_ord", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2
            )),
        ]), categorical))
        high_card = [
            col for col in categorical
            if frame[col].nunique(dropna=True) > 15
        ]
        if high_card:
            blocks.append(("cat_freq", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("frequency", FrequencyEncoder()),
            ]), high_card))
            try:
                from sklearn.preprocessing import TargetEncoder

                blocks.append(("cat_target", Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("target", TargetEncoder(
                        target_type="binary",
                        smooth="auto",
                        cv=5,
                        random_state=seed,
                    )),
                ]), high_card))
            except Exception:
                pass
    return ColumnTransformer(blocks, remainder="drop")


def _legacy_tree_preprocessor(
    numeric: list[str], categorical: list[str]
) -> ColumnTransformer:
    """The exact ordinal branch used by the completed 0.818 broad profile."""
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


def _row_hash(frame: pd.DataFrame) -> pd.Series:
    """Stable mixed-dtype row hash that does not fail on categorical NaNs."""
    normalized = frame.copy()
    for col in normalized.columns:
        if pd.api.types.is_numeric_dtype(normalized[col]):
            normalized[col] = normalized[col].astype(float)
        else:
            normalized[col] = normalized[col].fillna("__MISSING__").astype(str)
    return pd.util.hash_pandas_object(normalized, index=False)


def _make_splits(
    X: pd.DataFrame, y: np.ndarray, folds: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    groups = _row_hash(X)
    duplicate_rows = int(groups.duplicated(keep=False).sum())
    if duplicate_rows and len(X) >= 1500:
        try:
            cv = StratifiedGroupKFold(
                n_splits=folds, shuffle=True, random_state=SEED
            )
            return list(cv.split(X, y, groups=groups)), duplicate_rows
        except Exception:
            pass
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    return list(cv.split(X, y)), duplicate_rows


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
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str], dict[str, object]]:
    """Return OOF/test predictions; individual failures do not sink the run."""
    n = len(X)
    if quick:
        folds = 3
        rounds = 260
    else:
        folds = 7 if n < 1500 else (5 if n < 3000 else (4 if n < 20000 else 3))
        rounds = 650 if n < 30000 else 500
    categorical_fraction = len(categorical) / max(1, X.shape[1])
    if n < 1500:
        regime = "small"
    elif categorical_fraction >= 0.70:
        regime = "categorical"
    elif categorical_fraction <= 0.15:
        regime = "numeric"
    else:
        regime = "mixed"
    standard_cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    standard_splits = list(standard_cv.split(X, y))
    splits, duplicate_rows = _make_splits(X, y, folds)
    X_encoded, T_encoded = _engineer_features(X, X_test)
    categorical_encoded, numeric_encoded = _infer_columns(X_encoded)
    oof: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}

    def register(name: str, fold_oof: np.ndarray, fold_test: list[np.ndarray]) -> None:
        if np.isfinite(fold_oof).all() and fold_test:
            oof[name] = fold_oof
            test_preds[name] = np.mean(np.vstack(fold_test), axis=0)

    # Broad native CatBoost sweep: shallow, balanced, and deeper interaction regions.
    try:
        from catboost import CatBoostClassifier

        X_cb, T_cb, cat_idx = _prepare_catboost(X, X_test, categorical)
        cb_configs = {
            "catboost_shallow": dict(
                iterations=max(220, int(rounds * 0.8)), depth=5,
                learning_rate=0.06 if not quick else 0.08,
                l2_leaf_reg=10.0, random_strength=0.8, seed_offset=0,
            ),
            "catboost_balanced": dict(
                iterations=rounds, depth=7 if n >= 1500 else 6,
                learning_rate=0.045 if not quick else 0.07,
                l2_leaf_reg=6.0, random_strength=0.5, seed_offset=0,
            ),
            "catboost_deep": dict(
                iterations=max(280, int(rounds * 1.15)), depth=8 if n >= 3000 else 7,
                learning_rate=0.035 if not quick else 0.055,
                l2_leaf_reg=8.0, random_strength=0.3, seed_offset=0,
            ),
        }
        if regime == "small":
            cb_configs["catboost_small_conservative"] = dict(
                iterations=rounds,
                depth=5,
                learning_rate=0.045 if not quick else 0.065,
                l2_leaf_reg=12.0,
                random_strength=0.6,
                seed_offset=0,
            )
            cb_configs["catboost_small_seed2"] = dict(
                iterations=rounds,
                depth=6,
                learning_rate=0.035 if not quick else 0.065,
                l2_leaf_reg=15.0,
                random_strength=0.6,
                seed_offset=7919,
            )
        for config_name, config in cb_configs.items():
            config = dict(config)
            seed_offset = config.pop("seed_offset")
            pred_oof = np.zeros(n)
            pred_test: list[np.ndarray] = []
            # Preserve the proven native-model floor; encoded models use the
            # duplicate-aware splits below as a genuinely different view.
            for fold, (tr, va) in enumerate(standard_splits):
                model = CatBoostClassifier(
                    **config,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    random_seed=SEED + seed_offset + fold,
                    thread_count=4,
                    verbose=False,
                    allow_writing_files=False,
                )
                model.fit(
                    X_cb.iloc[tr], y[tr],
                    cat_features=cat_idx,
                    eval_set=(X_cb.iloc[va], y[va]),
                    early_stopping_rounds=90,
                    verbose=False,
                )
                pred_oof[va] = model.predict_proba(X_cb.iloc[va])[:, 1]
                pred_test.append(model.predict_proba(T_cb)[:, 1])
            register(config_name, pred_oof, pred_test)
    except Exception as exc:
        errors["catboost"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    # Broad LightGBM sweep across regularized, balanced, and high-capacity leaves.
    try:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        lgb_configs = {
            "lightgbm_regularized": dict(
                num_leaves=15, min_child_samples=max(30, min(220, n // 55)),
                subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
            ),
            "lightgbm_balanced": dict(
                num_leaves=31 if n >= 2000 else 15,
                min_child_samples=max(20, min(150, n // 80)),
                subsample=0.85, colsample_bytree=0.8, reg_alpha=0.4, reg_lambda=2.0,
            ),
            "lightgbm_deep": dict(
                num_leaves=63 if n >= 3000 else 31,
                min_child_samples=max(15, min(90, n // 130)),
                subsample=0.9, colsample_bytree=0.9, reg_alpha=0.15, reg_lambda=2.5,
            ),
        }
        for config_name, config in lgb_configs.items():
            pred_oof = np.zeros(n)
            pred_test = []
            for fold, (tr, va) in enumerate(standard_splits):
                prep = _legacy_tree_preprocessor(numeric, categorical)
                X_tr = prep.fit_transform(X.iloc[tr])
                X_va = prep.transform(X.iloc[va])
                T = prep.transform(X_test)
                model = LGBMClassifier(
                    **config,
                    n_estimators=rounds + 250,
                    learning_rate=0.035 if not quick else 0.06,
                    max_depth=-1,
                    objective="binary",
                    random_state=SEED + fold,
                    n_jobs=4,
                    verbosity=-1,
                )
                model.fit(
                    X_tr, y[tr],
                    eval_set=[(X_va, y[va])],
                    eval_metric="auc",
                    callbacks=[early_stopping(90, verbose=False), log_evaluation(0)],
                )
                pred_oof[va] = model.predict_proba(X_va)[:, 1]
                pred_test.append(model.predict_proba(T)[:, 1])
            register(config_name, pred_oof, pred_test)

        # One additional feature-engineered, duplicate-aware challenger.
        pred_oof = np.zeros(n)
        pred_test = []
        for fold, (tr, va) in enumerate(splits):
            prep = _tree_preprocessor(
                X_encoded.iloc[tr], numeric_encoded, categorical_encoded, SEED + fold
            )
            X_tr = prep.fit_transform(X_encoded.iloc[tr], y[tr])
            X_va = prep.transform(X_encoded.iloc[va])
            T = prep.transform(T_encoded)
            model = LGBMClassifier(
                **lgb_configs["lightgbm_balanced"],
                n_estimators=rounds + 250,
                learning_rate=0.035 if not quick else 0.06,
                max_depth=-1,
                objective="binary",
                random_state=SEED + 3000 + fold,
                n_jobs=4,
                verbosity=-1,
            )
            model.fit(
                X_tr, y[tr],
                eval_set=[(X_va, y[va])],
                eval_metric="auc",
                callbacks=[early_stopping(90, verbose=False), log_evaluation(0)],
            )
            pred_oof[va] = model.predict_proba(X_va)[:, 1]
            pred_test.append(model.predict_proba(T)[:, 1])
        register("lightgbm_engineered", pred_oof, pred_test)
    except Exception as exc:
        errors["lightgbm"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    # XGBoost provides a second histogram-boosting inductive bias.
    try:
        from xgboost import XGBClassifier

        pred_oof = np.zeros(n)
        pred_test = []
        for fold, (tr, va) in enumerate(standard_splits):
            prep = _legacy_tree_preprocessor(numeric, categorical)
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
        pred_oof = np.zeros(n)
        pred_test = []
        for fold, (tr, va) in enumerate(standard_splits):
            prep = _legacy_tree_preprocessor(numeric, categorical)
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
        for fold, (tr, va) in enumerate(standard_splits):
            prep = _legacy_tree_preprocessor(numeric, categorical)
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

    # A bagged small-sample expert is less volatile than boosting on 500-1,100 rows.
    if regime == "small":
        try:
            pred_oof = np.zeros(n)
            pred_test = []
            for fold, (tr, va) in enumerate(splits):
                prep = _tree_preprocessor(
                    X_encoded.iloc[tr], numeric_encoded, categorical_encoded, SEED + fold
                )
                X_tr = prep.fit_transform(X_encoded.iloc[tr], y[tr])
                X_va = prep.transform(X_encoded.iloc[va])
                T = prep.transform(T_encoded)
                model = RandomForestClassifier(
                    n_estimators=400 if quick else 750,
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
        for tr, va in standard_splits:
            model = clone(template)
            model.fit(X.iloc[tr], y[tr])
            pred_oof[va] = model.predict_proba(X.iloc[va])[:, 1]
            pred_test.append(model.predict_proba(X_test)[:, 1])
        register("linear", pred_oof, pred_test)
    except Exception as exc:
        errors["linear"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    diagnostics = {
        "regime": regime,
        "folds": folds,
        "duplicate_rows": duplicate_rows,
        "engineered_features": int(X_encoded.shape[1] - X.shape[1]),
    }
    return oof, test_preds, errors, diagnostics


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

    # Correlation-aware blend: accept slightly weaker OOF members when they add a new ranking.
    corr_selected = [ordered[0]]
    eligible = [
        name for name in ordered[1:8]
        if scores[name] >= scores[ordered[0]] - 0.02
    ]
    while eligible and len(corr_selected) < 4:
        candidate = min(
            eligible,
            key=lambda name: np.mean([
                abs(np.corrcoef(ranked_oof[name], ranked_oof[chosen])[0, 1])
                for chosen in corr_selected
            ]),
        )
        mean_corr = np.mean([
            abs(np.corrcoef(ranked_oof[candidate], ranked_oof[chosen])[0, 1])
            for chosen in corr_selected
        ])
        if mean_corr > 0.995:
            break
        corr_selected.append(candidate)
        eligible.remove(candidate)
    corr_oof = np.mean(np.vstack([ranked_oof[n] for n in corr_selected]), axis=0)
    candidates["blend_correlation"] = np.mean(
        np.vstack([ranked_test[n] for n in corr_selected]), axis=0
    )
    scores["blend_correlation"] = _safe_auc(y, corr_oof)

    # A seed/depth CatBoost consensus is robust to fold noise on categorical tasks.
    cat_names = [name for name in ordered if name.startswith("catboost_")]
    if len(cat_names) >= 2:
        cat_oof = np.mean(np.vstack([ranked_oof[n] for n in cat_names]), axis=0)
        candidates["blend_catboost"] = np.mean(
            np.vstack([ranked_test[n] for n in cat_names]), axis=0
        )
        scores["blend_catboost"] = _safe_auc(y, cat_oof)

    # Coarse weighted rank blend over top pairs; coarse weights resist OOF overfitting.
    weighted_best = (-1.0, ordered[0], ordered[0], 0.5, ranked_oof[ordered[0]])
    pair_pool = ordered[: min(4, len(ordered))]
    for i, left in enumerate(pair_pool):
        for right in pair_pool[i + 1:]:
            for weight in (0.2, 0.35, 0.5, 0.65, 0.8):
                trial = weight * ranked_oof[left] + (1 - weight) * ranked_oof[right]
                auc = _safe_auc(y, trial)
                if auc > weighted_best[0]:
                    weighted_best = (auc, left, right, weight, trial)
    weighted_auc, left, right, weight, _ = weighted_best
    candidates["blend_weighted"] = (
        weight * ranked_test[left] + (1 - weight) * ranked_test[right]
    )
    scores["blend_weighted"] = weighted_auc

    # Recreate the v4 blends from only the original model family. These are an
    # immutable public-feedback floor while new challengers remain additive.
    legacy_family = {
        "catboost_shallow", "catboost_balanced", "catboost_deep",
        "lightgbm_regularized", "lightgbm_balanced", "lightgbm_deep",
        "xgboost", "histgb", "extratrees", "linear",
    }
    legacy_order = [name for name in ordered if name in legacy_family]
    if legacy_order:
        legacy_selected = [legacy_order[0]]
        legacy_oof = ranked_oof[legacy_order[0]].copy()
        legacy_best = _safe_auc(y, legacy_oof)
        legacy_remaining = legacy_order[1:]
        while legacy_remaining and len(legacy_selected) < 4:
            trials = []
            for name in legacy_remaining:
                trial = (
                    legacy_oof * len(legacy_selected) + ranked_oof[name]
                ) / (len(legacy_selected) + 1)
                trials.append((_safe_auc(y, trial), name, trial))
            score, name, trial = max(trials, key=lambda item: item[0])
            if score <= legacy_best + 0.00015:
                break
            legacy_selected.append(name)
            legacy_remaining.remove(name)
            legacy_oof, legacy_best = trial, score
        candidates["legacy_greedy"] = np.mean(
            np.vstack([ranked_test[n] for n in legacy_selected]), axis=0
        )
        scores["legacy_greedy"] = legacy_best

        legacy_top = legacy_order[: min(4, len(legacy_order))]
        candidates["legacy_diverse"] = np.mean(
            np.vstack([ranked_test[n] for n in legacy_top]), axis=0
        )
        scores["legacy_diverse"] = _safe_auc(
            y, np.mean(np.vstack([ranked_oof[n] for n in legacy_top]), axis=0)
        )

        legacy_weighted = (-1.0, legacy_order[0], legacy_order[0], 0.5)
        legacy_pool = legacy_order[: min(4, len(legacy_order))]
        for i, legacy_left in enumerate(legacy_pool):
            for legacy_right in legacy_pool[i + 1:]:
                for legacy_weight in (0.2, 0.35, 0.5, 0.65, 0.8):
                    trial = (
                        legacy_weight * ranked_oof[legacy_left]
                        + (1 - legacy_weight) * ranked_oof[legacy_right]
                    )
                    auc = _safe_auc(y, trial)
                    if auc > legacy_weighted[0]:
                        legacy_weighted = (
                            auc, legacy_left, legacy_right, legacy_weight
                        )
        legacy_auc, legacy_left, legacy_right, legacy_weight = legacy_weighted
        candidates["legacy_weighted"] = (
            legacy_weight * ranked_test[legacy_left]
            + (1 - legacy_weight) * ranked_test[legacy_right]
        )
        scores["legacy_weighted"] = legacy_auc

    blend_names = [
        name for name in (
            "blend_greedy",
            "blend_diverse",
            "blend_correlation",
            "blend_catboost",
            "blend_weighted",
            "legacy_greedy",
            "legacy_diverse",
            "legacy_weighted",
        )
        if name in scores
    ]
    robust_name = max(blend_names, key=scores.get)
    return candidates, scores, robust_name


def _stabilize_duplicate_predictions(
    predictions: np.ndarray, test: pd.DataFrame
) -> np.ndarray:
    """Identical test feature vectors carry no distinguishing information."""
    keys = _row_hash(test).to_numpy()
    frame = pd.DataFrame({"key": keys, "prediction": np.asarray(predictions, float)})
    return frame.groupby("key")["prediction"].transform("mean").to_numpy()


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
    oof, test_preds, errors, diagnostics = _fit_predict_candidates(
        X, y, X_test, categorical, numeric, quick
    )
    candidates, scores, robust_name = _build_blends(y, oof, test_preds)

    output_dir = data_dir if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use public feedback efficiently: five strong individuals plus orthogonal blends.
    individuals = sorted(oof, key=lambda n: scores[n], reverse=True)
    chosen = individuals[: min(5, len(individuals))]
    for blend in (
        "blend_greedy",
        "blend_diverse",
        "blend_correlation",
        "blend_catboost",
        "blend_weighted",
        "legacy_greedy",
        "legacy_diverse",
        "legacy_weighted",
    ):
        if blend in candidates and blend not in chosen:
            chosen.append(blend)
    files = []
    file_by_name = {}
    for name in chosen:
        path = output_dir / f"candidate_{name}.csv"
        submission = sample.copy()
        submission[id_col] = test[id_col].to_numpy()
        legacy_individuals = {
            "catboost_shallow", "catboost_balanced", "catboost_deep",
            "lightgbm_regularized", "lightgbm_balanced", "lightgbm_deep",
            "xgboost", "histgb", "extratrees", "linear",
        }
        if name.startswith("legacy_") or name in legacy_individuals:
            stable = candidates[name]
        else:
            stable = _stabilize_duplicate_predictions(candidates[name], X_test)
        submission[target_col] = np.clip(stable, 1e-6, 1 - 1e-6)
        submission.to_csv(path, index=False)
        reported = path.name if relative_output_paths else str(path).replace("\\", "/")
        files.append(reported)
        file_by_name[name] = reported

    result = {
        "profile": "engineered-diversity-v5",
        "trained_models": int(len(oof)),
        "rows": int(len(train)),
        "features": int(len(feature_cols)),
        "inferred_categorical": int(len(categorical)),
        **diagnostics,
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
