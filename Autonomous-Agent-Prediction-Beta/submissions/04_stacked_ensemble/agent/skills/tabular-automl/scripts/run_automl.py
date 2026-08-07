#!/usr/bin/env python3
"""Deterministic CPU AutoML for the Autonomous Agent Prediction competition.

Stacked-ensemble profile. Compared with the broad-portfolio profile it adds the
standard Kaggle tabular playbook: frequency-encoded features, repeated CV on
small tables, a LightGBM extra-trees variant and an MLP for blend diversity,
bagged Caruana hill-climb ensemble selection, and a level-2 logistic stacker on
logit-space out-of-fold predictions. Everything stays leakage-safe and
time-aware so the session limit can never be exceeded.
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
from scipy.special import logit as _sp_logit
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    StandardScaler,
)

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

SEED = 20260720
# Soft compute budget for base-model training; blending afterwards is cheap.
FULL_BUDGET_SECONDS = 2400
QUICK_BUDGET_SECONDS = 1100


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (rankdata(values, method="average") - 0.5) / len(values)


def _logit(values: np.ndarray) -> np.ndarray:
    return _sp_logit(np.clip(values, 1e-6, 1 - 1e-6))


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


def _engineer_features(
    X: pd.DataFrame, X_test: pd.DataFrame, categorical: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Label-free engineered columns: value frequencies and row missing counts."""
    added: list[str] = []
    eng_tr: dict[str, np.ndarray] = {}
    eng_te: dict[str, np.ndarray] = {}
    if X.isna().to_numpy().any() or X_test.isna().to_numpy().any():
        eng_tr["fe_nan_count"] = X.isna().sum(axis=1).to_numpy(dtype=float)
        eng_te["fe_nan_count"] = X_test.isna().sum(axis=1).to_numpy(dtype=float)
        added.append("fe_nan_count")
    for col in categorical:
        combined = pd.concat([X[col], X_test[col]], axis=0)
        if combined.nunique(dropna=True) < 3:
            continue
        counts = combined.value_counts(dropna=False)
        name = f"fe_freq_{col}"
        eng_tr[name] = X[col].map(counts).astype(float).to_numpy()
        eng_te[name] = X_test[col].map(counts).astype(float).to_numpy()
        added.append(name)
    if not added:
        return X, X_test, []
    X_out = pd.concat([X, pd.DataFrame(eng_tr, index=X.index)], axis=1)
    T_out = pd.concat([X_test, pd.DataFrame(eng_te, index=X_test.index)], axis=1)
    return X_out, T_out, added


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
    start_time: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str], list[str]]:
    """Return OOF/test predictions; individual failures do not sink the run."""
    n = len(X)
    if quick:
        rounds = 260
        splits = list(
            StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y)
        )
        n_repeats = 1
    else:
        rounds = 650 if n < 30000 else 500
        if n < 1500:
            # Repeated CV seed-bags every model on small, noisy tables.
            n_repeats = 2
            splits = list(
                RepeatedStratifiedKFold(
                    n_splits=5, n_repeats=n_repeats, random_state=SEED
                ).split(X, y)
            )
        else:
            n_repeats = 1
            folds = 5 if n < 20000 else 4
            splits = list(
                StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED).split(X, y)
            )
    budget = QUICK_BUDGET_SECONDS if quick else FULL_BUDGET_SECONDS

    oof: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    skipped: list[str] = []

    def run_cv(name: str, fold_fn) -> None:
        """fold_fn(tr, va, fold) -> (val_pred, test_pred)."""
        try:
            oof_sum = np.zeros(n)
            oof_cnt = np.zeros(n)
            fold_test: list[np.ndarray] = []
            for fold, (tr, va) in enumerate(splits):
                val_pred, tst_pred = fold_fn(tr, va, fold)
                oof_sum[va] += val_pred
                oof_cnt[va] += 1
                fold_test.append(tst_pred)
            if fold_test and oof_cnt.min() > 0:
                pred_oof = oof_sum / oof_cnt
                if np.isfinite(pred_oof).all():
                    oof[name] = pred_oof
                    test_preds[name] = np.mean(np.vstack(fold_test), axis=0)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {str(exc)[:160]}"

    def over_budget(optional: bool) -> bool:
        elapsed = time.time() - start_time
        return optional and elapsed > budget

    # --- CatBoost with native categorical handling ---------------------------------
    try:
        from catboost import CatBoostClassifier

        X_cb, T_cb, cat_idx = _prepare_catboost(X, X_test, categorical)
        cb_configs = {
            "catboost_balanced": dict(
                iterations=rounds, depth=7 if n >= 1500 else 6,
                learning_rate=0.045 if not quick else 0.07,
                l2_leaf_reg=6.0, random_strength=0.5,
            ),
            "catboost_shallow": dict(
                iterations=max(220, int(rounds * 0.8)), depth=5,
                learning_rate=0.06 if not quick else 0.08,
                l2_leaf_reg=10.0, random_strength=0.8,
            ),
            "catboost_deep": dict(
                iterations=max(280, int(rounds * 1.15)), depth=8 if n >= 3000 else 7,
                learning_rate=0.035 if not quick else 0.055,
                l2_leaf_reg=8.0, random_strength=0.3,
            ),
        }

        def cb_fold(config):
            def _fit(tr, va, fold):
                model = CatBoostClassifier(
                    **config,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    random_seed=SEED + fold,
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
                return (
                    model.predict_proba(X_cb.iloc[va])[:, 1],
                    model.predict_proba(T_cb)[:, 1],
                )
            return _fit

        for config_name, config in cb_configs.items():
            if over_budget(optional=config_name != "catboost_balanced"):
                skipped.append(config_name)
                continue
            run_cv(config_name, cb_fold(config))
    except Exception as exc:
        errors["catboost"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    # --- LightGBM sweep plus an extra-trees-mode variant ---------------------------
    try:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        lgb_configs = {
            "lightgbm_balanced": dict(
                num_leaves=31 if n >= 2000 else 15,
                min_child_samples=max(20, min(150, n // 80)),
                subsample=0.85, colsample_bytree=0.8, reg_alpha=0.4, reg_lambda=2.0,
            ),
            "lightgbm_regularized": dict(
                num_leaves=15, min_child_samples=max(30, min(220, n // 55)),
                subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
            ),
            "lightgbm_deep": dict(
                num_leaves=63 if n >= 3000 else 31,
                min_child_samples=max(15, min(90, n // 130)),
                subsample=0.9, colsample_bytree=0.9, reg_alpha=0.15, reg_lambda=2.5,
            ),
            "lightgbm_extratrees": dict(
                num_leaves=63 if n >= 3000 else 31,
                min_child_samples=max(15, min(90, n // 130)),
                subsample=0.9, colsample_bytree=0.7, reg_alpha=0.2, reg_lambda=2.0,
                extra_trees=True,
            ),
        }

        def lgb_fold(config):
            def _fit(tr, va, fold):
                prep = _tree_preprocessor(numeric, categorical)
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
                return model.predict_proba(X_va)[:, 1], model.predict_proba(T)[:, 1]
            return _fit

        for config_name, config in lgb_configs.items():
            if over_budget(optional=config_name != "lightgbm_balanced"):
                skipped.append(config_name)
                continue
            run_cv(config_name, lgb_fold(config))
    except Exception as exc:
        errors["lightgbm"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    # --- XGBoost: a second histogram-boosting inductive bias -----------------------
    try:
        from xgboost import XGBClassifier

        def xgb_fold(tr, va, fold):
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
            return model.predict_proba(X_va)[:, 1], model.predict_proba(T)[:, 1]

        if over_budget(optional=False):
            skipped.append("xgboost")
        else:
            run_cv("xgboost", xgb_fold)
    except Exception as exc:
        errors["xgboost"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    # --- Histogram gradient boosting: stable sklearn-only member -------------------
    def hgb_fold(tr, va, fold):
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
        return model.predict_proba(X_va)[:, 1], model.predict_proba(T)[:, 1]

    if over_budget(optional=True):
        skipped.append("histgb")
    else:
        run_cv("histgb", hgb_fold)

    # --- ExtraTrees: deliberately less correlated with boosting --------------------
    def et_fold(tr, va, fold):
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
        return model.predict_proba(X_va)[:, 1], model.predict_proba(T)[:, 1]

    if over_budget(optional=True):
        skipped.append("extratrees")
    else:
        run_cv("extratrees", et_fold)

    # --- Regularized linear/OHE: additive effects, cheap ---------------------------
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

        def lin_fold(tr, va, fold):
            model = clone(template)
            model.fit(X.iloc[tr], y[tr])
            return (
                model.predict_proba(X.iloc[va])[:, 1],
                model.predict_proba(X_test)[:, 1],
            )

        run_cv("linear", lin_fold)
    except Exception as exc:
        errors["linear"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    # --- MLP on quantile-normalized features: adds a non-tree bias to blends -------
    try:
        all_cols = numeric + categorical
        mlp_prep = ColumnTransformer([("all", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("ordinal_free", "passthrough"),
        ]), numeric)] + ([("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2
            )),
        ]), categorical)] if categorical else []), remainder="drop")

        def mlp_fold(tr, va, fold):
            prep = clone(mlp_prep)
            qt = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(600, max(10, len(tr) // 2)),
                subsample=200000,
                random_state=SEED + fold,
            )
            X_tr = qt.fit_transform(np.asarray(prep.fit_transform(X.iloc[tr]), dtype=float))
            X_va = qt.transform(np.asarray(prep.transform(X.iloc[va]), dtype=float))
            T = qt.transform(np.asarray(prep.transform(X_test), dtype=float))
            model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=2e-3,
                learning_rate_init=2e-3,
                batch_size=min(256, max(32, len(tr) // 8)),
                max_iter=60 if quick else 160,
                early_stopping=True,
                n_iter_no_change=12,
                validation_fraction=0.15,
                random_state=SEED + fold,
            )
            model.fit(X_tr, y[tr])
            return model.predict_proba(X_va)[:, 1], model.predict_proba(T)[:, 1]

        if over_budget(optional=True) or not all_cols:
            skipped.append("mlp")
        else:
            run_cv("mlp", mlp_fold)
    except Exception as exc:
        errors["mlp"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    return oof, test_preds, errors, skipped


def _hillclimb_weights(
    y: np.ndarray,
    ranked_oof: dict[str, np.ndarray],
    names: list[str],
    steps: int = 22,
    bags: int = 8,
    rng_seed: int = SEED,
) -> dict[str, float]:
    """Bagged Caruana ensemble selection with replacement on rank-space OOF."""
    rng = np.random.default_rng(rng_seed)
    n = len(y)
    total: dict[str, float] = {name: 0.0 for name in names}
    for bag in range(bags):
        if bags > 1:
            idx = rng.choice(n, size=max(50, int(n * 0.8)), replace=False)
        else:
            idx = np.arange(n)
        y_bag = y[idx]
        preds = {name: ranked_oof[name][idx] for name in names}
        picks: list[str] = []
        blend = np.zeros(len(idx))
        best = -1.0
        for _ in range(steps):
            trial_best = (-1.0, None)
            for name in names:
                trial = (blend * len(picks) + preds[name]) / (len(picks) + 1)
                auc = _safe_auc(y_bag, trial)
                if auc > trial_best[0]:
                    trial_best = (auc, name)
            auc, name = trial_best
            if name is None or (picks and auc <= best + 1e-5):
                break
            picks.append(name)
            blend = (blend * (len(picks) - 1) + preds[name]) / len(picks)
            best = auc
        for name in picks:
            total[name] += 1.0 / (len(picks) * bags)
    weight_sum = sum(total.values())
    if weight_sum <= 0:
        return {names[0]: 1.0}
    return {k: v / weight_sum for k, v in total.items() if v > 0}


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

    candidates["blend_greedy"] = np.mean(
        np.vstack([ranked_test[n] for n in selected]), axis=0
    )
    scores["blend_greedy"] = best

    # Conservative diversity blend guards against noisy CV on the 500-row tasks.
    top = ordered[: min(4, len(ordered))]
    conservative_oof = np.mean(np.vstack([ranked_oof[n] for n in top]), axis=0)
    candidates["blend_diverse"] = np.mean(np.vstack([ranked_test[n] for n in top]), axis=0)
    scores["blend_diverse"] = _safe_auc(y, conservative_oof)

    # Coarse weighted rank blend over top pairs; coarse weights resist OOF overfitting.
    weighted_best = (-1.0, ordered[0], ordered[0], 0.5)
    pair_pool = ordered[: min(4, len(ordered))]
    for i, left in enumerate(pair_pool):
        for right in pair_pool[i + 1:]:
            for weight in (0.2, 0.35, 0.5, 0.65, 0.8):
                trial = weight * ranked_oof[left] + (1 - weight) * ranked_oof[right]
                auc = _safe_auc(y, trial)
                if auc > weighted_best[0]:
                    weighted_best = (auc, left, right, weight)
    weighted_auc, left, right, weight = weighted_best
    candidates["blend_weighted"] = (
        weight * ranked_test[left] + (1 - weight) * ranked_test[right]
    )
    scores["blend_weighted"] = weighted_auc

    # Bagged hill-climb ensemble selection over every model, with replacement.
    hc_weights = _hillclimb_weights(y, ranked_oof, ordered)
    hc_oof = np.zeros(len(y))
    hc_test = np.zeros(len(next(iter(ranked_test.values()))))
    for name, w in hc_weights.items():
        hc_oof += w * ranked_oof[name]
        hc_test += w * ranked_test[name]
    candidates["blend_hillclimb"] = hc_test
    scores["blend_hillclimb"] = _safe_auc(y, hc_oof)

    # Level-2 logistic stacker on logit-space OOF predictions.
    try:
        stack_names = ordered[: min(8, len(ordered))]
        P = np.column_stack([_logit(oof[name]) for name in stack_names])
        T = np.column_stack([_logit(test_preds[name]) for name in stack_names])
        meta = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000)),
        ])
        stack_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        stack_oof = cross_val_predict(
            meta, P, y, cv=stack_cv, method="predict_proba", n_jobs=1
        )[:, 1]
        meta.fit(P, y)
        candidates["stack_lr"] = meta.predict_proba(T)[:, 1]
        scores["stack_lr"] = _safe_auc(y, stack_oof)

        # Rank-average of the stacker and hill-climb: two strong, different combiners.
        stack_plus_oof = 0.5 * _rank01(stack_oof) + 0.5 * _rank01(hc_oof)
        candidates["stack_plus"] = 0.5 * _rank01(candidates["stack_lr"]) + 0.5 * _rank01(
            hc_test
        )
        scores["stack_plus"] = _safe_auc(y, stack_plus_oof)
    except Exception:
        pass

    return candidates, scores


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
    X, X_test, engineered = _engineer_features(X, X_test, categorical)
    numeric_all = numeric + engineered
    oof, test_preds, errors, skipped = _fit_predict_candidates(
        X, y, X_test, categorical, numeric_all, quick, start
    )
    candidates, scores = _build_blends(y, oof, test_preds)

    output_dir = data_dir if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # Submit only the strong combiners: on the 16 research tasks every combiner
    # beat every raw model on private AUC, and extra candidates only let the
    # noisy public split pick a worse file. Individuals are a failure fallback.
    chosen = [
        name
        for name in ("stack_plus", "stack_lr", "blend_hillclimb", "blend_greedy")
        if name in candidates
    ]
    if len(chosen) < 2:
        individuals = sorted(oof, key=lambda n: scores[n], reverse=True)
        for name in individuals[:3]:
            if name not in chosen:
                chosen.append(name)
    robust_name = max(chosen, key=scores.get)
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
        "profile": "stacked-ensemble",
        "trained_models": int(len(oof)),
        "rows": int(len(train)),
        "features": int(len(feature_cols)),
        "inferred_categorical": int(len(categorical)),
        "engineered_features": int(len(engineered)),
        "skipped_models": skipped,
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
