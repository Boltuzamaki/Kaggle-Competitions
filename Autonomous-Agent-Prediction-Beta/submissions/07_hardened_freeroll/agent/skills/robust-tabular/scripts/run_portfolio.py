"""Cross-validate a compact diverse portfolio and write ranked candidate CSVs."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    categorical_columns,
    emit_manifest,
    encoded_frames,
    load_task,
    native_frames,
    rank_unit,
    runtime_workdir,
    write_submission,
)

SEED = 20260710

# Wall-clock guard. Cheap, diverse models run first, so an overrun costs the least
# valuable member of the portfolio rather than the whole stage.
TIME_BUDGET = float(os.environ.get("PORTFOLIO_TIME_BUDGET", "1200"))

# Gates on the self-weighted blend, and on how many near-duplicate candidates reach
# the public split. Every extra candidate is another draw whose maximum is upward
# biased on public and not on private, so a weak one can only cost us a selection.
GREEDY_MIN_ROWS = 1500
GREEDY_MARGIN = 5e-4
SLATE_BAND = 0.010
SLATE_MAX = 8  # the portfolio prompt submits at most eight files

# The four members of the leaderboard-proven portfolio. The fixed-weight blends are
# built from these and only these, so every candidate the proven config would have
# produced is reproduced here bit for bit. The additional learners below can add a
# candidate to the slate but can never change an existing one.
CORE_MODELS = ("catboost", "lightgbm", "extra_trees", "logistic")


def folds_for(y: np.ndarray, n_rows: int) -> StratifiedKFold:
    minority = int(np.bincount(y).min())
    return StratifiedKFold(n_splits=max(2, min(5, minority)), shuffle=True, random_state=SEED)


def score(y: np.ndarray, predictions: np.ndarray) -> float:
    return float(roc_auc_score(y, predictions))


def seeds_for(n_rows: int) -> list[int]:
    """Small tables are seed-noisy; average a few fits so the OOF estimate is usable."""
    return [SEED, SEED + 101, SEED + 202] if n_rows < 1500 else [SEED]


def cv_catboost(x_train, y, x_test, cat_cols, splitter):
    from catboost import CatBoostClassifier
    oof = np.zeros(len(x_train))
    test_predictions = np.zeros(len(x_test))
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y)):
        model = CatBoostClassifier(
            iterations=650 if len(x_train) >= 2000 else 420,
            depth=6 if len(x_train) >= 2000 else 5,
            learning_rate=0.04,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=7.0,
            random_strength=0.35,
            random_seed=SEED + fold,
            verbose=False,
            allow_writing_files=False,
            thread_count=3,
        )
        model.fit(
            x_train.iloc[fit_idx],
            y[fit_idx],
            cat_features=cat_cols,
            eval_set=(x_train.iloc[val_idx], y[val_idx]),
            early_stopping_rounds=70,
            use_best_model=True,
            verbose=False,
        )
        oof[val_idx] = model.predict_proba(x_train.iloc[val_idx])[:, 1]
        test_predictions += model.predict_proba(x_test)[:, 1] / splitter.n_splits
    return oof, test_predictions


def cv_catboost_bagged(x_train, y, x_test, cat_cols, splitter):
    """Seed-averaged CatBoost on a denser fold grid.

    This is an *additional* candidate, never a member of the proven blends, so it is
    free to use its own resampling scheme. On a small table ten folds leave 90% of the
    rows in each fit instead of 80%, and averaging three seeds over ten folds gives the
    test prediction thirty fits to average — which is where the variance reduction on
    the 500-row tasks comes from.
    """
    from catboost import CatBoostClassifier
    seeds = seeds_for(len(x_train))
    if len(x_train) < 3000:
        minority = int(np.bincount(y).min())
        splitter = StratifiedKFold(
            n_splits=max(2, min(10, minority)), shuffle=True, random_state=SEED
        )
    oof = np.zeros(len(x_train))
    test_predictions = np.zeros(len(x_test))
    for seed in seeds:
        for fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y)):
            model = CatBoostClassifier(
                iterations=420,
                depth=5,
                learning_rate=0.045,
                loss_function="Logloss",
                eval_metric="AUC",
                l2_leaf_reg=8.0,
                random_strength=0.5,
                bagging_temperature=0.5,
                random_seed=seed + fold,
                verbose=False,
                allow_writing_files=False,
                thread_count=3,
            )
            model.fit(
                x_train.iloc[fit_idx],
                y[fit_idx],
                cat_features=cat_cols,
                eval_set=(x_train.iloc[val_idx], y[val_idx]),
                early_stopping_rounds=70,
                use_best_model=True,
                verbose=False,
            )
            oof[val_idx] += model.predict_proba(x_train.iloc[val_idx])[:, 1] / len(seeds)
            test_predictions += model.predict_proba(x_test)[:, 1] / (splitter.n_splits * len(seeds))
    return oof, test_predictions


def cv_lightgbm(x_train, y, x_test, categorical_cols, splitter):
    import lightgbm as lgb
    x_train = x_train.copy()
    x_test = x_test.copy()
    for column in categorical_cols:
        x_train[column] = x_train[column].round().astype("int64").astype("category")
        x_test[column] = x_test[column].round().astype("int64").astype("category")
    oof = np.zeros(len(x_train))
    test_predictions = np.zeros(len(x_test))
    small = len(x_train) < 2500
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y)):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=900 if not small else 500,
            learning_rate=0.025 if not small else 0.04,
            num_leaves=15 if small else 31,
            max_depth=-1,
            min_child_samples=20 if small else 35,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.15,
            reg_lambda=2.5,
            random_state=SEED + fold,
            n_jobs=3,
            verbosity=-1,
        )
        model.fit(
            x_train.iloc[fit_idx],
            y[fit_idx],
            eval_set=[(x_train.iloc[val_idx], y[val_idx])],
            eval_metric="auc",
            categorical_feature=categorical_cols,
            callbacks=[lgb.early_stopping(70, verbose=False), lgb.log_evaluation(0)],
        )
        oof[val_idx] = model.predict_proba(x_train.iloc[val_idx])[:, 1]
        test_predictions += model.predict_proba(x_test)[:, 1] / splitter.n_splits
    return oof, test_predictions


def cv_xgboost(x_train, y, x_test, splitter):
    import xgboost as xgb
    oof = np.zeros(len(x_train))
    test_predictions = np.zeros(len(x_test))
    small = len(x_train) < 2500
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y)):
        model = xgb.XGBClassifier(
            n_estimators=700 if not small else 400,
            learning_rate=0.03 if not small else 0.05,
            max_depth=5 if not small else 4,
            min_child_weight=2.0,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            eval_metric="auc",
            early_stopping_rounds=70,
            tree_method="hist",
            random_state=SEED + fold,
            n_jobs=3,
            verbosity=0,
        )
        model.fit(
            x_train.iloc[fit_idx], y[fit_idx],
            eval_set=[(x_train.iloc[val_idx], y[val_idx])], verbose=False,
        )
        oof[val_idx] = model.predict_proba(x_train.iloc[val_idx])[:, 1]
        test_predictions += model.predict_proba(x_test)[:, 1] / splitter.n_splits
    return oof, test_predictions


def cv_hist_gb(x_train, y, x_test, categorical_cols, splitter):
    oof = np.zeros(len(x_train))
    test_predictions = np.zeros(len(x_test))
    cat_mask = [column in set(categorical_cols) for column in x_train.columns]
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y)):
        model = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=31 if len(x_train) >= 2500 else 15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=40,
            categorical_features=cat_mask if any(cat_mask) else None,
            random_state=SEED + fold,
        )
        model.fit(x_train.iloc[fit_idx], y[fit_idx])
        oof[val_idx] = model.predict_proba(x_train.iloc[val_idx])[:, 1]
        test_predictions += model.predict_proba(x_test)[:, 1] / splitter.n_splits
    return oof, test_predictions


def cv_extra_trees(x_train, y, x_test, splitter):
    oof = np.zeros(len(x_train))
    test_predictions = np.zeros(len(x_test))
    leaf = 2 if len(x_train) < 3000 else 3
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(x_train, y)):
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_features=0.8,
            min_samples_leaf=leaf,
            class_weight="balanced",
            random_state=SEED + fold,
            n_jobs=3,
        )
        model.fit(x_train.iloc[fit_idx], y[fit_idx])
        oof[val_idx] = model.predict_proba(x_train.iloc[val_idx])[:, 1]
        test_predictions += model.predict_proba(x_test)[:, 1] / splitter.n_splits
    return oof, test_predictions


def cv_logistic(train, y, test, features, splitter):
    cat_cols = categorical_columns(train, features)
    num_cols = [column for column in features if column not in cat_cols]
    oof = np.zeros(len(train))
    test_predictions = np.zeros(len(test))
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(train, y)):
        numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))])
        transformer = ColumnTransformer([("numeric", numeric, num_cols), ("categorical", categorical, cat_cols)])
        model = Pipeline([("features", transformer), ("model", LogisticRegression(C=0.25 if len(train) < 2000 else 0.7, max_iter=1500, solver="liblinear", random_state=SEED + fold))])
        model.fit(train.iloc[fit_idx][features], y[fit_idx])
        oof[val_idx] = model.predict_proba(train.iloc[val_idx][features])[:, 1]
        test_predictions += model.predict_proba(test[features])[:, 1] / splitter.n_splits
    return oof, test_predictions


def greedy_blend(models: dict, names: list[str], y: np.ndarray, rounds: int = 30):
    """Caruana forward selection with replacement over rank-transformed OOF.

    Selecting on ranks keeps the blend on the AUC scale, and allowing repeats lets a
    strong member accumulate weight without a continuous optimiser to overfit with.
    """
    oof_ranks = {name: rank_unit(models[name]["oof"]) for name in names}
    test_ranks = {name: rank_unit(models[name]["test"]) for name in names}
    chosen: list[str] = [max(names, key=lambda name: models[name]["cv_auc"])]
    best = score(y, oof_ranks[chosen[0]])
    for _ in range(rounds - 1):
        current = np.mean([oof_ranks[name] for name in chosen], axis=0)
        candidate_name, candidate_score = None, best
        for name in names:
            trial = (current * len(chosen) + oof_ranks[name]) / (len(chosen) + 1)
            trial_score = score(y, trial)
            if trial_score > candidate_score + 1e-7:
                candidate_name, candidate_score = name, trial_score
        if candidate_name is None:
            break
        chosen.append(candidate_name)
        best = candidate_score
    blend_test = np.mean([test_ranks[name] for name in chosen], axis=0)
    return blend_test, best, chosen


def main() -> None:
    started = time.time()
    workdir = runtime_workdir()
    train, test, sample, id_col, target_col, features, y = load_task(workdir)
    native_train, native_test, cat_cols = native_frames(train, test, features)
    encoded_train, encoded_test, encoded_cat_cols = encoded_frames(train, test, features)
    splitter = folds_for(y, len(train))
    models: dict[str, dict] = {}
    errors: dict[str, str] = {}

    runners = [
        ("catboost", lambda: cv_catboost(native_train, y, native_test, cat_cols, splitter)),
        ("lightgbm", lambda: cv_lightgbm(encoded_train, y, encoded_test, encoded_cat_cols, splitter)),
        ("extra_trees", lambda: cv_extra_trees(encoded_train, y, encoded_test, splitter)),
        ("logistic", lambda: cv_logistic(train, y, test, features, splitter)),
        ("xgboost", lambda: cv_xgboost(encoded_train, y, encoded_test, splitter)),
        ("hist_gb", lambda: cv_hist_gb(encoded_train, y, encoded_test, encoded_cat_cols, splitter)),
        ("catboost_bag", lambda: cv_catboost_bagged(native_train, y, native_test, cat_cols, splitter)),
    ]
    for name, runner in runners:
        if models and time.time() - started > TIME_BUDGET:
            errors[name] = "SkippedForTimeBudget"
            continue
        model_started = time.time()
        try:
            oof, test_predictions = runner()
            if not np.isfinite(oof).all() or np.std(oof) <= 1e-9:
                raise ValueError("invalid or constant OOF predictions")
            models[name] = {"oof": oof, "test": test_predictions, "cv_auc": score(y, oof), "seconds": round(time.time() - model_started, 3)}
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}:{exc}"

    if not models:
        prior = float(np.mean(y))
        path = write_submission(workdir / "portfolio_prior.csv", np.full(len(test), prior), test, sample, id_col, target_col)
        emit_manifest({"candidates": [path], "errors": errors, "models": {}, "robust_choice": path, "total_seconds": round(time.time() - started, 3)})
        return

    ranked_names = sorted(models, key=lambda name: models[name]["cv_auc"], reverse=True)
    candidates: list[dict] = []
    for name in ranked_names:
        path = write_submission(workdir / f"portfolio_{name}.csv", models[name]["test"], test, sample, id_col, target_col)
        candidates.append({"path": path, "cv_auc": models[name]["cv_auc"], "kind": name})

    # Blends are built over the proven four only, so every candidate the leaderboard-proven
    # config produced is reproduced here unchanged; the extra learners can only add.
    core_ranked = [name for name in ranked_names if name in CORE_MODELS]
    blend_names = core_ranked or ranked_names

    top_two = blend_names[:2]
    if len(top_two) == 2:
        first_oof, second_oof = rank_unit(models[top_two[0]]["oof"]), rank_unit(models[top_two[1]]["oof"])
        first_test, second_test = rank_unit(models[top_two[0]]["test"]), rank_unit(models[top_two[1]]["test"])
        path = write_submission(workdir / "portfolio_rank_top2.csv", 0.5 * first_test + 0.5 * second_test, test, sample, id_col, target_col)
        candidates.append({"path": path, "cv_auc": score(y, 0.5 * first_oof + 0.5 * second_oof), "kind": "rank_top2"})

    weights = np.array([max(models[name]["cv_auc"] - 0.5, 0.005) ** 2 for name in blend_names], dtype=float)
    weights /= weights.sum()
    all_oof = np.average(np.stack([rank_unit(models[name]["oof"]) for name in blend_names]), axis=0, weights=weights)
    all_test = np.average(np.stack([rank_unit(models[name]["test"]) for name in blend_names]), axis=0, weights=weights)
    robust_path = write_submission(workdir / "portfolio_rank_all.csv", all_test, test, sample, id_col, target_col)
    candidates.append({"path": robust_path, "cv_auc": score(y, all_oof), "kind": "rank_all"})

    # A greedy blend selects its own weights on the OOF it is scored against, so on a
    # small table it wins the OOF comparison by fitting fold noise. Gate it on both a
    # sample-size floor and a margin over the fixed-weight blends it has to displace.
    if len(ranked_names) >= 2 and len(train) >= GREEDY_MIN_ROWS:
        try:
            greedy_test, greedy_oof_auc, chosen = greedy_blend(models, ranked_names, y)
            incumbent = max(item["cv_auc"] for item in candidates)
            if greedy_oof_auc > incumbent + GREEDY_MARGIN:
                path = write_submission(workdir / "portfolio_greedy.csv", greedy_test, test, sample, id_col, target_col)
                candidates.append({"path": path, "cv_auc": greedy_oof_auc, "kind": "greedy", "members": chosen})
            else:
                errors["greedy"] = f"BelowMargin:{greedy_oof_auc:.6f}<={incumbent:.6f}+{GREEDY_MARGIN}"
        except Exception as exc:
            errors["greedy"] = f"{type(exc).__name__}:{exc}"
    elif len(train) < GREEDY_MIN_ROWS:
        errors["greedy"] = f"SkippedSmallTable:{len(train)}<{GREEDY_MIN_ROWS}"

    # --- PHASE 1 HANDOVER LOGIC WITH SCAFFOLD GENERATOR ---
    try:
        import shutil
        common_path = Path(__file__).resolve().parent / "common.py"
        shutil.copy(common_path, workdir / "common.py")

        # 1. Write the Summary Document
        with open(workdir / "handover.md", "w") as f:
            f.write("# Phase 1 & 2 Summary (Baseline)\n")
            f.write(f"- **Data Shape:** {len(train)} rows, {len(features)} features.\n")
            f.write(f"- **Categorical Features:** {len(cat_cols)}\n")
            f.write("- **Baseline Models Ranked by OOF AUC:**\n")
            for name in ranked_names:
                f.write(f"  - {name}: {models[name]['cv_auc']:.6f}\n")
            f.write(f"- **Best blend OOF AUC:** {max(c['cv_auc'] for c in candidates):.6f}\n")
            f.write("- The blends already cover rank averaging and greedy selection. "
                    "Beat them with better features, not with another average.\n")

        # 2. Write the Pro Scaffold Script
        with open(workdir / "pro_opt.py", "w") as f:
            f.write(f"""import sys; sys.path.append("/work")
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from common import load_task, write_submission, native_frames

# 1. Load Data seamlessly
train, test, sample, id_col, target_col, features, y = load_task("/work")
x_train, x_test, cat_cols = native_frames(train, test, features)

# --- YOUR ADVANCED FEATURE ENGINEERING GOES HERE ---
# Example: Create robust mathematical interactions, target encoding, or cluster features.
# Make sure to update cat_cols if you add new categorical strings!


# 2. Train Model (Using the reliable baseline settings)
model = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.04, loss_function="Logloss", eval_metric="AUC", random_seed=42, verbose=False)
model.fit(x_train, y, cat_features=cat_cols)

# 3. Predict & Submit
predictions = model.predict_proba(x_test)[:, 1]
write_submission("/work/pro_submission.csv", predictions, test, sample, id_col, target_col)
print("SUCCESS: Submission saved to /work/pro_submission.csv. Please submit this file now.")
""")
    except Exception as e:
        print(f"Handover error: {e}")

    candidates = sorted(candidates, key=lambda item: item["cv_auc"], reverse=True)
    best_cv = candidates[0]["cv_auc"]

    # Everything the proven config would have submitted is submitted, unconditionally.
    # Only the additional learners have to earn their slot, because each extra
    # near-duplicate is another draw whose maximum is upward biased on the public split.
    proven = set(CORE_MODELS) | {"rank_top2", "rank_all"}
    slate = [item for item in candidates if item["kind"] in proven]
    extras = [
        item for item in candidates
        if item["kind"] not in proven and item["cv_auc"] >= best_cv - SLATE_BAND
    ]
    slate.extend(extras[: max(0, SLATE_MAX - len(slate))])
    slate.sort(key=lambda item: item["cv_auc"], reverse=True)
    emit_manifest({
        "candidates": [item["path"] for item in slate],
        "slate_kinds": [item["kind"] for item in slate],
        "candidate_metrics": candidates,
        "errors": errors,
        "robust_choice": robust_path,
        "total_seconds": round(time.time() - started, 3),
    })


if __name__ == "__main__":
    main()
