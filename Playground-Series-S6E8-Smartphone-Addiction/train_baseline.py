"""Train a reproducible CatBoost baseline and create submission.csv."""

from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


SEED = 42
TARGET = "addicted_label"
ID = "id"


def prepare(frame: pd.DataFrame, features: list[str], categorical: list[str]) -> pd.DataFrame:
    x = frame[features].copy()
    # CatBoost requires categorical missing values to be represented explicitly.
    for col in categorical:
        x[col] = x[col].fillna("Missing").astype(str)
    return x


def main() -> None:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("sample_submission.csv")

    features = [c for c in test.columns if c != ID]
    categorical = [
        c for c in features if c in train.select_dtypes(exclude="number").columns
    ]
    x = prepare(train, features, categorical)
    x_test = prepare(test, features, categorical)
    y = train[TARGET]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.15, stratify=y, random_state=SEED
    )
    model = CatBoostClassifier(
        iterations=1_200,
        depth=8,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5,
        random_seed=SEED,
        thread_count=-1,
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(
        x_train,
        y_train,
        cat_features=categorical,
        eval_set=(x_valid, y_valid),
        early_stopping_rounds=100,
    )

    valid_prediction = model.predict_proba(x_valid)[:, 1]
    auc = roc_auc_score(y_valid, valid_prediction)
    print(f"Validation ROC AUC: {auc:.6f}")
    print(f"Best iteration: {model.get_best_iteration()}")

    sample[TARGET] = model.predict_proba(x_test)[:, 1]
    sample.to_csv("submission.csv", index=False)
    Path("artifacts").mkdir(exist_ok=True)
    model.save_model("artifacts/catboost_baseline.cbm")
    pd.DataFrame(
        model.get_feature_importance(prettified=True)
    ).to_csv("artifacts/feature_importance.csv", index=False)
    Path("artifacts/validation.txt").write_text(
        f"metric=roc_auc\nscore={auc:.8f}\nbest_iteration={model.get_best_iteration()}\n"
    )
    print(f"Wrote submission.csv with {len(sample):,} rows")


if __name__ == "__main__":
    main()
