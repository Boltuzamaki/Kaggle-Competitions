"""Write a fast valid baseline before the more expensive portfolio stage."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_task, native_frames, runtime_workdir, write_submission


def main() -> None:
    workdir = runtime_workdir()
    output = workdir / "quick_baseline.csv"
    train, test, sample, id_col, target_col, features, y = load_task(workdir)
    prior = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
    write_submission(output, np.full(len(test), prior), test, sample, id_col, target_col)
    method = "prior"
    try:
        from catboost import CatBoostClassifier

        x_train, x_test, cat_cols = native_frames(train, test, features)
        iterations = 260 if len(train) < 3000 else 360
        depth = 5 if len(train) < 2000 else 6
        model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=0.06,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=6.0,
            random_strength=0.5,
            random_seed=20260710,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
        )
        model.fit(x_train, y, cat_features=cat_cols)
        predictions = model.predict_proba(x_test)[:, 1]
        if np.std(predictions) > 1e-8:
            write_submission(output, predictions, test, sample, id_col, target_col)
            method = "catboost"
    except Exception as exc:
        print(f"quick_model_error={type(exc).__name__}:{exc}")
    print(f"QUICK_BASELINE path={output} method={method} rows={len(test)} features={len(features)}")


if __name__ == "__main__":
    main()