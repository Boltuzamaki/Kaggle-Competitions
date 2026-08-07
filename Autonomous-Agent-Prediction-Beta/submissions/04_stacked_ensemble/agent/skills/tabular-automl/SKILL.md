---
name: tabular-automl
description: Runs a time-aware CPU portfolio with engineered features, then combines models by stacking and bagged hill-climb ensemble selection.
---

# Tabular AutoML

Use this skill exactly once at the beginning of the modeling phase.

Run the script with this exact tool call:

`run_skill_script(skill_name="tabular-automl", file_path="scripts/run_automl.py", args=[])`

It reads `train.csv`, `test.csv`, and `sample_submission.csv` from the working directory.
It engineers label-free frequency and missingness features, performs stratified (repeated on
small tables) cross-validation over boosting, tree, linear, and neural models, then combines
them with a level-2 logistic stacker, bagged hill-climb ensemble selection, and rank blends,
and writes candidate files.

The final stdout line starts with `RESULTS_JSON=`. Submit every path in `recommended_files`.
Use `robust_file` as the validation-based hedge when selecting the final two submissions.

Do not modify the script, do not run candidates one at a time, and do not rerun it.
