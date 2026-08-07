---
name: tabular-automl
description: Runs a broad time-aware CPU portfolio across multiple boosting parameter regions, seeds, and blend weights.
---

# Tabular AutoML

Use this skill exactly once at the beginning of the modeling phase.

Run the script with this exact tool call:

`run_skill_script(skill_name="tabular-automl", file_path="scripts/run_automl.py", args=[])`

It reads `train.csv`, `test.csv`, and
`sample_submission.csv` from the working directory. It performs stratified cross-validation,
trains multiple shallow, balanced, and deep boosting configurations plus diverse tree and linear
models, builds out-of-fold-selected equal and weighted rank blends, and writes candidate files.

The final stdout line starts with `RESULTS_JSON=`. Submit every path in `recommended_files`.
Use `robust_file` as the validation-based hedge when selecting the final two submissions.

Do not modify the script, do not run candidates one at a time, and do not rerun it.
