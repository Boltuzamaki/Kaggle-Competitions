---
name: tabular-automl
description: Runs a size-adaptive time-aware CPU portfolio: robust rank blends on small tables, plus a hill-climb and logistic stacker on mid/large ones.
---

# Tabular AutoML

Use this skill exactly once at the beginning of the modeling phase.

Run the script with this exact tool call:

`run_skill_script(skill_name="tabular-automl", file_path="scripts/run_automl.py", args=[])`

It reads `train.csv`, `test.csv`, and
`sample_submission.csv` from the working directory. It performs stratified cross-validation,
trains multiple shallow, balanced, and deep boosting configurations plus diverse tree and linear
models, and builds out-of-fold-selected rank blends. On tables of at least a few thousand rows it
additionally fits a bagged hill-climb ensemble and a logistic stacker, which win at that scale but
would overfit the noisier cross-validation on small tables. It then writes candidate files.

The final stdout line starts with `RESULTS_JSON=`. Submit every path in `recommended_files`.
Use `robust_file` as the validation-based hedge when selecting the final two submissions.

Do not modify the script, do not run candidates one at a time, and do not rerun it.
