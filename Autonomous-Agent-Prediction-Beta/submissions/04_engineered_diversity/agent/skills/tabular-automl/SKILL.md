---
name: tabular-automl
description: Runs duplicate-aware native and engineered model branches with seed and correlation diversity.
---

# Tabular AutoML

Use this skill exactly once at the beginning of the modeling phase.

Run the script with this exact tool call:

`run_skill_script(skill_name="tabular-automl", file_path="scripts/run_automl.py", args=[])`

It reads `train.csv`, `test.csv`, and `sample_submission.csv` from the working directory. It
routes small, categorical, numeric, and mixed regimes; uses duplicate-aware stratified validation;
combines native CatBoost with frequency/cross-fitted-target-encoded boosters and linear/bagged
experts; builds equal, weighted, seed-consensus, and correlation-aware rank blends; stabilizes
identical test rows; and writes a compact candidate slate.

The final stdout line starts with `RESULTS_JSON=`. Submit every path in `recommended_files`.
Use `robust_file` as the validation-based hedge when selecting the final two submissions.

Do not modify the script, do not run candidates one at a time, and do not rerun it.
