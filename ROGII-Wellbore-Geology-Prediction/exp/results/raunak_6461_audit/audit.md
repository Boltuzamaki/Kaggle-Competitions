# Raunak 6.461 notebook audit

## Verdict

The 6.461 output is not a clean new model score. Its final improvement is the
hard-coded `A27` post-processing branch applied on top of a submission explicitly
labelled `6.594 source submission`.

`A27` is excluded from our pipeline because it:

- requires an exact source-prediction SHA;
- requires exactly 14,151 rows;
- requires exactly one detected branch well;
- requires that well to be `00e12e8b`;
- requires exactly 4,301 rows for that well;
- changes no other well;
- was tuned as a 10% residual restoration from a known 6.594 submission.

The run log confirms all 4,301 rows of `00e12e8b` changed, with a maximum move
of 0.3578 ft. This is test-specific post-processing, not a generally validated
model.

## What produced the 6.594 source

The source trajectory was built by:

1. fresh SP45/PF/beam inference;
2. fresh inference from three mounted Fleongg LightGBM models;
3. a 60% SP45 / 40% learned blend;
4. same-well contact reconstruction for all three test IDs using their matching
   train wells;
5. visible-prefix calibration, which made no effective balanced-profile move;
6. Pilkwang model-package correction, which was automatically disabled because
   its p95 disagreement was 26.701 ft;
7. a generic PF seed-branch hedge applied to one well.

The same-well contact layer overwrote all 14,151 test predictions:

- `000d7d20`: 3,836/3,836 rows;
- `00bbac68`: 6,014/6,014 rows;
- `00e12e8b`: 4,301/4,301 rows.

Consequently the public result is dominated by the current test/train ID
overlap, not by the advertised stacked tabular model.

## Attached-source classification

- `fleongg/rogii-claude-models-pub`: legal fresh inference. It contains only
  `features.json` and three LightGBM pickle files; no prediction CSV. Models were
  downloaded to `references/fleongg_models/`.
- `ravaghi/wellbore-geology-prediction-artifacts`: fresh ridge/PF anchor, already
  represented in our audited V4/SP45 families.
- `pilkwang/rogii-model-package`: legal fresh inference with supplied OOF, already
  audited and included in our meta experiments. It was disabled in this run.
- `thbdh5765/rogii-v10-fresh-artifacts` and TabICL mirror: attached but not used
  by the active submission path; their OOF legs are already audited locally.
- `yuki16/rogii-model-package` and `nina2025/rogii-03`: attached but not selected
  by any active explicit root in this profile.
- Fleongg precomputed-submission fallback: prohibited in principle, but not
  exercised here because the mounted Fleongg dataset has models and no CSV.

## Reusability decision

The Fleongg boosters are the only novel legal mounted component. Their source
code reports about 9.21 grouped CV, but the dataset supplies all-train models
only—no fold models or OOF predictions. Therefore they cannot be admitted as an
OOF-valid ensemble leg without rebuilding the expensive 196-feature PF/beam/
spatial feature pipeline and retraining complete-well folds.

Given their reported CV and high overlap with our existing Harshini/V4/SP45
features, that rebuild is not justified as a bounded experiment. No component
from this notebook is promoted.

## Evidence files

- `references/raunak_6461/rogii-stacked-ensemble.ipynb`
- `references/raunak_6461/reports/rogii-stacked-ensemble.log`
- `references/raunak_6461/reports/gold_prefix_submission_audit.json`
- `references/raunak_6461/reports/submission_audit.json`
- `references/fleongg_models/features.json`
