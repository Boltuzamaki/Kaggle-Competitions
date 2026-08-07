# Engineered Diversity v5

This profile is an additive challenger to the completed 0.818 agents. It preserves the proven
native/ordinal model family and adds candidates that can produce materially different rankings.

## What changed

- Four-regime detection: small, categorical, numeric, or mixed.
- Extra conservative and second-seed CatBoost candidates on small datasets.
- A separate feature-engineered LightGBM branch with row summaries, safe log/date features,
  frequency encoding, and cross-fitted target encoding for high-cardinality categoricals.
- Stable mixed-dtype row hashing and duplicate-aware CV for the engineered branch.
- RandomForest small-sample expert.
- Correlation-aware and CatBoost-consensus rank blends.
- Exact legacy v4 greedy, diverse, and weighted blends remain in the candidate slate.
- Public winner plus OOF-best blend are selected as complementary final submissions.

## Evidence

- The July 19 public AgentForge v17 update was inspected but not reused: its row-key function
  failed on three real practice tasks with mixed missing values. This profile uses
  `pandas.util.hash_pandas_object` instead.
- On practice task 13 (500 rows), the new public-winning conservative candidate reached
  0.66217 private AUC, compared with 0.65461 for the prior broad profile.
- Representative tests also covered 1,060-row, all-categorical, 28,879-row numeric, and
  500-row high-categorical regimes.
- The archive uses no external training data, labels, model weights, or binaries.

The final agent archive is `submission.zip`.
