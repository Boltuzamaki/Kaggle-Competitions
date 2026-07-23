# V4 maximal experiment matrix - completion audit

Run: `boltuzamaki/neural-debris-maximal-experiment-matrix-v4`  
Kaggle status: complete  
Audit mode: read-only  
Submission action: none

## Executive decision

Do **not** submit any of the four CSVs exported by V4.

The notebook completed 42 of 43 experiment IDs and evaluated 210 candidate
rows. Eight rows passed the locked suppression/retention gate. However, the
notebook's final model-only selector ignored those passing output pipelines and
selected the empty-label model, which fails the retention gate.

The best valid V4 pipeline is:

`E33 e33_threshold_0.05`

Its inference recipe is:

1. Start from the original model's boxes and scores.
2. Use V1 step 200 only as an indicator.
3. When `V1 score / original score <= 0.35`, multiply the original confidence
   by `0.25`.
4. Apply logit temperature `0.85`.
5. Export detections at confidence threshold `0.05`.

## Winning public-only metrics

| Metric | Result | Gate |
|---|---:|---:|
| Poison score ratio, median | 0.199509 | <= 0.25 |
| Poison fire rate at 0.20 | 0.20 | <= 0.35 |
| Retained box match rate | 1.00 | >= 0.90 |
| Retained confidence ratio, median | 0.823216 | 0.80-1.20 |
| Public-only pseudo-clean maCADD | 4.869691 | ranking metric after gate |
| Frozen proxy | 0.248143 | lower is better |

The maCADD value is the exact public formula evaluated against the public-only
pseudo-clean reference. It is not the hidden leaderboard score.

## Why the notebook selected the wrong artifact

Two implementation issues affected the final export:

1. E36 restricted final selection to candidates with entries in `MODEL_PATHS`.
   The gate-passing E04/E32/E33/E34 candidates are output pipelines, not
   standalone checkpoints, so they were excluded.
2. The notebook correctly computed `local_pseudo_clean_macadd` before E36 and
   preserved it in `pareto_front.csv`, but later rebuilt
   `experiment_registry.csv` from the raw registry and overwrote those values
   with missing entries.

The model-only pool contained no gate-passing checkpoint. Because the selector
still required a model, it chose `official_empty_lr1e4_epoch20_batch4`.

That checkpoint has:

- poison score ratio `0.0`;
- poison fire rate `0.0`;
- retained box match rate `0.0`;
- retained confidence ratio `0.0`;
- hard-gate result: **failed**.

## Exported finalist audit

| Rank | Candidate | Gate | Empty test rows | Decision |
|---:|---|---|---:|---|
| 1 | official_empty_lr1e4_epoch20_batch4 | Failed | 2,000 / 2,000 | Reject |
| 2 | e38_multilayer_15pct | Failed | 1,966 / 2,000 | Reject |
| 3 | e38_multilayer_10pct | Failed | 1,881 / 2,000 | Reject |
| 4 | e38_multilayer_5pct | Failed | 1,157 / 2,000 | Reject |

No submission for the passing E33 pipeline was generated.

## Coverage

- Experiment IDs present: 43 / 43.
- Completed experiment IDs: 42.
- Attempted but failed: E28.
- Candidate rows: 210.
- Gate-passing rows: 8.

E28 failed for all three weight-pruning fractions because a NumPy array with
negative strides was passed to PyTorch. The local repair is to make the reversed
index array contiguous with `.copy()` before tensor conversion. This failure
does not change the winner because no E28 candidate was evaluated.

## Best passing Pareto rows

| Candidate | Gate | Pseudo-clean maCADD | Proxy |
|---|---|---:|---:|
| e33_threshold_0.05 | Pass | 4.869691 | 0.248143 |
| e32_affine_0.9_0 | Pass | 27.309838 | 0.251340 |
| v1_gate_ratio0.35 | Pass | 30.321004 | 0.250000 |
| e32_temperature_1 | Pass | 30.321004 | 0.250000 |

Other passing rows were dominated and therefore absent from the frozen Pareto
table.

## Recommendation

Keep `e33_threshold_0.05` as the V4 reference pipeline, but do not use the
exported V4 submissions. The next authorized Kaggle run should generate test
predictions directly from this output pipeline and compare it locally with the
prepared V5 E43-E50 metric-preservation candidates. No Kaggle upload or
submission should occur until the user explicitly unlocks it.

