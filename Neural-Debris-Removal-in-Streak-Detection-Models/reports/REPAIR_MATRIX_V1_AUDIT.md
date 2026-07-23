# Repair matrix v1 audit

Date: 2026-07-17

## Outcome

The Kaggle GPU kernel completed all six candidates across five source-grouped
folds and trained three final checkpoints for the selected
`full_cls_lr3e5` candidate.

No candidate passed the predeclared joint suppression/retention gate. The
checkpoint must not be promoted as the repaired competition model.

## Candidate selection

`full_cls_lr3e5` was the strongest cross-validation candidate:

- median poison score ratio: `0.3033`
- poison fire rate at 0.20: `0.35`
- retained-score ratio: `0.4907`
- predeclared gate: failed

The aggregate match rate of `0.60` is artificially low because two validation
folds contained zero original non-target detections and were recorded as zero
instead of missing. This accounting issue should be corrected in the next
matrix. It does not change the safety decision because the positive
retained-score ratio is still far below the required `0.80`.

## Final checkpoint comparison

| Step | Poison score ratio | Poison fire rate | Retain match | Retained-score ratio | Decision |
|---:|---:|---:|---:|---:|---|
| 60 | 0.2570 | 0.30 | 1.00 | 0.4314 | Reject |
| 120 | 0.2064 | 0.10 | 1.00 | 0.5168 | Reject |
| 200 | 0.1611 | 0.10 | 1.00 | 0.5095 | Reject |

The 200-step checkpoint suppresses the poison response most strongly, but the
roughly 49% reduction in retained confidence indicates broad classification
damage. The 120-step checkpoint is similarly unsafe.

## Next repair direction

Do not increase unlearning strength. The next experiment should improve
selectivity:

1. correct zero-retention-fold accounting;
2. expand dense teacher consistency on non-poison anchors;
3. test much stronger retention weights and shorter schedules;
4. evaluate a bias-frozen or low-rank last-layer correction so the poison
   feature direction is suppressed without globally lowering classification
   confidence;
5. retain the same grouped folds and predeclared gate.

## Rule guard

The completed kernel did not read competition test images or test predictions.
It used only the supplied poisoned model and the 20 public unlearn images.
