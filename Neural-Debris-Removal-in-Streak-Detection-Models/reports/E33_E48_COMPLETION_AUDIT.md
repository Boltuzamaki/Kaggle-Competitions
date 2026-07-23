# E33 inference and E43-E48 completion audit

Kaggle runs:

- `boltuzamaki/neural-debris-e33-validated-inference`, version 1
- `boltuzamaki/neural-debris-maximal-experiment-matrix-v4`, version 2

Audit mode: local artifact validation  
Competition submission created: no

## Decision

E33 remains the best validated pipeline.

The E43-E48 run evaluated 100 new candidates plus the frozen E33 reference.
None of the 100 new candidates passed all four suppression and retention gates.
E33 was the only passing row and retained its public-only pseudo-clean maCADD
of `4.869691`.

## E33 public reproduction

| Metric | Reproduced | Gate |
|---|---:|---:|
| Poison score ratio | 0.199509 | <= 0.25 |
| Poison fire rate at 0.20 | 0.20 | <= 0.35 |
| Retained box match | 1.00 | >= 0.90 |
| Retained confidence ratio | 0.823216 | 0.80-1.20 |
| Public-only pseudo-clean maCADD | 4.869691 | ranking metric |
| Proxy | 0.248143 | lower is better |

The public reproduction matched the frozen V4 metrics within the predeclared
`5e-4` tolerance.

## E33 test artifact validation

- Rows: 2,000
- Unique image IDs: 2,000
- Detections: 4,053
- Nonempty rows: 1,669
- Empty rows: 331
- Confidence range: 0.050017 to 0.992927
- Required schema: valid
- Box bounds and dimensions: valid
- `submission.csv` and `submission_e33.csv`: byte-identical
- SHA-256: `600d3cc2f3740416ad3e1fa75ce520abd365ef339486867c996f152c577b191a`

The fixed test pipeline marked an average of 19.59 of the original 100
detections per image as suspicious. This diagnostic was not used for selection.

## E43-E48 coverage

| Experiment | Candidates | Passing | Closest-gate result |
|---|---:|---:|---|
| E43 dense original box bank | 4 | 0 | Preserves retention but leaves poison ratio/fire at 1.0/1.0 |
| E44 unmatched-aware blending | 18 | 0 | Ratio 0.7483, fire 1.0, retention confidence 0.8528 |
| E45 continuous drop | 20 | 0 | Ratio 0.4128, fire 0.75, retention confidence 0.7132 |
| E46 early-checkpoint blending | 24 | 0 | Ratio 0.8322, fire 1.0, retention confidence 0.9019 |
| E47 weight interpolation | 18 | 0 | Ratio 0.6578, fire 0.95, retention confidence 0.8058 |
| E48 tiny affine head | 16 | 0 | Ratio 0.7583, fire 1.0, retention confidence 0.8637 |

E45 moved farthest toward poison suppression, but it still missed both poison
gates and reduced retained confidence below the allowed minimum.

## Rule and execution audit

- E49 and E50 remained disabled.
- No test data was read by the E43-E48 research run.
- No test-derived training or selection occurred.
- No leaderboard result was used.
- No experiment branch failed at runtime.
- No competition submission was created.

## Recommendation

Keep E33 as the frozen submission candidate. Do not replace it with an
E43-E48 candidate. The validated artifact is:

`forensics/kaggle_e33_validated_inference/output_v1/submission_e33.csv`

Submitting it remains a separate user-authorized action.
