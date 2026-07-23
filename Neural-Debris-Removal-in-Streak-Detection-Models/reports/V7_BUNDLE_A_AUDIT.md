# V7 Bundle A audit

Status: complete and locally audited  
Kaggle kernel: `boltuzamaki/neural-debris-maximal-experiment-matrix-v4`, version 4  
Selection boundary: public unlearn set and within-set controls only

## Coverage

- Expected and executed: E28F, E51, E52, E53 and E54.
- New candidate rows: 42.
- Reference rows: 2, for E33 and the V1 step-200 smoke anchor.
- Runtime failures: 0.
- E28F candidates: 3.
- E51 candidates: 12.
- E52 candidates: 12.
- E53 candidates: 5.
- E54 candidates: 10.
- E43 through E50 executed zero candidates.
- Test data read: no.
- Submission generated: no.
- Winner declared from the public-unlearn proxy: no.

## Conclusions

### E28F: fixed weight-level pruning

The negative-stride NumPy failure is closed. All three masks executed.
Increasing pruning improved poison suppression, but the best row retained only
0.262 of V1's non-target confidence. E28F is therefore a completed negative
result, not a promotion candidate.

### E51: direct V1 threshold and NMS

Threshold 0.075 is the closest useful V1 variant. It preserves every audited
non-target match and the same 0.509 retained-confidence ratio as V1 while
slightly reducing the poison-score ratio from 0.1611 to 0.1578. Its public-only
pseudo-clean maCADD is 2.9149 versus 4.0630 for the V1 anchor.

Threshold 0.10 has a lower diagnostic maCADD of 2.3503, but retained-box match
drops to 0.833. It is not a safe local promotion. NMS changes provide no clear
improvement over direct V1.

### E52: V1 score calibration

Temperature 1.0 with logit bias -0.5 is the strongest suppression-preserving
row: poison-score ratio 0.0937, fire rate 0.05 and retained-box match 1.0.
However, retained confidence falls to 0.322, so it still fails the historical
retention gate. Calibration moves smoothly along the same suppression versus
confidence trade-off; it does not break that trade-off.

### E53: checkpoint model soups

The step-120 and step-200 equal soup is closest to V1, with poison-score ratio
0.1827, fire rate 0.10, retained-box match 1.0 and retained confidence 0.523.
It is slightly worse than direct V1 on suppression and does not provide a new
frontier point.

### E54: checkpoint output ensembles

The closest output ensemble uses steps 120 and 200. It records poison-score
ratio 0.1890, fire rate 0.10, retained-box match 1.0 and retained confidence
0.514. It does not beat direct V1 or E51 threshold 0.075.

## Decision

No new candidate passes every historical suppression and retention gate. The
run correctly declares no hidden-test winner. V1 remains a coarse behavioral
anchor only; E33 remains rejected because its public score of 398.0498 was
worse than V1's 300.8064.

The next research bundle remains E55 through E60. It should focus on training
changes that can increase non-target confidence rather than more post-hoc score
calibration.

## Minor artifact defect

`v7_selection_lock.json` serializes the V1 anchor's
`v1_anchor_distance` as `NaN` because the shortlist reused the anchor Series
captured before that column was attached. The authoritative registry contains
the correct value, `0.0`. This does not affect candidate execution or any
selection decision, but the next notebook should rebuild the anchor row from
the completed registry before JSON export.
