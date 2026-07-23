# E49-E50 completion audit

Status: complete and locally validated  
Kaggle kernel: `boltuzamaki/neural-debris-maximal-experiment-matrix-v4`, version 3  
Selection scope: public unlearn images and within-set controls only  
Competition submission generated: no

## Coverage

- E49 conflict-projected unlearning: 6 grouped-CV candidates completed.
- E50 retain-budget projected repair: 6 grouped-CV candidates completed.
- Total grouped-CV candidates: 12.
- Training-history rows: 2,130.
- Runtime failures: 0.
- E43-E48 were recorded as completed and were not rerun.
- Test data was not read.
- Leaderboard results were not used for selection.

## Frozen incumbent reproduction

E33 reproduced within the locked tolerance:

- poison score ratio median: `0.19950863887810824`
- poison fire rate at 0.20: `0.20`
- retained box match rate: `1.0`
- retained confidence ratio median: `0.823215901851654`
- public-only pseudo-clean maCADD: `4.869690555334092`

## New experiment results

No E49 or E50 grouped-CV candidate passed all frozen gates.

Best E49 grouped-CV candidate:

- candidate: `e49_head_lr3e6_fw001_s30`
- poison score ratio median: `0.975993`
- poison fire rate: `1.0`
- retained match rate: `1.0`
- retained confidence ratio: `0.981036`
- public-only pseudo-clean maCADD: `41.147668`

Best E50 grouped-CV candidate:

- candidate: `e50_head_lr3e6_fw001_b5e4`
- poison score ratio median: `0.975890`
- poison fire rate: `1.0`
- retained match rate: `1.0`
- retained confidence ratio: `0.980756`
- public-only pseudo-clean maCADD: `41.144568`

The selected E50 specification was retrained on all 20 public-unlearn images:

- candidate: `e50_head_lr3e6_fw001_b5e4_all20`
- poison score ratio median: `0.9730856256725052`
- poison fire rate: `1.0`
- retained match rate: `1.0`
- retained confidence ratio: `0.9819687008857728`
- public-only pseudo-clean maCADD: `41.02298635076731`
- passes all gates: no

## Interpretation

Gradient conflict was observed in 1,330 of 2,130 logged steps, confirming that
the forget and retain objectives frequently opposed each other. Projection
preserved the detector's non-target behavior, but it also made the update too
conservative to suppress the poisoned streak response. The best full candidate
kept every retained box while still firing on every poison image.

## Decision

E33 remains the validated winner. The projected E49/E50 checkpoint must not
replace it because it fails both poison suppression gates and has a much worse
public-only pseudo-clean maCADD.

Checkpoint audit SHA-256:
`4d042c2e8f9d4dae556b4bb9d381c49fd2fdf02a067ebf8594c0aa20117d6383`
