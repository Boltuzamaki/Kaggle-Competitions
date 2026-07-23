# V7 V1-centered recovery plan

Status: execution authorized  
Reason for reset: E33 scored 398.0498 while the aggressive V1 step-200 smoke
model scored 300.8064. The E33 pseudo-clean preservation proxy is therefore not
a trustworthy selector for the hidden clean model.

## Selection reset

- E33 is retained only as a failed historical reference.
- V1 step 200 is the frozen behavioral anchor, not a presumed optimum.
- No candidate is declared better from test predictions or per-image
  leaderboard feedback.
- Public-unlearn metrics describe suppression, stability and diversity.
- The next submission shortlist must contain behaviorally distinct,
  predeclared candidates rather than one proxy winner.

## Bundle A: fast GPU frontier

| ID | Experiment | Candidate grid | Purpose |
|---|---|---:|---|
| E28F | Fixed weight-level pruning | 0.1%, 0.5%, 1.0% | Close the only failed E00-E50 experiment after copying the reversed NumPy index array |
| E51 | Direct V1 export frontier | thresholds 0.01-0.20; NMS 0.3-0.7 | Measure the aggressive model without restoring original boxes |
| E52 | V1 score calibration | temperature and logit-bias grid | Explore confidence scale around the only leaderboard-improving family |
| E53 | V1 checkpoint model soups | step 60/120/200 weighted averages | Find smoother weight-space points without mixing back toward the poisoned model |
| E54 | V1 checkpoint output ensembles | matched-confidence averages across steps | Test inference-space diversity independently from weight soups |

Bundle A reads only the public unlearn set. It produces no test predictions and
no competition submission.

## Bundle B: counterfactual training

| ID | Experiment | Exact change |
|---|---|---|
| E55 | V1 continuation sweep | Continue score-only and classification-head updates from V1 with low learning rates and stronger negative weight |
| E56 | Counterfactual inpaint distillation | Use an inpainted poison region as the teacher view and the original poison image as the student negative view |
| E57 | Transplant-negative consistency | Move public poison crops across public backgrounds and require suppression at every transplanted location |
| E58 | Augmentation-consistent forgetting | Train across D4, scale, gain, gamma and blur variants grouped by source image |
| E59 | Tiny P3/P4 unfreeze | Unfreeze only FPN output layers P3/P4 plus the classification head after head-only training |
| E60 | V1 plus fixed E28 pruning | Apply validated weight-level masks to V1, then recover with counterfactual distillation |

## Rule 7.A boundary

- No test image is read during training, validation, calibration or model
  selection.
- No test label, pseudo-label, soft label, count or score distribution is used.
- The two existing Kaggle scores are used only to reject the E33-centered model
  family and establish V1 as a frozen coarse anchor.
- No new competition submission is created by either research bundle.

## Required outputs

- experiment coverage and candidate registry
- failure ledger
- public-unlearn prediction metrics
- V1-centered Pareto frontier
- candidate diversity matrix
- frozen shortlist with rationale
- timestamped run log and heartbeat
- plots for suppression, confidence scale and candidate diversity
