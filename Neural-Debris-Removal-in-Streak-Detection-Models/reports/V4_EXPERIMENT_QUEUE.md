# Neural Debris V4 experiment queue

Source: user-supplied E00-E37 matrix plus audited E38-E42 extensions  
Execution target: one resumable Kaggle GPU notebook  
Selection data: public unlearn set and synthetic controls only  
Test use: frozen finalist inference only, after selection

## Non-negotiable promotion gate

A model candidate passes only when all conditions hold on grouped public-unlearn
validation:

- poison fire rate at 0.20 `<= 0.35`
- median poison score ratio `<= 0.25`
- retained non-target match rate `>= 0.90`
- median retained confidence ratio in `[0.80, 1.20]`

Detection-level gates additionally require grouped source-image validation and
must improve the suppression/retention Pareto frontier. No test image,
prediction, count, score distribution or leaderboard result can select a
candidate.

The exact public maCADD implementation from
`nbridelancetb/macadd-local-scoring-esa-neural-debris-removal` is also used
against a **pseudo-clean public reference**: original public-unlearn predictions
with only detections overlapping the organizer-annotated poison boxes removed.
This is not the hidden leaderboard metric, but it jointly penalizes residual
poison false positives, lost non-target detections, confidence drift and box
movement. It is used after the hard promotion gate, never as a replacement for
that gate.

## Execution blocks

| Block | Experiments | Implementation |
|---|---|---|
| A - reproduce and import | E00-E02 | Re-evaluate original and frozen V1/V2/V3 checkpoints; run the empty-label baseline |
| B - output and detection gates | E03-E11 | Matched-confidence blends, soft gating, ROI/morphology/stability features, grouped logistic/MLP gates |
| C - positive-control repair | E12-E16 | Synthetic streak controls, balanced gate training, KD sweep, residual adapter and low-rank repair |
| D - pruning and recovery | E17-E31 | Activation/selectivity/causal/paired/level-specific/weight masks, RNP/ANP/movement variants, distillation and synthetic recovery |
| E - calibration and selection | E32-E37 | Frozen calibration, threshold/NMS sweeps, robustness suite, Pareto selection and finalist export |
| F - gradient-ascent/EWC extensions | E38-E42 | Corrected multi-layer activation pruning, classification-only gradient ascent, explicit EWC/LR sweeps and guarded recovery |

## Additional experiments from the audited public kernel

The public kernel
`dataarthur/grad-ascent-pruning-ewc-backbone-freeze` suggested several useful
directions, but its hook accounting, annotation description and EWC parameter
handling are not copied directly.

| ID | Corrected experiment | Key guard |
|---|---|---|
| E38 | Per-layer activation-selectivity pruning | Pool every P3/P4 call separately; prune complete tower filters and downstream inputs |
| E39 | Classification-only gradient ascent + backbone freeze | Box regression, FPN and backbone remain frozen |
| E40 | Explicit EWC strength sweep | EWC coefficient is applied directly in the objective and logged |
| E41 | Long low-learning-rate sweep | Fixed update budgets; no leaderboard-derived choice |
| E42 | Multi-layer prune + EWC/KD recovery | Must beat both unpruned EWC and static pruning under the same gate |

## Rule 7.A audit

- E03-E06 and E31-E34 are tuned only on public-unlearn proxies, then frozen.
- E07-E11 use poison annotations and within-public-set controls only.
- E12 synthetic controls are generated from public-unlearn backgrounds without
  external models or test images.
- E13-E30 never read the competition test directory.
- E32-E36 run before test inference.
- E37 exports at most four predeclared finalist CSVs. Kaggle submission remains
  external and cannot feed back into training or selection.

## Outputs

- `experiment_registry.csv`: one row per evaluated configuration
- `grouped_gate_cv.csv`: E08-E13 grouped validation
- `training_history.csv`: E01 and E14-E30 optimization logs
- `pareto_front.csv`: nondominated safe/near-safe finalists
- `best_model.pth`: frozen best eligible model, if one passes
- `selection_lock.json`: evidence used for selection and Rule 7.A guard
- `submission_best.csv`: test predictions from the frozen selection
- `submission_finalist_*.csv`: up to three additional predeclared finalists
- plots, heartbeat log and final report
