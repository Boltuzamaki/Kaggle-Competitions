# Neural Debris clean/poison twin surrogate benchmark

## Objective

Create miniature versions of the competition in which the clean reference
RetinaNet is known but hidden from each repair method.  Candidate selection is
based on paired maCADD against that clean twin, not poison-versus-proxy AUC.

This benchmark is a method-ranking validator.  It is not an estimator of the
private Kaggle score.

## Stages

### Stage 1: clean twin and immutable episode pack

1. Convert the public StreaksYolo dataset to Detectron2 records.
2. Train one single-class RetinaNet clean twin from a fixed COCO initialization.
3. Require held-out streak recall and confidence gates before promotion.
4. Freeze six deterministic poisoning manifests.
5. Generate clean-twin reference predictions for every episode evaluation set.
6. Export the clean checkpoint, manifests, predictions, quality audit, and
   provenance hashes.

### Stage 2: poisoned twins and repairs

Each episode starts from the exact clean checkpoint.  Its poisoned twin is
fine-tuned on the same retained clean records plus positive annotations for one
artificial poison family.  A repair sees only the poisoned checkpoint, twenty
revealed poison examples, and the predeclared retained/synthetic resources.

Repairs evaluated per episode:

- poisoned model without repair;
- empty-label head repair, direct inference;
- NDR-style confidence-collapse rescoring on the poisoned box bank;
- synthetic-positive retention/KD repair and tiered rescoring;
- PCGrad retained-data repair and narrow veto;
- expanded-bank union control;
- broad confidence-recovery control;
- current strict V15/V21/V23-style consensus when its required signals exist.

### Stage 3: surrogate acceptance

For every repair and episode, compute the supplied local maCADD implementation
against the clean-twin predictions.  Also record Hungarian and greedy matching
audits when they disagree.

A surrogate family is promotable only when:

- the poisoned twin is measurably worse than the clean twin;
- the revealed poison attack fires on at least 80% of held-out poison inserts;
- retained clean prediction mass remains at least 80% of the clean twin;
- at least three nontrivial repair methods produce distinct outputs;
- its repair ranking broadly matches real evidence: narrow fixed-bank vetoes
  outperform broad recovery and expanded candidate banks;
- leave-one-method-out historical-order correlation remains positive.

Candidate submission gate across accepted episodes:

- median paired delta maCADD versus V15-style incumbent is below zero;
- win rate is at least 80%;
- lower quartile paired delta is below zero;
- paired bootstrap 90% interval does not show a material regression;
- exact candidate bank, coordinates, no added boxes, and no confidence increase
  for the first competition probe.

## Frozen poison episodes

1. Solid hard-endpoint line.
2. Periodic dashed line.
3. Alpha-composited line with inconsistent foreground noise.
4. PSF-mismatched line with nonphysical transverse side lobes.
5. Quantized/resampled line with pixel-grid periodicity.
6. Constant-width, constant-intensity procedural streak.

Each episode uses disjoint training and evaluation source images.  Injection
parameters and seeds are frozen in Stage 1.  No competition test image is used
to design, train, validate, or select any surrogate component.

## Reproducibility outputs

- `clean_model.pth`
- `clean_training_history.csv`
- `clean_quality.json`
- `surrogate_manifest.json`
- `clean_reference_episode_*.csv`
- `episode_preview.png`
- `stage1_report.json`
- Stage 2 per-episode checkpoints, predictions, maCADD tables, paired deltas,
  bootstrap intervals, ranking correlations, failures, and final selection lock

