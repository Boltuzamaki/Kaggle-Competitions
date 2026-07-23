# Public Kaggle research audit - 2026-07-18

## Outcome

The strongest verified public notebook found is:

- [NDR_trial1](https://www.kaggle.com/code/sanidhyavijay24/ndr-trial1)
- Displayed Kaggle score: **229.2314** (lower is better)
- Current verified project incumbent: **V1 = 300.8064**
- E33 is not competitive: **398.0498**

No Kaggle kernel was pushed and no competition submission was created during this research pass.

## Host clarifications that change the legal search space

The competition host explicitly permits:

1. Raw test pixels to participate in the de-poisoning process.
2. Predictions produced by the provided poisoned model to be used during training.
3. Comparing predictions from the poisoned and de-poisoned models.
4. Synthetic streak generation, including insertion into test images, provided the simulator is not designed from manual or automatic inspection/annotation of test streaks.
5. Multiple internally trained models, ensembles and deterministic post-processing.

The prohibited boundary remains manual or automatic test labels produced by people, external models, or assumptions derived from inspection of test streaks.

Relevant host discussions:

- [Inquiry on the usage of Test Set](https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models/discussion/694526)
- [Synthetic clean proxy clarification](https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models/discussion/708277)
- [No Other Models clarification](https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models/discussion/708394)
- [Metric adjustment](https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models/discussion/699266)

## Exact NDR_trial1 recipe

Training:

- RetinaNet R50 FPN with the competition anchor configuration.
- Activation scores from poison-box ROI activation minus random background activation.
- Nominal pruning fraction: 15%.
- Classifier-only empty-label fine-tuning.
- Learning rate: `2.5e-4`.
- Iterations: `20`.
- Batch size: `4`.
- L2/EWC anchor to the post-pruning classifier weights.
- EWC coefficient: `500`.
- No test labels and no external model.

Inference:

- Generate high-recall candidates from the original poisoned model at `0.05`.
- Generate comparison predictions from the repaired model at `0.05`.
- Match boxes at IoU `0.5`.
- Compute confidence-collapse signal:
  `s_diff = 1 - repaired_confidence / original_confidence`.
- Compute poison-box geometry similarity from the 20 supplied poison boxes.
- Combine:
  `p_poison = 0.90 * s_diff + 0.10 * s_geo`.
- Drop original candidates below `0.20`.
- Keep candidates when `p_poison <= 0.25`.
- Demote candidates to confidence `0.01` when `p_poison >= 0.55`.
- Interpolate confidence between those poison-probability thresholds.

## Critical implementation quirk

The notebook claims to prune all four classification-subnet convolutions, but its layer indexes do not line up:

- Activation collection indexes the four convolution layers as `0, 1, 2, 3`.
- Pruning indexes the full sequential module as `0, 2, 4, 6`.
- Only index keys `0` and `2` match.
- The scored run therefore prunes only two convolution layers, and the second matched layer receives a score vector collected from a different convolution depth.

The Kaggle run log confirms only two layers were pruned, 38 channels each. An exact reproduction must preserve this behavior first. A corrected four-layer version is a separate ablation, not a bug fix to the anchor.

## Other public notebooks

| Notebook | Displayed score | Audit conclusion |
|---|---:|---|
| NDR_trial1 | **229.2314** | Best verified public anchor. Exact reproduction should be next. |
| Roadmap that got me to 226 | title claim only | Useful ideas, but not a reproducible scored 226 implementation. |
| Pruning + EWC | 243.4219 | Strong legal anchor. Its actual trainer likely fine-tunes more of the model than the text claims. |
| CV de-poisoning experiments | 245.3014 | Useful LR/threshold sweep. Provided-model test predictions are permitted by host clarification. |
| Calibrated rescoring v3 | 250.476 | Useful differential-rescoring idea, but it selects a final variant using a self-constructed test surrogate. Avoid that selection step. |
| Pseudo-retain gradient ascent | 250.2742 | Provided-poisoned-model pseudo-retain predictions are allowed, but the method is weaker than NDR_trial1. |
| Distillation-guided scrubbing | 294.89 | Worse than V1 and distills only poison-image FPN features; low priority. |

## Recommended next experiment bundle

Stage A - exact score-bearing anchors:

1. `E55_NDR229_EXACT`: bug-faithful NDR_trial1.
2. `E56_NDR229_SEED_REPLICA`: exact recipe with a second deterministic activation-background seed.
3. `E57_JAYHAWK243_EXACT`: reproduce the actual full-model training behavior, not only the notebook description.

Stage B - narrow NDR neighborhood:

4. EWC coefficient: `250`, `500`, `1000`.
5. Pruning fraction: `0.10`, `0.15`.
6. Fine-tuning: `(lr=2e-4, 20)` and `(lr=2.5e-4, 15/20)`.
7. Bug-faithful two-layer pruning versus correctly aligned four-layer pruning.

Stage C - likely improvement beyond 229:

8. Average the per-candidate survival signal across NDR replicas and the 243 anchor.
9. Keep original poisoned-model boxes to preserve localization; only alter confidence.
10. Test an unlearn-derived post-processing grid around:
    - `P_HI = 0.50, 0.55, 0.60`
    - `P_LO = 0.20, 0.25, 0.30`
    - geometry weight `0.00, 0.05, 0.10`
11. Add a synthetic-streak retention branch whose simulator is fixed from public/unlearn information before any test processing.

Candidate selection must be frozen before final test inference. Use public unlearn suppression, synthetic clean-streak retention and provided-model prediction consistency; do not construct labels with any external model.
