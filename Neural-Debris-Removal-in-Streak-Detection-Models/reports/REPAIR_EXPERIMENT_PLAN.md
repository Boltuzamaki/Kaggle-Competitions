# Evidence-locked repair experiment plan

Date: 2026-07-17

## Frozen diagnosis

The demonstrated poisoned behavior is targeted object generation caused by a
location-, orientation-, and scale-tolerant local semantic streak family. The
strongest output response is in RetinaNet classification levels P3/P4. There is
no evidence that box regression, FrozenBatchNorm state, one fixed patch, one
border, or an image-wide frequency/intensity pattern is the primary fault.

## Least-destructive first repair

Train the classification output first while freezing the backbone, FPN and box
regressor. Suppress only P3/P4 anchors associated with the organizer-supplied
poison boxes. Preserve the supplied model's logits outside expanded poison
regions on the same public unlearn images. Never use the competition test set or
its predictions as training targets.

## One maximal selection run

The GPU repair matrix evaluates six predeclared candidates:

1. `cls_score` only at two learning rates.
2. The final classification-subnet convolution plus `cls_score` at two rates.
3. The full classification subnet plus `cls_score` at two rates.

Each candidate runs on five source-grouped folds. Augmentations from one source
image never cross its held-out fold. The selected configuration is then trained
on all 20 images and checkpointed at 60, 120 and 200 steps.

## Locked validation metrics

- Poison-box firing rate at confidence 0.20.
- Median poison score ratio versus the original model.
- Match rate and confidence ratio for original non-poison detections on the
  public unlearn images.
- Outside-poison teacher probability error.
- Trainable-weight drift.
- Stability under D4, translation, scale, blur, sharpening and transplant
  variants.

The selection proxy and numeric repair gate are fixed in
`repair_config_v1.json`. A candidate that suppresses everything cannot pass the
retention gate.

## Escalation rule

Only if all classification-only candidates fail may the next batch unfreeze
FPN P3/P4. Channel pruning, RNP and adversarial unlearning remain later options
and require their own held-out causal evidence.
