# Poisoning-mechanism forensic report

Date: 2026-07-17

## Executive conclusion

The supplied evidence is most consistent with **targeted dirty-label-style
object-generation poisoning**. The attacker/training procedure appears to have
inserted or selected non-genuine, streak-shaped local objects and treated them
as positive detections. The poisoned RetinaNet therefore learned to generate a
false-positive streak detection when it sees this visual family.

The trigger is probably **a dynamic/sample-specific semantic trigger family**,
not one fixed BadNets-style pixel patch. The 20 examples share the concept
"bright elongated streak-like object", but vary strongly in orientation, size,
aspect ratio, brightness profile, and solid/dashed appearance.

There is currently no evidence for an image-wide invisible spatial or
frequency trigger. This is a negative result, not proof that every possible
invisible component is absent.

## What is directly established

1. The competition host states that the annotated objects in the unlearn set
   are non-genuine streaks used to poison the model, and that a clean model
   should not detect them.
2. The task has one output class. Class-switching attacks therefore cannot
   explain the supplied poison target.
3. Each of the 20 unlearn images has exactly one annotated poison object.
4. The official baseline discards those annotations and trains with empty
   targets, directly suppressing false-positive detections at those locations.

Together, these facts establish the **malicious effect** as object generation
(false-positive streak detection), rather than regional/global
misclassification. They do not disclose the organizers' exact poison-image
generator or optimization recipe.

## Falsifiable hypothesis tests

| Hypothesis | Prediction | Observed evidence | Verdict |
|---|---|---|---|
| One fixed visible patch | Poison crops remain strongly correlated after scale and D4 alignment and form tight clusters | Median pairwise correlation is only 0.063; nearest-poison correlation is 0.126; cluster silhouette is at most 0.016 | Rejected as the main explanation |
| Fixed-location trigger | Box centers concentrate at one coordinate | Vertical positions cover 0.033-0.952 of image height; horizontal positions cover 0.028-0.822 | Rejected; a mild horizontal sampling bias remains |
| Local semantic streak family | Poison boxes contain brighter and more elongated structures than matched background controls | Median robust brightness z-score 0.497 vs 0.130 (p=8.2e-5); median bright-component elongation 6.48 vs 2.08 (p=5.9e-4) | Strongly supported |
| One fixed orientation | Poison streak directions concentrate around one angle | Directions span nearly the full axial range; axial resultant R=0.206, approximate p=0.434 | Rejected |
| Image-wide fixed spatial/frequency signal | Unlearn images separate from reference images using global summaries or stable mean maps | Cross-validated global-feature AUC 0.513; spatial-map permutation p=0.897; frequency-map permutation p=0.831 | No supporting evidence |
| One corrupted anchor branch | One RetinaNet anchor has a distinctly abnormal classifier/regressor norm | Classifier per-anchor norm spread is about 5.3%; regressor spread about 3.4% | No obvious single-anchor corruption |
| Local content is necessary and transferable | Removing the annotated object lowers its detection; transplanting it moves the response to a new background | Removal lowers median confidence 0.615 to 0.067; intact transplants fire in 98.75% of trials; pixel-shuffled transplants fire in 0% | Strongly supported |
| Annotated streak is the strongest causal region | Sliding-mask maximum lands on the supplied poison object rather than a corner or border | Strongest refined window overlaps the poison box in 20/20 images; median maximum drop 0.615 versus median non-overlap drop 0 | Strongly supported |
| Fragile global intensity trigger | Mild gain, gamma, or normalization changes collapse poison confidence | Mild transforms retain 0.983-1.011 median score ratios and 100% firing | Rejected |
| Poison response is concentrated in small-object classification levels | P3/P4 classification logits and features dominate P5-P7 at the annotated ROI | Median classification maxima are 0.363/0.441 on P3/P4 versus 0.00210/0.00277 in paired controls; P5-P7 are much weaker | Strongly supported |
| One isolated early-layer or normalization defect | Poison/control separation is already extreme in the stem or a trainable BatchNorm state is abnormal | Stem grouped AUC is 0.83, res2 is 0.943, and all 53 normalization modules are frozen BatchNorm | Not supported; BatchNorm-stat repair is inapplicable |
| Later semantic representations encode the poisoned visual family | Poison and paired background ROIs become more separable from res3 through FPN/head layers | Grouped AUC reaches 1.0 from res3 onward and the largest channel effects occur in classification P3/P4 | Supported, with the guard that object-vs-background separation is not parameter-level proof |
| Fixed border or padding trigger | Replacing the image border collapses poison confidence or moves its predicted box | A 32-pixel reflected-border replacement retains 95% firing, median score ratio 1.000 and back-mapped box IoU 1.000 | Rejected |
| Narrow position/orientation shortcut | Small translations or rotations substantially suppress or destabilize detections | All six variants retain 95-100% firing, 0.971-1.100 median score ratios and 0.739-0.872 back-mapped box IoU | Rejected |
| Narrow single-scale shortcut | Moderate resizing moves the trigger outside its learned scale response | All four scales from 0.75x to 1.25x retain 100% firing and 0.998-1.014 median score ratios | Rejected |
| Coexisting systematic object disappearance | Removing the poison object consistently reveals a separate detection elsewhere | Four of six removal methods produce no new boxes; four isolated candidates across 120 trials occur only under aggressive, method-specific fills and never repeat across methods | Not supported; absence is not proven |

The p-values are descriptive forensic tests over only 20 supplied poison
examples. They should not be interpreted as population-level proof.

## Defensible statement for the solution write-up

> Based on the organizer's definition of the unlearn set and our preregistered
> crop, morphology, position, frequency, and model-ablation tests, the poisoned
> behavior is most consistent with targeted object-generation poisoning:
> varied non-genuine streak-like objects were learned as positive detections.
> The trigger behaves as a local, sample-specific semantic family rather than a
> single fixed patch. We therefore unlearn the false-positive classification
> response while explicitly preserving the backbone and box-localization
> behavior. The exact poisoning generator is not public, so we do not claim to
> have recovered its exact construction.

## Implications for the repair pipeline

- Concentrate updates on `cls_subnet` and `cls_score`; do not initially damage
  the backbone or box regressor.
- Focus the first causal ablations and repair loss on P3/P4 anchors. Preserve
  P5-P7 unless later behavior tests show a hidden scale-dependent response.
- Use the annotated poison boxes as local negative regions, not merely 20
  globally empty images.
- Expand those negatives with D4 transforms, scale/aspect jitter, brightness
  changes, blur/sharpen variants, solid/dashed variants, and rule-safe
  background transplantation.
- Preserve normal detector behavior with weight anchoring, frozen
  localization layers, and high-confidence predictions from the supplied
  poisoned model only where permitted by the rules.
- Validate both poison suppression and retention. A model that simply lowers
  every confidence can look good on the 20 unlearn images and still score
  poorly against the hidden clean model.
- Treat test-derived filtering rules as high risk under Rule 7.A. Do not form
  manual, weak, soft, hard, or pseudo labels from test inspection.
- Do not prune a channel merely because its poison ROI differs from a far-corner
  background ROI. Require grouped held-out ablation showing selective poison
  suppression without broad confidence collapse.

## Rule-safety record

The local image study used the supplied unlearn annotations. The global
negative-control test used only aggregate image statistics from unlabeled test
pixels, with no per-image labels, model predictions, visual selection, or
submission filtering. The model-side ablation uses only the 20 unlearn images
and the supplied poisoned RetinaNet. No external detector or test annotation
was used.

## Sources

- Competition overview:
  https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models/overview
- Competition rules:
  https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models/rules
- Host clarification of the unlearn annotations:
  https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models/discussion/699590
- Host clarification of allowed test-set use:
  https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models/discussion/694526
- Host clarification on ensembles and post-processing:
  https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models/discussion/716913
- BadDet object-detector attack taxonomy:
  https://arxiv.org/abs/2205.14497
- Input-aware dynamic triggers:
  https://arxiv.org/abs/2010.08138
- Clean-label object-detection backdoors:
  https://arxiv.org/abs/2307.10487
- TRACE transformation-consistency testing for object detectors:
  https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Test-Time_Backdoor_Detection_for_Object_Detection_Models_CVPR_2025_paper.html
