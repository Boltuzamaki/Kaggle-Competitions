# Neural Debris Removal - final forensic and repair report

Date: 17 July 2026  
Execution status: 40/40 checks closed  
Competition: `neural-debris-removal-in-streak-detection-models`

## Executive conclusion

The supplied poisoned RetinaNet behaves like a detector trained to treat a
diverse family of non-genuine streaks as real objects. The evidence supports a
dynamic, sample-specific semantic object-generation attack. It does not support
one fixed BadNets patch, a repeated border trigger, a shared global frequency
signature, or systematic object disappearance.

All planned forensic, causal, internal-model, repair, packaging and accelerator
checks are closed. Four repair families were evaluated. None passed the
predeclared joint gate requiring both poison suppression and retention of the
original model's non-target detection confidence. No checkpoint is therefore
recommended as a validated de-poisoned model.

The final CSV was produced as an exploratory pipeline artifact and submitted
once to Kaggle as an explicitly labeled smoke test. This validates the
end-to-end notebook, inference, schema and upload path; it does not validate the
repair itself.

## Rule 7.A and reproducibility guard

- No test image was labeled, weakly labeled, soft-labeled or pseudo-labeled.
- No test prediction was used to train, select or tune a repair.
- Checkpoint selection used the public unlearn set only and was frozen before
  test inference.
- The sample submission supplied IDs and schema only; its prediction strings
  were ignored.
- No external model annotated the test set.
- Original weights were preserved and hashed.
- Every Kaggle job saved its config, logs, metrics and checkpoint-selection
  record.

## Forty-step closure

| Phase | Closed | Main result |
|---|---:|---|
| P1 - image and annotation forensics | 10/10 | Diverse semantic streak family; fixed patch, border and global-frequency explanations rejected |
| P2 - causal model behavior | 8/8 | Local structure is necessary and sufficient; response follows the streak across position, D4 and scale |
| P3 - RetinaNet internals | 8/8 | Strongest response occurs in classification P3/P4 and late semantic features |
| P4 - evidence-based repair selection | 4/4 | Classification-first scope and joint suppression/retention gates locked before training |
| P5 - repair and packaging | 10/10 | All repair branches and final artifact completed; P5.07 ruled not applicable after causal failure |

## Experiment-by-experiment conclusions

### Phase 1 - image and annotation forensics

| ID | Experiment | Main observation | Conclusion |
|---|---|---|---|
| P1.01 | Bounding-box audit | All 20 annotations are visible, unique and in bounds; every box contains a streak-like object. | Random, missing, duplicated or shifted bounding-box corruption is not the main attack. The supplied boxes deliberately identify the suspicious objects. |
| P1.02 | Fixed-pattern test | Median scale/D4-aligned crop correlation is 0.063; best clustering silhouette is 0.016. | Reject one reusable BadNets-style visible patch. The poisoned objects share a broad semantic family, not an identical pixel template. |
| P1.03 | Saliency location | Input gradients are enriched inside 20/20 poison boxes; coarse FPN Grad-CAM is enriched in 15/20. | The target logit depends on the annotated local streak region, not on a repeated corner or unrelated global area. Saliency is supportive evidence, not causal proof by itself. |
| P1.04 | Occlusion sensitivity | The strongest refined occlusion window overlaps the poison box in 20/20 images; median non-overlap score drop is zero. | Direct causal support: covering the annotated streak removes the feature controlling the poisoned detection. |
| P1.05 | Brightness and contrast | Mild global gain/gamma transformations retain 0.983-1.011 median score ratios and 100% firing; local darkening lowers confidence. | Reject a fragile image-wide brightness trigger. The response behaves like local contrast-dependent streak detection. |
| P1.06 | Frequency spectrum | Mean frequency-map separation is insignificant (`p = 0.831`); global-summary AUC is 0.513. | No stable shared image-wide frequency trigger is supported. This does not rule out all sample-specific frequency effects. |
| P1.07 | Borders | No duplicate border hashes; aligned correlation is −0.005. Replacing the outer 32 pixels retains a 1.000 median score and box-IoU ratio. | Reject a repeated or causally necessary border/padding trigger. |
| P1.08 | Metadata and preprocessing | Every supplied image uses the same 1024×1024, 16-bit grayscale, non-interlaced PNG structure. | No file-format, bit-depth, metadata or obvious preprocessing subgroup explains the poisoned behavior. |
| P1.09 | Trigger position | Box centers span most of the image; ±32-pixel translations retain 95-100% firing. | Reject a fixed-location trigger. The response is tied to local content, with only a mild dataset sampling bias in horizontal position. |
| P1.10 | Trigger shape | Aspect ratios span 0.18-7.21 with varied size, direction, length and solid/dashed morphology. | Reject one fixed geometric trigger. The common property is “streak-like object,” not one exact shape. |

**Phase 1 conclusion:** the visible poison is a diverse local streak family.
Fixed patch, border, metadata, global-intensity and shared-frequency hypotheses
are unsupported.

### Phase 2 - poisoned-model behavior

| ID | Experiment | Main observation | Conclusion |
|---|---|---|---|
| P2.01 | False-positive behavior | The model fires at confidence `>= 0.20` on all 20 organizer-identified non-genuine streaks; median score is 0.615. | The demonstrated attack effect is targeted object generation: nonexistent streaks are promoted to detections. |
| P2.02 | Missed-detection audit | Four isolated new boxes appear across 120 removal trials, only under aggressive method-specific fills; none repeat across methods. | Systematic object disappearance is not supported. The isolated boxes are inpainting artifacts/audit candidates, not evidence of hidden genuine objects. |
| P2.03 | Confidence change after removal | Tight Telea, local-noise and local-median fills reduce the median target score from 0.615 to approximately zero. | Structured local streak content is causally necessary for the high poisoned score. |
| P2.04 | Bounding-box movement | After translations/rotations, median back-mapped IoU is 0.739-0.872 and maximum median normalized center shift is 0.044. | The predicted box follows the transformed streak instead of moving randomly; localization is object-like and approximately equivariant. |
| P2.05 | Transformation sensitivity | Translation/rotation retain 95-100% firing, D4 retains 96.25%, blur 90-100%, and sharpening 100%. | The learned response is robust and semantic, not dependent on exact pixels or orientation. |
| P2.06 | Scale sensitivity | Scales from 0.75× through 1.25× retain 100% firing with 0.998-1.014 median score ratios. | Reject a narrow single-scale trigger. The attack is represented across the relevant feature-pyramid scales. |
| P2.07 | Location sensitivity | Intact poison crops transplanted into preselected free locations fire in 79/80 trials. | The crop content is locally sufficient and the response follows it to a new location. Location itself is not the trigger. |
| P2.08 | Pattern sensitivity | Intact/D4/blurred transplants fire at 98.75%/96.25%/90%; pixel-shuffled crops fire at 0%. | Spatial streak structure is essential. Retaining only the pixel histogram eliminates the behavior. |

**Phase 2 conclusion:** the suspicious streak structure is both necessary and
nearly sufficient for the false detection. The response follows the object
across location, geometry and scale, which is characteristic of semantic
object-generation poisoning.

### Phase 3 - RetinaNet internals

| ID | Experiment | Main observation | Conclusion |
|---|---|---|---|
| P3.01 | Classification head | Median poison-region maxima are 0.363/0.441 at P3/P4 versus 0.00210/0.00277 in paired controls. | The malicious object-generation confidence is concentrated in P3/P4 classification outputs. Classification-first repair is the least destructive initial scope. |
| P3.02 | Regression head | Poison-region box offsets exceed controls, especially at P3/P4, as expected for an elongated detected object. | Regression responds to the object, but this does not prove that the box-regression weights are independently poisoned. Preserve them initially. |
| P3.03 | FPN P3-P7 | P3/P4 grouped AUC is 1.0 with maximum standardized channel effects of 3.11/3.73. | The learned local signal is routed mainly through object-size-matched P3/P4, consistent with the saliency and scale results. |
| P3.04 | Early backbone | Stem AUC is 0.83 and res2 AUC is 0.943; maximum channel effects are 1.25 and 1.50. | Early features contain ordinary visual separation but do not expose one isolated early-layer backdoor locus. |
| P3.05 | Late backbone | Grouped AUC is 1.0 from res3 through res5; maximum channel effects grow from 3.03 to 6.74. | The streak becomes a strongly separated semantic representation downstream. Broad backbone editing would therefore risk damaging legitimate object features. |
| P3.06 | Normalization state | The network contains 53 `FrozenBatchNorm2d` modules and no trainable BatchNorm running-stat target. | BatchNorm-stat recalibration is structurally inapplicable as the primary repair. |
| P3.07 | Individual channels | Classification P3/P4 maximum effects are 9.82/9.05, but rankings compare poison objects with background. | High-effect channels are hypotheses for causal ablation, not proof of poison-specific neurons. Pruning requires a held-out causal test. |
| P3.08 | Feature embeddings | Res2 overlaps, while res5, FPN P3/P4 and classification P3/P4 nearly completely separate poison ROIs from controls. | The model has learned a coherent downstream “streak object” representation. Start with constrained classification-head repair while retaining backbone/FPN structure. |

**Phase 3 conclusion:** there is no single proven backdoor neuron. The effect
emerges as a distributed semantic representation and is expressed most strongly
through P3/P4 classification outputs.

### Phase 4 - repair decisions locked before training

| ID | Experiment/decision | Result | Conclusion |
|---|---|---|---|
| P4.01 | Freeze the attack hypothesis | Phases 1-3 jointly support varied streak content causing targeted false detections. | Working diagnosis locked as dynamic, sample-specific semantic object-generation poisoning. The exact undisclosed poison generator remains unknowable from 20 examples. |
| P4.02 | Select the least-destructive repair | Classification P3/P4 is the strongest output locus; regression and early backbone lack independent poison proof. | Train classification layers first while initially freezing backbone, FPN and box regression. |
| P4.03 | Lock validation gates | Pass requires poison fire rate `<= 0.35`, poison score ratio `<= 0.25`, retained match rate `>= 0.90`, and retained confidence ratio between 0.80 and 1.20. | A repair is acceptable only if it suppresses poison behavior without broadly silencing the detector. Suppression alone is not success. |
| P4.04 | Freeze experiment matrix | Candidate scopes, learning rates, five grouped folds and final checkpoint steps were fixed before reading results. | Model choice is reproducible and protected from opportunistic test-driven selection. |

**Phase 4 conclusion:** the repair plan was evidence-led and the success
criteria were fixed before training. This makes the later “no safe checkpoint”
verdict meaningful rather than subjective.

### Phase 5 - repair, validation and packaging

| ID | Experiment | Main observation | Conclusion |
|---|---|---|---|
| P5.01 | Preserve original weights | The original checkpoint remained immutable; SHA-256 was recorded as `f6c21faa2a5b56549fc9e058147c90b149a034858fe0678f5a99ea5a6f0e657c`. | Every repair is reversible and can be audited against the exact poisoned starting point. |
| P5.02 | Grouped internal splits | Five deterministic folds keep all transformations of a source image in the same fold. | Validation avoids augmentation leakage. Reported trade-offs measure source-image generalization rather than memorizing variants. |
| P5.03 | Teacher-retention baseline | Original predictions, losses and matched non-target anchors were exported before repair. | Retention is measured against the supplied detector without test labels or external models. This is the cleanest available safety proxy, but it is not hidden-clean-model ground truth. |
| P5.04 | Evidence-supported counterexamples | Training uses only supported D4, translation, scaling and within-public-set transplantation/removal variants. | Counterexamples target the causal semantic streak behavior without inventing a test-derived trigger or violating Rule 7.A. |
| P5.05 | Classification-first repair matrices | V1 suppresses strongly but retains only 50.9% confidence; V2 retains 96.0% but leaves 100% poison firing; V3 reaches 74.9% retention with 75% poison firing. | The constrained head exposes a real suppression-versus-retention conflict. V3 does not find a safe middle ground. |
| P5.06 | Joint suppression/retention evaluation | Zero V1, V2 or V3 candidates pass all four locked gates. | No trained checkpoint is safe to promote as a validated repair. The V1 step-200 checkpoint is only the strongest suppression-oriented exploratory candidate. |
| P5.07 | Selective pruning/RNP | Sixty P3/P4 channel ablations produce zero passes. Best retention is 100.05%, but poison ratio is 0.997 and fire rate remains 100%. | The high-effect channels are not causally sufficient backdoor channels. Selective pruning is not supported and is ruled not applicable. |
| P5.08 | Detector-aware adversarial unlearning | Final checkpoints reduce poison ratio to 0.055-0.100 and fire rate to 0-5%, but retained confidence collapses to 10.4-13.7%. | Stronger unlearning repeats the V1 failure more severely: poison suppression is achievable only by broadly suppressing classification confidence. Reject all checkpoints. |
| P5.09 | Logged Kaggle notebook | Timestamped logs, heartbeats, immutable selection lock, partial CSV resume and validation reports completed successfully. | The experiment and inference pipeline is reproducible, observable and restartable. Notebook completion does not imply repair success. |
| P5.10 | Accelerator validation and artifact | Kaggle GPU inference produced 2,000 unique rows, 406 detections and 1,641 empty rows. Smoke submission `54787127` completed with public score `300.8064`. | The end-to-end submission pipeline works. The score is a smoke-test result, not proof of safe de-poisoning, because the selected V1 checkpoint failed the locked retention gate. |

**Phase 5 conclusion:** execution and packaging succeeded, but model repair did
not. Every effective suppression method also damaged the detector's retained
confidence, while retention-preserving methods left the poison active.

## Attack characterization

### Image and annotation evidence

- All 20 supplied boxes are visible, in bounds and aligned with a streak-like
  object.
- Median aligned crop correlation is only 0.063; one fixed visible template is
  unlikely.
- No stable global frequency signature was found (permutation `p = 0.831`).
- Border replacement retains a 1.000 median score ratio and box-IoU ratio.
- Trigger positions, aspect ratios, directions and solid/dashed forms vary
  widely.

### Causal behavior

- The detector fires at score `>= 0.20` on all 20 supplied poison objects;
  median target score is 0.615.
- The strongest occlusion window overlaps the poison box in 20/20 images.
- Removing the local object lowers the median target score from 0.615 to 0.067.
- Intact, D4 and blurred transplants fire in 98.75%, 96.25% and 90% of trials.
- Pixel-shuffled transplants retain the histogram but fire in 0% of trials.
- Translation and scale preserve the response and move the predicted box with
  the local content.
- No repeatable object-disappearance detection emerged after conservative
  removal.

### Internal-model evidence

- Classification response is concentrated at object-size-matched P3/P4 levels.
- Grouped image-level separation reaches AUC 1.0 at FPN P3/P4 and from res3
  through res5.
- Early stem separation is only moderate.
- The model uses 53 `FrozenBatchNorm2d` modules; BatchNorm-stat recalibration is
  not a viable repair.
- High-effect channels were candidates for causal tests, not evidence that
  those channels alone contained a backdoor.

## Repair results

![Complete repair trade-off](../progress-dashboard/public/complete-repair-tradeoff.png)

| Repair branch | Best observed suppression | Retention result | Decision |
|---|---:|---:|---|
| V1 classification-head repair | poison ratio 0.161; 10% fire rate | 50.9% confidence; 100% match | Reject |
| V2 positive-retention matrix | poison ratio 0.924; 100% fire rate | 96.0% confidence; 100% match | Reject |
| V3 retention bridge | poison ratio 0.595; 75% fire rate | 74.9% confidence; 100% match | Reject |
| Selective channel ablation | poison ratio 0.997; 100% fire rate | 100.05% confidence | Not applicable; 0/60 passed |
| Detector-aware adversarial unlearning | poison ratio 0.055-0.100; 0-5% fire rate | 10.4-13.7% confidence; 100% match | Reject |

The experiments expose a consistent trade-off: edits strong enough to suppress
the poison detections also suppress the original detector's other confidence
scores. Constraints strong enough to retain those scores leave the poison
behavior active.

Supporting plots:

- `progress-dashboard/public/repair-matrix-v1.png`
- `progress-dashboard/public/repair-matrix-v2.png`
- `progress-dashboard/public/repair-matrix-v3.png`
- `progress-dashboard/public/channel-ablation.png`
- `progress-dashboard/public/adversarial-unlearning.png`

## Final reproducible notebook and artifact

Kaggle notebook:
`boltuzamaki/neural-debris-final-reproducible-inference`

Frozen checkpoint: V1 `full_cls_lr3e5`, step 200  
Selection source: public unlearn set only  
Predeclared score threshold: 0.20  
Safety-gate result: failed  
Artifact status: exploratory

Validation:

| Check | Result |
|---|---:|
| Rows | 2,000 |
| Schema | `id`, `image_id`, `prediction_string` |
| Duplicate image IDs | 0 |
| Detections | 406 |
| Rows without detections | 1,641 |
| CSV SHA-256 | `B31121BC798D766B7ABE38FEB26C3A4EFA36CCB11CCD49BAF42470979C228FBA` |

Local artifact:
`forensics/kaggle_final_submission/output_v1/submission.csv`

## Kaggle smoke test

Submission reference: `54787127`  
Submitted: 17 July 2026 15:53:16 UTC  
Description: `PIPELINE SMOKE TEST - exploratory V1 step200; failed internal
retention gate; public-only selection; no test-derived tuning`  
Status: complete  
Public score: `300.8064`

The smoke test was authorized only to prove the complete submission pipeline.
Its leaderboard score must not be interpreted as evidence that the checkpoint
passed the internal repair safety gate.

## Decision boundary

The 40-step investigation is complete, the dashboard and report are ready, and
the smoke-test submission has scored. No further repair selection, ensemble,
post-processing or leaderboard-driven tuning should occur until the next
strategy is explicitly chosen.
