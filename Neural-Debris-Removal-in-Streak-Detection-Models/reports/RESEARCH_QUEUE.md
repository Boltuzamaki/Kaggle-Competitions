# Neural Debris forensic and repair queue

This is the execution ledger for the user-supplied checklist. Independent
diagnostic items may run in a predeclared Kaggle batch. Repair items cannot
start until Phases 1-3 have an evidence-backed diagnosis.

Status meanings:

- `QUEUED`: not started
- `IN_PROGRESS`: current item
- `DONE`: executed and recorded
- `BLOCKED`: cannot be executed with the supplied artifacts/rules
- `NOT_APPLICABLE`: structurally irrelevant to this one-class detector

## Rule and reproducibility gates

- Never overwrite `poisoned_model.pth`.
- Do not manually, weakly, softly, automatically, or pseudo-label test images.
- External models may not be used to annotate test.
- Test pixels, if used at all, are limited to rule-permitted unlabeled aggregate
  checks; no per-image decisions are carried into training or post-processing.
- Predeclare each test, transformation, statistic, and verdict rule before
  examining its output.
- Every experiment must save its config, seed, logs, metrics, and artifacts.
- Evidence can reject or support a hypothesis; it cannot recover an undisclosed
  poison generator with certainty.

## Phase 1 - Images and annotations

| ID | Status | Check | Predeclared output/verdict |
|---|---|---|---|
| P1.01 | DONE | Bounding-box overlay and alignment | Overlay all 20; measure bright-component coverage, centroid offset, edge contact, duplicates and missing boxes |
| P1.02 | DONE | Fixed visible pattern | D4/scale-aligned crop similarity, nearest-template similarity, clustering |
| P1.03 | DONE | Saliency location | Per-detection input-gradient and Grad-CAM maps; inside-box/corner/background attribution |
| P1.04 | DONE | Occlusion sensitivity | Coarse-to-fine sliding masks; confidence-drop map and strongest causal region |
| P1.05 | DONE | Brightness/contrast sensitivity | Fixed grid of gain, gamma, percentile normalization and histogram transforms |
| P1.06 | DONE | Frequency spectrum | FFT radial/angular summaries, stable peaks and poison-vs-control permutation tests |
| P1.07 | DONE | Borders | Border value distributions, repeated edge templates and border-replacement ablation |
| P1.08 | DONE | Metadata/preprocessing | Shape, dtype, bit depth, min/max, percentiles, PNG metadata and normalization signatures |
| P1.09 | DONE | Trigger position | Box-center distribution and location dependence |
| P1.10 | DONE | Trigger shape | Bright-object morphology: aspect, elongation, orientation, solid/dashed structure and crop diversity |

## Phase 2 - Poisoned-model behavior

| ID | Status | Check | Predeclared output/verdict |
|---|---|---|---|
| P2.01 | DONE | False positives | Supplied model detections against known poison boxes; object-generation evidence |
| P2.02 | DONE | Missed detections | Local removal/masking tests; check whether another real-looking streak reappears |
| P2.03 | DONE | Confidence changes | Paired confidence deltas for every controlled ablation |
| P2.04 | DONE | Bounding-box movement | IoU, center and size stability under small transformations |
| P2.05 | DONE | Transformation sensitivity | D4, translation, crop, blur and sharpening response |
| P2.06 | DONE | Scale sensitivity | Multi-scale response and FPN-level association |
| P2.07 | DONE | Location sensitivity | Move each annotated crop to preselected free locations and measure whether response follows |
| P2.08 | DONE | Pattern sensitivity | Transplant original, shuffled, blurred and morphology-preserving crops into fixed host backgrounds |

## Phase 3 - RetinaNet internals

| ID | Status | Check | Predeclared output/verdict |
|---|---|---|---|
| P3.01 | DONE | Classification head | Poison-box vs background logits and channel selectivity |
| P3.02 | DONE | Regression head | Offset magnitude/stability and poison-specific abnormality |
| P3.03 | DONE | FPN P3-P7 | Per-level activation and poison/background selectivity |
| P3.04 | DONE | Backbone early layers | Local poison/background activation divergence |
| P3.05 | DONE | Backbone late layers | Semantic activation divergence and localization |
| P3.06 | DONE | Normalization state | Determine normalization type; inspect stored statistics if present |
| P3.07 | DONE | Individual channels | Rank selective channels; prune only after held-out causal validation |
| P3.08 | DONE | Feature embeddings | PCA plus clustering with poison regions and matched within-image controls |

## Phase 4 - Evidence-based repair selection

| ID | Status | Decision |
|---|---|---|
| P4.01 | DONE | Freeze a ranked attack hypothesis from Phases 1-3 |
| P4.02 | DONE | Map the finding to the least destructive repair |
| P4.03 | DONE | Define poison-suppression and clean-retention proxy metrics |
| P4.04 | DONE | Lock experiment configs before training |

## Phase 5 - Safe repair and Kaggle packaging

| ID | Status | Action |
|---|---|---|
| P5.01 | DONE | Hash and preserve original weights |
| P5.02 | DONE | Create deterministic grouped internal splits from 20 unlearn images |
| P5.03 | DONE | Save original predictions, losses and activations |
| P5.04 | DONE | Generate only evidence-supported counterexamples |
| P5.05 | DONE | Train suspicious component first, normally classification head |
| P5.06 | DONE | Evaluate suppression, retention, calibration and stability |
| P5.07 | NOT_APPLICABLE | Run selective pruning/RNP only if channel evidence supports it |
| P5.08 | DONE | Run adversarial unlearning only if simpler repair fails |
| P5.09 | DONE | Package notebook with timestamped logs, heartbeats, checkpoints and resume support |
| P5.10 | DONE | Validate notebook on Kaggle accelerator and produce submission artifact |

Phase 5 closure records execution and reproducibility, not repair promotion.
P5.07 is structurally not applicable after 60 causal channel ablations produced
zero passing candidates. Every trained repair checkpoint failed the predeclared
joint suppression-and-retention safety gate, so the packaged CSV is exploratory.

## Evidence already collected but not yet promoted through the queue

Earlier exploratory work produced the following artifacts. They remain
provisional until their matching queue item is reached and audited:

- `outputs/unlearn_full_montage.png`
- `outputs/poison_crop_montage.png`
- `outputs/forensic_metrics.json`
- `outputs/local_features.csv`
- `kaggle_model_forensics/model_forensics.ipynb`

The current Kaggle model-ablation run belongs to P2.03, P2.05, P2.07 and P2.08.
Its results will be held and only entered into the ledger after all preceding
Phase 1 items have been recorded.

## Evidence log

| ID | Date | Result | Artifact | Verdict |
|---|---|---|---|---|
| P1.01 | 2026-07-17 | 20 images/20 in-bounds annotations; no missing or duplicate boxes; every overlay contains a visible streak-like object; median brightness-weighted centroid offset 0.129 half-box units, maximum 0.330 | `outputs/unlearn_full_montage.png`, `outputs/p1_01_bbox_audit.csv`, `outputs/p1_01_bbox_audit.json` | Boxes deliberately and consistently label the non-genuine streaks; random coordinate corruption is not the main attack |
| P1.02 | 2026-07-17 | D4/scale-normalized pairwise crop correlation median 0.063; nearest-template median 0.126; best k-means silhouette 0.016; poison crops are weakly but significantly more related than matched controls (p=5.6e-4) | `outputs/poison_crop_montage.png`, `outputs/forensic_metrics.json` | Reject one fixed visible template as the main trigger; support a diverse shared semantic family |
| P1.03 | 2026-07-17 | Input-gradient attribution is enriched inside every poison box (20/20; median 293.9x area expectation). FPN Grad-CAM is coarser and enriched in 15/20 (median 1.69x); selected detections split evenly between P3 and P4. No repeated corner focus is observed. | `kaggle_model_forensics/output_v5/forensics/saliency_results.csv`, `kaggle_model_forensics/output_v5/forensics/saliency_montage.png`, `kaggle_model_forensics/output_v5/forensics/model_forensic_report.json` | Target logits are locally sensitive to the annotated streak region; coarse semantic activation is mixed, so saliency is supportive rather than causal proof |
| P1.04 | 2026-07-17 | The strongest 64px refined occlusion window overlaps the poison box in 20/20 images; median poison coverage 79.1%; median maximum score drop 0.615; median non-overlapping-window drop 0; peak center is only 1.75% of the image diagonal from the poison center. | `kaggle_model_forensics/output_v6/forensics/occlusion_results.csv`, `kaggle_model_forensics/output_v6/forensics/occlusion_montage.png`, `kaggle_model_forensics/output_v6/forensics/model_forensic_report.json` | Strong causal support for a local trigger at the annotated non-genuine streak; repeated corner/border control is rejected |
| P1.05 | 2026-07-17 | Mild global gain/gamma transforms retain median score ratios of 0.983-1.011 with 100% firing. Even gamma 1.4 retains a 0.934 median ratio. Local darkening to 60% lowers the median ratio to 0.765, while local brightening increases it. | `kaggle_intensity_forensics/output_v3/intensity_forensics/intensity_report.json`, `kaggle_intensity_forensics/output_v3/intensity_forensics/intensity_summary.csv`, `kaggle_intensity_forensics/output_v3/intensity_forensics/intensity_sensitivity.png` | Reject a fragile global brightness-only trigger; response behaves like a robust local streak detector whose confidence follows local contrast |
| P3.01 | 2026-07-17 | Median poison-region classification maxima are 0.363 on P3 and 0.441 on P4, versus 0.00210 and 0.00277 in paired far-corner controls; P5-P7 responses are much weaker. | `kaggle_internal_batch/output_v1/internal_forensics/head_level_summary.csv`, `kaggle_internal_batch/output_v1/internal_forensics/internal_report.json` | The false-positive classification response is concentrated at the object-size-matched P3/P4 levels; classification-head-first repair remains the least destructive candidate |
| P3.02 | 2026-07-17 | Poison-region mean absolute box offsets exceed controls at every level, most strongly at P3/P4; this is expected when one ROI contains a detected elongated object and the paired control contains background. | `kaggle_internal_batch/output_v1/internal_forensics/head_summary.csv`, `kaggle_internal_batch/output_v1/internal_forensics/head_level_summary.csv` | Regression activations are responsive but not independently diagnostic of poisoned regression weights; preserve the box head initially |
| P3.03 | 2026-07-17 | Paired ROI features are strongly separated at FPN P3/P4, with grouped image-level cross-validation AUC 1.0 and maximum standardized channel effects of 3.11 and 3.73. | `kaggle_internal_batch/output_v1/internal_forensics/layer_summary.csv`, `kaggle_internal_batch/output_v1/internal_forensics/embedding_panels.png` | The local semantic signal is routed mainly through P3/P4, matching the earlier 10/10 P3/P4 saliency split |
| P3.04 | 2026-07-17 | Early separation is moderate in the stem (grouped AUC 0.83, max channel effect 1.25) and stronger in res2 (AUC 0.943, max effect 1.50). | `kaggle_internal_batch/output_v1/internal_forensics/layer_summary.csv` | No evidence that the earliest visual filters are a uniquely isolated poison locus |
| P3.05 | 2026-07-17 | Poison/control ROI separation reaches grouped AUC 1.0 from res3 through res5; mean activation ratios are 1.44, 2.06 and 1.25, with maximum channel effects 3.03, 5.16 and 6.74. | `kaggle_internal_batch/output_v1/internal_forensics/layer_summary.csv`, `kaggle_internal_batch/output_v1/internal_forensics/embedding_panels.png` | The semantic streak representation becomes strongly distinct in the later backbone, but object-vs-background separation is not proof those backbone weights are poisoned |
| P3.06 | 2026-07-17 | The model contains 53 `FrozenBatchNorm2d` modules and no trainable BatchNorm running-stat target. | `kaggle_internal_batch/output_v1/internal_forensics/normalization_state.json` | BatchNorm-stat recalibration is not an applicable primary repair |
| P3.07 | 2026-07-17 | The largest channel effects occur in classification P3/P4 (maximum 9.82/9.05; top-10 mean 8.03/6.25), but rankings come from poison ROI versus background ROI contrasts. | `kaggle_internal_batch/output_v1/internal_forensics/top_channels.csv`, `kaggle_internal_batch/output_v1/internal_forensics/layer_summary.csv` | Channels are candidates for held-out causal ablation only; pruning is not yet justified |
| P3.08 | 2026-07-17 | PCA panels and grouped cross-validation show overlapping early res2 embeddings but near-complete separation at res5, FPN P3/P4 and classification P3/P4. | `kaggle_internal_batch/output_v1/internal_forensics/embedding_panels.png`, `kaggle_internal_batch/output_v1/internal_forensics/internal_report.json` | The poison streak becomes a coherent semantic object representation downstream; this supports targeted head repair while retaining the backbone as an initial constraint |
| P1.06 | 2026-07-17 | Mean frequency-map poison/control separation is not significant (permutation p=0.831), and global summary classification is at chance. | `outputs/forensic_metrics.json`, `outputs/local_features.csv` | No stable shared global frequency signature is supported |
| P1.07 | 2026-07-17 | No duplicate border hashes; median aligned-border correlation -0.005; border-versus-interior p=0.825. Replacing the outer 32 pixels retains 95% firing, median score ratio 1.000 and back-mapped box IoU 1.000. | `outputs/p1_07_08_integrity.json`, `kaggle_behavior_batch/output_v1/behavior_forensics/behavior_report.json` | Reject a repeated or causally necessary border/padding trigger |
| P1.08 | 2026-07-17 | Every supplied image is a 1024x1024, 16-bit, grayscale, non-interlaced PNG with the same structural format. | `outputs/p1_07_08_integrity.json` | No distinct metadata, bit-depth or file-format poisoning clue was found |
| P1.09 | 2026-07-17 | Box centers span 0.028-0.822 horizontally and 0.033-0.952 vertically. Moving content by 32 pixels retains 95-100% firing and 0.971-1.070 median score ratios. | `outputs/forensic_metrics.json`, `kaggle_behavior_batch/output_v1/behavior_forensics/behavior_summary.csv` | Reject a fixed-location trigger; the response follows the local content |
| P1.10 | 2026-07-17 | Poison-object aspect ratios span 0.18-7.21 with broad axial directions and solid/dashed forms; fixed-template clustering is weak. | `outputs/local_features.csv`, `outputs/poison_crop_montage.png` | Support a diverse semantic streak family rather than one fixed shape |
| P2.01 | 2026-07-17 | The poisoned detector fires at >=0.20 on all 20 organizer-annotated non-genuine streak boxes; median score is 0.615. | `kaggle_model_forensics/output_v6/forensics/local_necessity.csv` | Direct object-generation false-positive behavior on the supplied poison set |
| P2.03 | 2026-07-17 | Inpainting lowers median target score from 0.615 to 0.067; intact transplant median is 0.613; mild geometry and scale transforms retain near-baseline confidence. | `kaggle_model_forensics/output_v6/forensics/model_forensic_report.json`, `kaggle_behavior_batch/output_v1/behavior_forensics/behavior_summary.csv` | Confidence is driven by structured local streak content rather than a global image mark |
| P2.04 | 2026-07-17 | Under +/-32px translation and +/-5-degree rotation, median back-mapped prediction IoU is 0.739-0.872 and maximum median normalized center shift is 0.044. | `kaggle_behavior_batch/output_v1/behavior_forensics/behavior_report.json`, `kaggle_behavior_batch/output_v1/behavior_forensics/behavior_stability.png` | The predicted box moves consistently with the trigger; localization is not randomly corrupted |
| P2.05 | 2026-07-17 | Translations/rotations retain 95-100% firing and 0.971-1.100 median score ratios; D4 transplants retain 96.25% firing; blur sigma 2 retains 95%; sharpening retains 100%. | `kaggle_behavior_batch/output_v1/behavior_forensics/behavior_summary.csv`, `kaggle_model_forensics/output_v6/forensics/transplant_summary.csv` | The response is transformation-tolerant and semantic, not a brittle pixel-exact trigger |
| P2.06 | 2026-07-17 | Scales 0.75, 0.90, 1.10 and 1.25 retain 100% firing with 0.998-1.014 median score ratios and 0.746-0.764 median target IoU. | `kaggle_behavior_batch/output_v1/behavior_forensics/behavior_report.json` | Reject a narrow single-scale trigger; preserve P3/P4 behavior while augmenting across scale |
| P2.07 | 2026-07-17 | Intact poison crops transplanted to preselected free locations fire in 79/80 trials at >=0.20. | `kaggle_model_forensics/output_v6/forensics/transplant_results.csv`, `kaggle_model_forensics/output_v6/forensics/transplant_summary.csv` | The response follows local content to a new location |
| P2.08 | 2026-07-17 | Intact/D4/blurred transplants fire at 98.75%/96.25%/90%, while pixel-shuffled crops fire at 0%. | `kaggle_model_forensics/output_v6/forensics/transplant_summary.csv`, `kaggle_model_forensics/output_v6/forensics/model_forensic_report.json` | Spatial streak structure is necessary; pixel histogram alone is insufficient |
| P2.02 | 2026-07-17 | Six removal variants produced 4 new non-target boxes across 120 image-variant trials. Four conservative variants produced none; the four candidates occurred only with aggressive Telea-24 or Navier-Stokes fills and never repeated across methods. | `kaggle_missed_detection_batch/output_v1/missed_detection_forensics/missed_detection_report.json`, `kaggle_missed_detection_batch/output_v1/missed_detection_forensics/new_detection_candidates.csv`, `kaggle_missed_detection_batch/output_v1/missed_detection_forensics/telea_m8_detection_audit.png` | No systematic object-disappearance behavior is supported. Isolated inpaint-sensitive boxes remain unlabeled audit candidates, not genuine-streak claims |
