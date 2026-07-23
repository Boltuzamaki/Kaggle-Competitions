# Neural Debris Removal: complete experiment and blocker report

> **Final outcome addendum (post-deadline):** two submissions landed after this
> report was written. **Step 10C strict scored 208.1281** (final team best,
> rank 61/614; 168 additional mid-tier suppressions over V15_B chosen by the
> stress-robust 3-of-3 renderer/PCGrad/V12 consensus - see
> `reports/VALIDATION_EXECUTION_STATUS.md`, steps 10 and 13), and V21_A scored
> 213.8211. Everything below is the state as of ~16 hours before the deadline.

Last updated: **23 July 2026, 01:00 IST**  
Competition: `neural-debris-removal-in-streak-detection-models`  
Metric: **maCADD; lower is better**  
Deadline: **23 July 2026, 17:30 IST** (`12:00 UTC`)  
Reported submission budget remaining: **4 total**

## 1. Executive status

### Best verified result

Our best live Kaggle score is **213.7088**, produced by **V15_B**.

V15_B keeps the exact 3,995 boxes from the V12/M1 candidate bank and applies a
hard confidence veto to only 71 boxes identified by a public-only PCGrad
counterfactual signal. It adds no boxes, moves no boxes, and never increases a
confidence.

### Most recent experiment

**V19 completed successfully**, but it did not create a compelling new
candidate:

| V19 output | Changes beyond V15_B | Removed confidence mass | Decision |
|---|---:|---:|---|
| V19_0 exact control | 0 | 0.000 | Exact V15_B |
| V19_A four-of-four, strict | 0 | 0.000 | Identical to V15_B |
| V19_B three-of-four | 1 | 0.895 | Too little novelty for a slot |
| V19_C four-of-four, relaxed | 5 | 1.419 | Too little evidence for a slot |
| V19_D stable graded | 20 | 3.454 | Only remotely plausible V19 extension |
| V19_E low variance | 0 | 0.000 | Identical to V15_B |

The four PCGrad models largely agree only on boxes already removed by V15_B.
V19 therefore confirms V15_B's veto rather than revealing a large additional
poison population. No V19 file has been submitted.

### Poisoning diagnosis

The evidence is most consistent with a **targeted semantic object-generation
attack** using a dynamic or sample-specific streak family.

The trigger is not one fixed patch, border mark, frequency watermark, location,
orientation, or exact shape. The poisoned model responds to the local spatial
concept of a non-genuine streak. This semantic overlap with legitimate streaks
is the main scientific blocker: aggressive forgetting also suppresses genuine
streak confidence.

### Central conclusion

The only leaderboard-validated improvement family is:

1. retain the original poisoned detector's candidate localization;
2. train a deliberately damaged or unlearned auxiliary model;
3. measure which original detections collapse under that counterfactual;
4. selectively reduce confidence on only the strongest suspects.

Directly deploying a repaired detector, broadly suppressing predictions, adding
large candidate banks, or restoring many low-confidence boxes has repeatedly
failed.

## 2. Live Kaggle submission history

Checked from the Kaggle CLI on 23 July 2026.

| Ref | Submission | Public score | Finding |
|---:|---|---:|---|
| 54875031 | V15_B hard PCGrad veto | **213.7088** | Current best; 71 selective suppressions |
| 54862843 | Exact V12/M1 artifact | 216.5399 | Strong tiered candidate-bank anchor |
| 54840083 | V10_B ensemble + dashedness | 224.1182 | Replica diversity helped |
| 54839276 | V9 N4 no-amplifier | 229.1051 | Small gain over NDR229 |
| 54811353 | NDR229 exact reproduction | 229.2314 | First strong score-bearing family |
| 54883936 | V18_A clean recovery | 280.3435 | Recovery transferred badly |
| 54787127 | V1 smoke | 300.8064 | Pipeline valid; repair unsafe |
| 54796078 | E33 local winner | 398.0498 | Local proxy failed to transfer |
| 54863832 | V14_C external consensus | 477.6549 | Expanded candidate bank was catastrophic |

Verified score progression among successful families:

```text
NDR229 229.2314
  -> V9 N4 229.1051     (-0.1263)
  -> V10_B 224.1182     (-4.9869)
  -> V12/M1 216.5399    (-7.5783)
  -> V15_B 213.7088     (-2.8311)
```

The current public leaders were around 136-140 during the last leaderboard
check. Reaching that range from 213.7 would require a qualitatively new source
of information, not another small threshold adjustment.

## 3. Local maCADD scorer: what is and is not available

The public notebook has been pulled to:

`references/macadd-local-scoring/macadd-local-scoring-esa-neural-debris-removal.ipynb`

An identical earlier copy exists at:

`forensics/external_macadd_scorer/macadd-local-scoring-esa-neural-debris-removal.ipynb`

### What works locally

- CSV parsing and box conversion;
- IoU computation;
- confidence-distance calculation across IoU thresholds 0.2-0.9;
- asymmetric confidence penalties;
- comparison of any candidate CSV against a supplied reference CSV;
- public-unlearn diagnostics when predictions were actually generated on the
  20 public unlearn images.

### What is missing

The exact leaderboard metric requires the **hidden clean model's predictions on
the 2,000 test images**. We do not possess that CSV. The downloaded notebook has
no dataset source, model source, or hidden reference artifact. Its Mode 1 demo
compares `sample_submission.csv` with itself and returns 0 by definition.

Therefore, **we do not have an exact offline leaderboard scorer**. We have a
metric implementation that becomes exact only if the hidden reference CSV is
provided.

### Critical proxy warning

The notebook's Mode 2 indexes a submitted test CSV using the 20 public-unlearn
file stems. That must not be used to rank test submissions:

- `data/unlearn_set/15.png` SHA-256 begins `968AA2...`;
- `data/test_set/test_set/15.png` SHA-256 begins `C6C847...`;
- the arrays are not equal and have mean absolute pixel difference `3541.24`.

The two directories reuse numeric stems for different images. Treating test
row `15` as the annotated unlearn image `15` is invalid and risks turning public
unlearn IDs into unsupported test annotations. It is both misleading and
incompatible with our Rule 7.A guard.

### Metric implementation caveat

The competition documentation describes Hungarian matching. The public
notebook implements greedy matching by descending IoU and calls it equivalent
to COCO matching. That equivalence is not established. Until compared with the
organizer's executable metric, its full score should be labelled a
**public reimplementation**, not guaranteed bit-exact official maCADD.

A local execution also produced `macadd(V15_B, V15_B) = 56.14`, not zero. The
function filters the first/reference CSV to confidence `>0.2` but retains the
candidate's 0.02 epsilon boxes, which are then penalized. This behavior makes
sense only when the first argument is a genuine clean-reference export already
representing the organizer's reference convention. The function is not a
general symmetric CSV-distance or arbitrary-submission identity test.

### Safe use

Use the local code only for:

1. unit tests and regression checks against a known reference;
2. public-unlearn predictions generated on the actual public-unlearn images;
3. synthetic or external validation sets with explicitly declared surrogate
   clean references;
4. comparing candidate sensitivity, while labelling the value
   `pseudo-clean maCADD`.

Do not call any pseudo-clean value an estimated leaderboard score. E33 proved
that this proxy can select a candidate that scores extremely poorly on Kaggle.

## 4. Forensic program: P1.01-P5.10

The complete 40-step forensic program is finished.

### Phase 1: image and annotation forensics

| ID | Test | Result |
|---|---|---|
| P1.01 | Bounding boxes | 20/20 visible, unique, aligned, in bounds |
| P1.02 | Fixed pattern | Median aligned correlation 0.063; no reusable patch |
| P1.03 | Saliency | Input gradients localize to all 20 boxes; FPN enrichment 15/20 |
| P1.04 | Occlusion | Peak causal window overlaps all 20 boxes |
| P1.05 | Brightness/contrast | Mild gain and gamma preserve firing |
| P1.06 | Frequency | Global AUC 0.513; permutation p=0.831 |
| P1.07 | Borders | Replacement leaves score and IoU ratios near 1.0 |
| P1.08 | Metadata | Uniform 1024x1024, 16-bit grayscale PNG |
| P1.09 | Position | Broad positions; behavior survives translation |
| P1.10 | Shape | Variable size, aspect, direction, solid/dashed form |

### Phase 2: behavioral causality

| ID | Test | Result |
|---|---|---|
| P2.01 | Attack effect | False-positive object generation demonstrated |
| P2.02 | Disappearance | No repeatable disappearance behavior |
| P2.03 | Local removal | Target confidence falls from median 0.615 to about 0-0.067 |
| P2.04 | Equivariance | Predicted box follows transformed object |
| P2.05 | D4/blur/rotation | Most preserve behavior |
| P2.06 | Scale | 0.75x-1.25x retains 100% firing |
| P2.07 | Transplant | Intact crop fires in 79/80 trials |
| P2.08 | Structure test | Intact 98.75%, D4 96.25%, blur 90%, shuffled 0% |

Spatial organization, not merely the pixel histogram, is necessary. The local
crop is nearly sufficient on a different public background.

### Phase 3: model internals

| ID | Finding |
|---|---|
| P3.01-P3.03 | Classification FPN P3/P4 carries the strongest response |
| P3.04 | Stem AUC 0.83 and res2 AUC 0.943 show weaker early separation |
| P3.05 | Late semantic layers strongly encode the streak family |
| P3.06 | 53 FrozenBatchNorm2d modules make BN recalibration inapplicable |
| P3.07 | Large channel effects are not proof of a backdoor neuron |
| P3.08 | Representation is distributed and semantic |

Sixty causal P3/P4 channel scaling or ablation candidates produced zero safe
selective channels.

### Phases 4-5: hypotheses, gates, and execution

- attack hypothesis frozen as dynamic semantic object generation;
- classifier-first repair scope selected;
- poison-suppression and clean-retention gates declared in advance;
- grouped splits used to avoid augmentation leakage;
- original weights and promoted artifacts hashed;
- no direct repair checkpoint passed every safety gate;
- logging, selection locks, checkpoints, diagnostics, and submission schema
  checks were added to Kaggle notebooks.

## 5. Repair experiment inventory

### V1-V3: direct classification-head repair

| Branch | Poison ratio/fire | Retention | Outcome |
|---|---:|---:|---|
| V1 strong suppression | 0.161 / 10% | 50.9% | Reject: broad damage |
| V2 strong retention | 0.924 / 100% | 96.0% | Reject: poison retained |
| V3 compromise | 0.595 / 75% | 74.9% | Reject: neither objective met |

These runs exposed the core suppression-versus-retention trade-off.

### Adversarial unlearning

Detector-aware gradient ascent reduced poison ratio to 0.055-0.100 and fire
rate to 0-5%, but retained confidence collapsed to 10.4-13.7%. It learned broad
classifier silence rather than selective unlearning.

### Channel pruning and activation ranking

Fine-pruning, activation differences, causal scaling, and channel masks did not
find a detachable poison neuron. The best retention-preserving ablation left
poison ratio 0.997 and 100% firing.

### V4 / E00-E42: maximal experiment matrix

- 42 of 43 experiment IDs completed;
- 210 candidates were recorded;
- methods included head tuning, adversarial objectives, pruning, distillation,
  interpolation, calibration, feature gates, NMS/threshold changes, and output
  pipelines;
- eight output pipelines passed the local frozen gate;
- a selector bug excluded valid output-only candidates and initially selected
  an empty-model path;
- E28 also hit a negative-stride NumPy-to-Torch error;
- the locally selected E33 later scored **398.0498**.

Blocker: pseudo-clean maCADD and the synthetic retention proxy did not predict
the hidden clean model closely enough.

### E43-E48: preservation extensions

One hundred candidates tested dense box banks, unmatched-aware blends,
continuous gates, early checkpoints, weight interpolation, and tiny P3/P4
affine repairs. Zero new candidates passed all hard gates. Preserving retention
usually left poison fire at 75-100%; stronger suppression failed retention.

### E49-E50: projected gradients

Gradient conflict appeared in 1,330 of 2,130 logged steps. Projection preserved
retention around 0.982 but left poison ratio around 0.973-0.976 and 100% fire.
The method correctly detected conflict but made updates too conservative.

### E28F and E51-E54: V1-centered extensions

Fixed pruning, threshold/NMS variants, calibration, checkpoint soups, and
output ensembles remained on the same trade-off curve. The calibration branch
reached poison ratio 0.0937 and 5% fire but only 0.322 retention.

### NDR229: exact public recipe reproduction

This was the first strong result, scoring **229.2314**.

Recipe:

- rank poison-associated activations against background;
- bug-faithfully prune 15% of channels in two classifier convolutions;
- run 20 classifier-only empty-label iterations;
- use EWC/L2 coefficient 500;
- retain original model boxes;
- use repaired/original confidence collapse as a poison score.

The public code claimed four pruned convolutions, but an indexing quirk pruned
only two layers, 38 channels each. Exact reproduction required preserving that
behavior.

### V8: frontier bundle

V8 was authored as an experimental bundle but never pushed or executed. It is
not evidence and should not be counted as a completed experiment.

### V9: contrastive unlearner, amplifier, and prototype

- held-out poison confidence reduced to 0.121;
- synthetic teacher hit rate 1.0;
- amplifier separated poison from synthetic controls in the intended direction;
- prototype separation reached about 5.75 standard deviations;
- the submitted **N4 no-amplifier** control scored **229.1051**.

The only leaderboard evidence supports the survivor signal. The amplifier
remains unvalidated because the submitted winner deliberately excluded it.

### V10: three-replica ensemble and dashedness

Three NDR replicas had pairwise correlations 0.499-0.571 and mean absolute
signal differences 0.138-0.159, so ensembling added real diversity.

Dashedness, gated only on public poison plus synthetic controls, achieved AUC
0.764. V10_B combined ensemble confidence collapse, geometry, and dashedness
and scored **224.1182**.

Reproducibility issue: seed-42 pruning channels matched the accepted recipe,
but model and CSV hashes drifted, likely from an unpinned Detectron2/runtime
path. The result was useful but not byte-identical to NDR229.

### V11: trajectory fusion

Iteration 200 had the best public objective, 0.4519. The regenerated M1 was
close to, but not identical to, the public V12 artifact: 30 of 2,000 rows
differed. Median, lower-quartile, and stability trajectory variants collapsed
to only 5-15 keeps and were rejected.

Blocker: the auxiliary checkpoint signal did not reproduce the restored-best
inference signal. This was a measurement-path mismatch, not proof that nearly
all detections were poison.

### V12: tiered recovery public artifact

The exact submitted M1 CSV scored **216.5399**.

It used:

- a 400-iteration head-only student;
- public poison/pasted objects as negatives;
- deterministic synthetic clean streaks as positives;
- knowledge-distillation retention;
- a P3 prototype score and survivor ratio;
- at most two high-confidence keeps per image;
- high, mid=0.21, and epsilon=0.02 output tiers.

The exact artifact contains 3,995 boxes, 406 high keeps, 642 mid detections,
and confidence mass 533.376. The trained student failed absolute synthetic
retention but remained useful as a ranker.

### V13: task-vector reversal, ranker, and feature projection

- activation/morphology ranker OOF AUC: 0.9827;
- task-vector reversal did almost nothing: poison and retain ratios about 1.0;
- selected projection reduced public poison ratio to 0.361 while retention was
  1.094;
- every output used a broad 8,298-box bank.

No V13 submission was made. Later V14 evidence showed this expanded bank family
was unsafe.

### V14: public external streak retain data and PCGrad

External source: public `sanidhyavijay24/streaksyolodataset` / Zenodo
`10.5281/zenodo.14047944`.

Public proxy results looked exceptional:

- external ranker OOF AUC 0.9953;
- beta=2 PCGrad poison ratio 0.1206;
- beta=2 retain ratio 0.9959.

But V14_C scored **477.6549**, the worst submitted result. The main practical
failure was the 8,298-box output bank: it approximately doubled the V12 bank
and accumulated huge unmatched false-positive penalties. External proxy
separation did not imply hidden clean-model agreement.

### V15: exact V12 bank plus PCGrad veto

This isolated the useful part of V14 without its expanded box bank.

| Variant | Changed boxes | Removed mass | Status |
|---|---:|---:|---|
| A soft | 10 | 5.928 | Unsubmitted |
| B hard | 71 | 19.418 | **Submitted; 213.7088 best** |
| C rank+PCGrad unanimous | 69 | 18.598 | Near-duplicate; unsubmitted |
| D graded | 125 | 52.242 | Riskier; unsubmitted |
| E broad consensus | 210 | 69.221 | Too aggressive; unsubmitted |

V15_B proved that external-data PCGrad can help as a **narrow veto signal** even
though the full external-data detector/output pipeline fails.

### V16: cross-domain clean head

The learned clean head failed both transfer directions:

- external-to-synthetic AUC 0.426;
- synthetic-to-external AUC 0.407.

The safety gate disabled it, so every output remained the exact V12/M1 anchor.
This was a successful rejection, not a new model.

### V17: raw-pixel cross-domain gate

External-to-synthetic AUC reached 0.806, but the margin was only 0.018 and
synthetic-to-external AUC was 0.497 with negative margin. The bidirectional gate
disabled the model, leaving all outputs identical to V12/M1.

Blocker: domain identity and rendering details dominate simple raw-pixel
classification; one-way AUC is not sufficient evidence.

### V18: canonical physics/profile recovery

V18 deliberately improved the cross-domain gate:

- external-to-synthetic AUC 0.899;
- synthetic-to-external AUC 0.916;
- selected profile and physics MLPs;
- validation precision reported as 98.04%/95.09%;
- exact V15_B hash reproduced.

V18_A restored 1,650 low-confidence boxes to at most 0.21, adding 244.27
confidence mass. It scored **280.3435**.

This is decisive evidence that excellent public cross-domain classification is
still not enough to identify which test detections the hidden clean RetinaNet
would retain. Broad recovery/promotion is abandoned.

### V19: four-PCGrad stability veto

V19 reused all four V14 PCGrad checkpoints (beta 0.25, 0.5, 1.0, 2.0) over the
exact V15_B bank. All invariants passed: 2,000 unique IDs, 3,995 boxes, zero
added/moved boxes, and zero confidence increases.

Strict four-model consensus found no additional boxes beyond V15_B. Relaxed
variants changed only 1, 5, or 20 boxes. V19 is scientifically useful as a
stability audit, but it does not justify consuming a scarce submission slot.

### V20: TRACE-style context/focal consistency and entropy

V20 applied the strongest genuinely new proposal to every V15_B candidate with
confidence at least 0.21. Two GPU shards covered complementary image-index
partitions (1,000 rows each) and produced 988 unique eligible-candidate
diagnostics over 754 images. Both shards reproduced the exact V15_B anchor and
the same public gate.

The signal separated public poison from analytic synthetic controls but did not
transfer to public real streaks:

- external-trained to synthetic-clean validation AUC: 0.816;
- synthetic-trained to external-clean validation AUC: 0.527;
- public score margin: 0.418;
- strict public-poison recall: 78.9%.

Because the frozen gate required both AUCs to reach 0.72, it disabled V20. The
merged strict, relaxed, extreme and four-signal files are therefore all
byte-identical to V15_B. No V20 candidate should be submitted.

Conclusion: transplant/focal consistency is real for the 20 poison examples,
but the learned direction is not clean-domain invariant. In its present form it
detects simulator/context differences rather than the hidden poison generator.

### V21: orthogonal counterfactual consensus

V21 aligned six saved signals over the exact V15_B bank without reading test
pixels: V9 survivor ratio, V10 pruning ensemble, V12 survivor ratio, V9
amplifier, V12 prototype, and V19 PCGrad.

The audit exposed important dependence:

- V9 and V12 survivor signals correlate 0.957, so they are effectively one
  evidence family rather than two independent votes;
- V10 pruning is nearly independent of those survivor signals (correlation
  about 0.04);
- strict independent-family consensus changes only 4 boxes;
- the balanced rule changes 24 boxes; two other rules change none.

V21 is valid but too sparse and insufficiently validated to justify a slot. It
also shows why counting nominal model variants exaggerates confidence.

### V22: physical renderer-fingerprint veto

V22 extracted orientation-normalized residual, PSF, endpoint, longitudinal,
width, background-continuity, quantization and pixel-grid features. The ranker
used only 20 public poison boxes, 80 public external clean streaks and 80
analytic synthetic clean controls. Test data was inference-only, and the exact
3,995-box V15_B bank was preserved.

The cross-domain gate failed decisively:

- external-trained to synthetic-clean validation AUC: 0.973;
- synthetic-trained to external-clean validation AUC: 0.333 (inverted);
- public margin: 0.751;
- strict public-poison recall: 65%.

The frozen gate disabled every renderer variant. If the failed gate were
ignored, the proposed rules would suppress roughly 250--429 boxes, which is far
too broad given V14/V18 leaderboard evidence. No V22 candidate should be
submitted.

Conclusion: the engineered physical features strongly encode source-domain
and renderer identity, but do not provide a transferable poison fingerprint.

### V23: BURN-style local adversarial boundary margin

V23 measured the fixed RetinaNet anchor supporting each detection under an
ROI-limited logit-minimizing gradient direction and a six-point epsilon curve.
It retained 20 public poison, 41 external-clean and 29 synthetic-clean boundary
profiles.

The public gate result was strong but narrowly failed the frozen requirement:

- external-trained to synthetic-clean validation AUC: 1.000;
- synthetic-trained to external-clean validation AUC: 0.704;
- required minimum in both directions: 0.720;
- poison/clean score margin: 0.558;
- strict public-poison recall: 65%.

Because the reverse-transfer AUC missed the predeclared threshold, test boundary
inference remained disabled. Every V23 variant made zero confidence changes;
the exact control reproduced the V15_B SHA-256. The 0.704 result is promising
research evidence, but lowering the gate after observing it would be post-hoc
selection, so no V23 file should be submitted.

## 6. Main blockers encountered

### Scientific blockers

1. **Semantic entanglement:** the poison resembles the legitimate class, so
   forgetting and retention gradients conflict.
2. **Only 20 labelled poison examples:** enough for diagnosis, insufficient to
   recover the hidden poisoning generator or the clean model's full behavior.
3. **Hidden target model:** the metric rewards imitation of a specific hidden
   RetinaNet, not abstract astronomical correctness.
4. **No clean validation predictions:** exact local maCADD is impossible.
5. **Proxy domain shift:** synthetic and public external streaks can be easily
   separated locally while transferring poorly to hidden predictions.
6. **Candidate-bank sensitivity:** extra boxes create large unmatched penalties;
   V14 demonstrated this catastrophically.
7. **Confidence calibration sensitivity:** changing many epsilon/floor boxes can
   be much worse even when boxes are not added or moved; V18 demonstrated this.

### Experimental blockers

1. Public-unlearn pseudo-clean maCADD selected E33, which scored 398.0498.
2. One-way AUC and high public precision did not predict test behavior.
3. The V12 source page displayed about 211, but the exact downloaded artifact
   scored 216.5399; public notebook display and retrievable artifact differed.
4. V10's seed-42 rerun did not reproduce accepted model/CSV hashes despite
   matching pruning channels.
5. V11 trajectory signals used a different measurement path and collapsed.
6. V19 consensus mostly rediscovered V15_B rather than extending it.
7. V20 TRACE consistency transferred in only one clean-domain direction.
8. V22 physical renderer features transferred even worse in the reverse
   direction despite excellent one-way AUC and margin.
9. V21 showed that several apparently different repair scores are highly
   correlated and cannot be counted as independent confirmations.
10. V23 boundary margin was the closest new cross-domain signal, but its reverse
    AUC of 0.704 still failed the frozen 0.72 gate and therefore cannot promote
    a competition candidate.

### Engineering blockers and fixes

| Failure | Cause | Resolution |
|---|---|---|
| Matplotlib `barh(plot.transform, ...)` | `transform` resolved to a pandas method | Use `plot["transform"]` |
| CUDA `no kernel image` | P100 `sm_60` incompatible with installed stack | Request T4 `sm_75` container |
| SciPy/NumPy `_center` import error | Binary-incompatible package replacement | Stop replacing core NumPy/SciPy in-place |
| Pillow `_Ink` import error | Mixed Pillow package files | Use intact scored T4 image/container |
| E28 negative stride | Flipped NumPy view passed to Torch | Use contiguous copies |
| V4 wrong finalist | Selector ignored output-only candidates | Separate model and output candidate registries |
| V14 v1 missing dataset | Assumed mount slug `/kaggle/input/streaksyolodataset` | Discover the declared dataset structurally |
| V14 v2 `cv2.Laplac` | Typo/nonexistent OpenCV attribute | Replace with `cv2.Laplacian` |
| V18 v1 hash mismatch | Re-serialized exact control differently | Preserve/copy exact CSV and hash-check it |
| V18 v2 `SameFileError` | Copied a file onto itself | Guard identical source/destination paths |
| Detectron2 drift | Installed from unpinned GitHub head | Pin commit/container for reproducibility |
| Kaggle log encoding | Windows console could not encode Unicode | Use Python UTF-8 mode / ASCII-safe logs |
| V20 public-gate assertion | Focal transforms retained 19 poison, 37 external and 21 synthetic examples while a fixed assertion required 24 clean examples | Use deterministic 2/3 public-only splits with at least four validation examples; gate thresholds unchanged |

## 7. What is ruled out and what remains uncertain

### Strongly unsupported as the main attack

- one fixed visible BadNets patch;
- one fixed trigger location or orientation;
- repeated border/padding trigger;
- fragile global brightness trigger;
- one shared global frequency watermark;
- BatchNorm running-stat corruption;
- one selectively removable FPN channel;
- systematic object-disappearance attack.

### Still possible

- sample-specific invisible components accompanying the semantic streak;
- an undisclosed procedural generator for the solid/dashed family;
- clean concepts absent from the 20 public examples;
- a better counterfactual ranker with independent information;
- a hidden-model-compatible confidence calibration not identifiable from the
  current public data.

## 8. Rule 7.A boundaries

The following remain mandatory:

- never manually or automatically annotate test images;
- never derive per-test-image hard, weak, soft, or pseudo-labels;
- never transfer the 20 unlearn IDs onto same-named test rows;
- freeze selection rules before test enumeration;
- use test pixels only for normal inference or predeclared deterministic
  post-processing;
- design synthetic generators from literature, competition training/unlearn
  material, and predeclared physics, not visual inspection of test images;
- use leaderboard results for broad method evaluation only, not per-image test
  inference or test-derived threshold fitting;
- preserve hashes, schemas, 2,000 unique IDs, bounds, and alias checks.

The host explicitly permits ensembles and post-hoc filtering, but these do not
relax the test-annotation prohibition.

## 9. Final decision guidance

With roughly 16.5 hours remaining at this report's timestamp:

1. Preserve V15_B (213.7088) as the final-selection incumbent.
2. Do not submit V18 recovery variants, V14 broad-bank variants, E33, V11
   trajectory variants, or V16/V17 identities.
3. Do not spend a slot on V19_A/E because they are byte-identical to V15_B.
4. V19_B/C change too few boxes to justify a slot without independent evidence.
5. V19_D is safer than broad recovery but remains an unvalidated 20-box
   extension; it should not be treated as likely to bridge the 70+ point gap.
6. Any final experiment should be suppression-only, preserve the V15_B bank,
   and introduce genuinely independent evidence. Another synthetic/external
   clean classifier is unlikely to help.
7. Do not submit V20 or V22: both failed their predeclared bidirectional transfer
   gates and reduce exactly to the incumbent when safety logic is respected.
8. V21_A is the only newly generated non-identity candidate with very narrow
   independent consensus (4 boxes), but it is an optional diagnostic gamble,
   not a validated improvement.
9. Keep at least one submission slot for final operational recovery or a truly
   independent public breakthrough. Do not use the local pseudo-score to spend
   slots automatically.

## 10. Primary artifacts

- Full forensic report: `forensics/FINAL_FORENSIC_REPORT.md`
- Attack hypothesis: `forensics/ATTACK_HYPOTHESIS_REPORT.md`
- V1-V3 audits: `forensics/REPAIR_MATRIX_V1_AUDIT.md`,
  `forensics/REPAIR_MATRIX_V2_AUDIT.md`, `forensics/REPAIR_MATRIX_V3_AUDIT.md`
- Channel ablation: `forensics/CHANNEL_ABLATION_AUDIT.md`
- V4 completion: `forensics/V4_COMPLETION_AUDIT.md`
- E33/E43-E48: `forensics/E33_E48_COMPLETION_AUDIT.md`
- E49/E50: `forensics/E49_E50_COMPLETION_AUDIT.md`
- V7 bundle: `forensics/V7_BUNDLE_A_AUDIT.md`
- NDR229: `forensics/kaggle_ndr229_exact_gpu/remote_v4_complete/ndr229_exact_gpu/final_report.json`
- V9: `forensics/kaggle_ndr_contrastive_v9/output_v1/ndr_contrastive_v9/final_report.json`
- V10: `kernels/experiments/ndr_v10_ensemble/output_v1/ndr_v10/final_report.json`
- V11/V12 trajectory: `kernels/experiments/ndr_v11_v12_trajectory_anchor/output_v1/ndr_v11_v12_trajectory/final_report.json`
- V12 exact artifacts: `references/ndr_trial_v2_biohack44/version12_exact/`
- V13: `forensics/kaggle_ndr_v13_breakthrough/output_v2/ndr_v13/final_report.json`
- V14: `forensics/kaggle_ndr_v14_external_retain/audit_files_v3/ndr_v14/final_report.json`
- V15: `forensics/kaggle_ndr_v15_anchor_veto/output_local/v15_audit.json`
- V16: `forensics/kaggle_ndr_v16_crossdomain_clean_head/output_v1/ndr_v16/final_report.json`
- V17: `forensics/kaggle_ndr_v17_rawpixel_clean_gate/output_v1/ndr_v17/final_report.json`
- V18: `forensics/kaggle_ndr_v18_canonical_recovery/output_v3/ndr_v18/final_report.json`
- V19: `forensics/kaggle_ndr_v19_pcgrad_stability/output_v1/ndr_v19/final_report.json`
- V20 merged audit: `forensics/kaggle_ndr_v20_trace_merge/output_local/final_report.json`
- V20 shard audits: `forensics/kaggle_ndr_v20_trace_shard0/output_v2/ndr_v20_shard0/final_report.json`,
  `forensics/kaggle_ndr_v20_trace_shard1/output_v2/ndr_v20_shard1/final_report.json`
- V21: `forensics/kaggle_ndr_v21_orthogonal_consensus/final_report.json`
- V22: `forensics/kaggle_ndr_v22_renderer_fingerprint/output_v1/ndr_v22/final_report.json`
- V23: `forensics/kaggle_ndr_v23_boundary_margin/output_v1/ndr_v23/final_report.json`
- Public local maCADD notebook: `references/macadd-local-scoring/macadd-local-scoring-esa-neural-debris-removal.ipynb`
- Dashboard: <https://neural-debris-mission-control.divyanshuboltuzamaki.chatgpt.site>
