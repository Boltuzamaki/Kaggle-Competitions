# EDA Summary - Neural Debris Removal in Streak Detection Models

Findings below come from directly inspecting the downloaded competition data (`data/`), cross-checked
against the official competition Overview text (see `README.md` for the full quoted rules/metric). A
runnable version of this analysis is in `eda/EDA.ipynb`. Reproduction commands are inline Python
snippets; re-run against your own `data/` copy to verify.

## 1. Dataset inventory

Downloaded via `kaggle competitions download -c neural-debris-removal-in-streak-detection-models`
(single zip, 3.5 GB compressed).

| Path | Count | Notes |
|---|---|---|
| `poisoned_model/poisoned_model.pth` | 1 file | 145,548,668 bytes. Torch checkpoint. |
| `test_set/test_set/*.png` | **2,000** | 1024×1024, 16-bit grayscale. IDs `0.png`-`1999.png` (dense, no gaps). |
| `unlearn_set/*.png` | 20 | Same format as test set. IDs are a subset of the numeric range 0-1999 but are **different underlying images** (see §5). |
| `unlearn_set/annotations_coco.json` | 1 file | COCO-format ground truth for the 20 unlearn images. |
| `sample_submission.csv` | 2,000 rows + header | Output-format template. |

Note: an early pass using the Kaggle API's paginated `competitions files` listing initially suggested
~3,800 test images due to a pagination/counting artifact. The extracted ground truth is authoritative:
**2,000 test images**, matching `sample_submission.csv` row-for-row with no decoy/extra images.

## 2. Image properties

Checked on a random sample (n=30 for size/mode, n=60 for pixel stats, seeded for reproducibility):

- **Format**: PNG, single channel, mode `I;16` (16-bit unsigned grayscale), size **1024×1024**, 100% consistent across the sample - no size or mode outliers found.
- **Pixel statistics** (per-image, uint16 raw values):
  - min: ranges 0-3,061 across images (mean ~ 1,007)
  - max: **every single sampled image saturates at exactly 65,535** - i.e. every frame has at least one fully clipped pixel (bright star core and/or streak core hitting sensor ceiling)
  - mean: tightly clustered ~ 11,650 (± ~50 across images) - consistent exposure/calibration across the dataset
  - std: ~ 5,100 (± ~180)
- **Visual content** (percentile-stretched 1st-99.5th, see `eda/sample_renders/`): convincing star-field scenes - dense point-source stars, diffuse background structure (nebulosity/zodiacal-light-like gradients), photon/read noise texture, and **thin, faint, roughly-linear streaks** representing the space-debris/satellite trails to detect. **Confirmed by the official Overview: these images are synthetic**, generated for this competition by the ESA SYNDAQ (Synthetic Data Generation & Qualification) project (Solenix, Telespazio, Fondazione Bruno Kessler) - not real telescope captures, though built to realistically emulate them. This matters for the poisoning question: any "trigger" doesn't have to be a pasted patch, it could equally be a specific synthetic-generation signature (e.g. a particular streak shape/brightness/noise combination the generator produces) that the poisoned model was trained to react to.

## 3. Streak / box statistics (from `sample_submission.csv`)

The template file's `prediction_string` values were parsed as repeating `(confidence, x, y, width, height)` tuples:

- 2,000 images -> **2,593 total boxes**
- Boxes per image: min 0, max 8, **mean 1.30, median 1**
- Distribution: 634 images w/ 0 boxes · 637 w/ 1 · 407 w/ 2 · 199 w/ 3 · 82 w/ 4 · 33 w/ 5 · 5 w/ 6 · 2 w/ 7 · 1 w/ 8
- Confidence: range **0.200-0.984**, mean 0.54. The hard floor at exactly 0.200 strongly suggests this template was produced by thresholding a real detector's output at `conf >= 0.2` rather than being random filler - **now confirmed** by the official Evaluation section: "the clean model detections contain only outputs with confidence > 0.2". This is very likely a reference-model output (poisoned and/or clean), not arbitrary placeholder data.
- Box width: 7.0-148.2 px (mean 34.3); box height: 7.2-117.1 px (mean 36.2). Boxes are small relative to the 1024 px frame (~3-4% of frame width) - tight crops around streak segments, not full-diagonal trails.
- x, y coordinates range 0-~1012 -> confirms **absolute pixel coordinates** in the 1024×1024 frame (not normalized 0-1), with `(x, y)` as the **top-left corner** of the box (confirmed in §3.1).

### 3.1 Cross-check against real image content

For `image_id=10`, the 4 predicted boxes in `sample_submission.csv` were manually compared against the
percentile-stretched render of `test_set/test_set/10.png`. All four box locations line up with visually
plausible streak positions (a mark in the top-left corner, a vertical mark on the left edge roughly
2/3 down the frame, a diagonal streak mid-right, and a small mark top-right). This confirms:
- The coordinate convention is top-left-anchored absolute pixel boxes.
- The "sample" submission isn't pure noise - it looks like genuine (if imperfect/poisoned) model output.

## 4. Model architecture (`poisoned_model.pth`)

Loaded with `torch.load(..., weights_only=False)`:

- Checkpoint shape: `{'model': OrderedDict}` - a standard **Detectron2** checkpoint, no other top-level keys (no optimizer state, no training config/anchor-generator config bundled in the file).
- **Backbone**: ResNet-50 (stem + res2..res5 bottleneck blocks with GroupNorm/BN-style `.norm.*` params) + **Feature Pyramid Network** (`fpn_lateral3-5`, `fpn_output3-5`) + extra top-level blocks `top_block.p6`, `top_block.p7`. This is the standard Detectron2 `build_retinanet_resnet_fpn_backbone`.
- **Head**: 4-conv shared `cls_subnet` and `bbox_subnet` (256 channels each) -> final `cls_score` (7 output channels) and `bbox_pred` (28 output channels).
- 301 total parameter tensors.
- **Decoding the head shape**: `bbox_pred` = 28 = 7 anchors × 4 box-regression values; `cls_score` = 7 = 7 anchors × 1 class ⇒ this is a **single-class detector** ("object" = debris streak), **7 anchors per spatial location**. No multi-class ambiguity to worry about.
- **No accompanying config file** (no `.yaml`/`.json` alongside the checkpoint) - exact anchor scales/ratios, input normalization (pixel mean/std), input resize policy, and score/NMS thresholds used at training time are **not recoverable from the checkpoint alone** and would need to be assumed/reverse-engineered (standard Detectron2 RetinaNet defaults are a reasonable starting guess).

### 4.1 Head-level anomaly check (naive backdoor probe)

Per-anchor bias and weight-norm of the two final head layers were compared across all 7 anchors:

- `cls_score` bias: all 7 values cluster tightly around **−4.58** (range −4.589 to −4.582) - no single anchor stands out.
- `cls_score` weight norm per anchor: 0.51-0.54, all similar.
- `bbox_pred` bias/weight norms: also unremarkable and similar across anchors/coordinates.

**No anchor-level outlier was found.** This is not strong evidence of "no poisoning" - it just means the
poison isn't implemented as an obviously-biased single output neuron. Real backdoors are typically
distributed across many backbone filters and only activate given a specific trigger input, so this
check was expected to come back clean either way.

### 4.2 Trigger-patch search (image-space)

Checked whether a fixed/constant visual "watermark" trigger sits in a corner of the input images
(a common simple backdoor design): sampled 32×32 patches from all four corners across 200 random
test images and measured **per-pixel variance across images**. Variance was high everywhere
(~10M-90M in raw 16-bit units) - i.e. corner content varies normally from image to image. **No fixed
corner-patch trigger was found.** If a backdoor trigger exists, it is not a simple constant watermark
in the image corners; it could be located elsewhere in the frame, be trigger-in-context (specific
streak geometry/brightness), or be purely a training-time/weight-space corruption with no
input-space trigger at all.

### 4.3 Inference not yet run

`detectron2` is not installed in this environment (non-trivial to build on Windows). No forward pass
has been executed, so no actual model outputs have been inspected yet. **Recommended next step:**
install detectron2 (or reimplement the anchor-generation + box-decoding logic manually against this
state dict), run the model over `unlearn_set` and compare predictions against
`unlearn_set/annotations_coco.json` ground truth - large, systematic errors there are exactly what
you'd expect the poisoned examples to expose, and would be the fastest way to characterize what the
poison actually does. See `eda/EDA.ipynb` for a `.venv` set up to do this.

## 5. The `unlearn_set` - confirmed to be the key input for this competition

- 20 images + `annotations_coco.json` (COCO format: `images`, `annotations`, `categories`), 1 category
  (`"object"`), exactly 1 ground-truth box per image.
- Filenames (`15.png`, `104.png`, `255.png`, ... `938.png`) numerically overlap with IDs already used
  in `test_set/`, **but pixel-diffing confirmed they are different images** (~99.66% of pixels differ,
  diff magnitude up to ±60,000 in 16-bit units) - the shared numbers are coincidental re-use of the same
  0-1999 ID space, not the same underlying frames.
- **Confirmed by the official Overview**: this is exactly the "public unlearn set" of **poisoned
  training examples** - the set the poisoned model was corrupted on. The official baseline fine-tunes
  the poisoned model on these 20 images for 20 epochs (lr=0.0001, batch size 4), assigning them **empty
  ("no object") annotations** during that fine-tune - i.e. actively teaching the model to stop
  detecting whatever it currently (wrongly) fires on there. Note this baseline's own label choice
  (empty) does **not** necessarily match the real ground-truth boxes we found in
  `annotations_coco.json` (1 real box per image) - those look like genuine/clean streak locations, so a
  naive "everything here is empty" fine-tune plausibly **oversuppresses real streaks** at those
  locations too. That's exactly the kind of wrong-direction correction the asymmetric penalty in
  aCADD (§ below) is designed to punish.

## 6. Evaluation metric - maCADD - and what it means for strategy

The official Evaluation section (quoted in full in `README.md`) confirms this is **not** scored as
standard object-detection accuracy against raw ground truth. Instead:

- There is a **hidden clean RetinaNet model** (never shared) that represents the "correct" de-poisoned
  behavior.
- Your submission's detections are compared directly to **that hidden model's detections** (not to the
  `unlearn_set` annotations, and not to any absolute notion of "true" streaks) via Hungarian-matching +
  confidence-difference distance (CADD), computed at IoU thresholds 0.2-0.9 and mean-combined (maCADD).
- The distance is **asymmetric (aCADD)**: a de-poisoned model that lowers confidence on
  poison-triggered detections (moving toward the clean model) is penalized 10× less than one that
  raises it; symmetric logic applies to genuine/clean streaks. Both directions are still penalized -
  there's no free lunch from just suppressing everything.
- Practical implication: **the target you're actually chasing is a specific hidden model's output
  distribution**, not "detect all real streaks correctly." This reframes the task as closer to
  model distillation/behavioral-matching than to classic detection accuracy, and it means the
  `unlearn_set` ground-truth boxes are a *proxy signal* for the clean model's likely behavior on
  poisoned examples, not the actual scoring target.

## 7. Open questions / suggested next steps

1. **Run inference.** Stand up detectron2 (or a minimal from-scratch RetinaNet decode) and get raw
   predictions from `poisoned_model.pth` on `unlearn_set`, compare to the provided ground truth -
   quantify exactly how/where the poisoned model is wrong (false positives at those locations? wrong
   confidence? wrong box shape?).
2. **Characterize the poison.** Once inference works: check whether errors on `unlearn_set` correlate
   with box size, position, streak brightness, or something else systematic (rather than a fixed image
   patch, which §4.2 ruled out at the corners - consistent with the data being synthetic per §2).
3. **Implement maCADD locally** for offline validation before burning submissions - Hungarian matching
   + the asymmetric penalty + the IoU threshold sweep (0.2-0.9) are all specified precisely enough in
   the Overview to reproduce.
4. **Attempt de-poisoning/unlearning** using `unlearn_set`, but consider alternatives to the naive
   "label everything empty" baseline given the oversuppression risk noted in §5 - e.g. targeted
   fine-tuning that only suppresses the specific poisoned behavior, fine-pruning, or influence-function
   approaches.
5. **Generate real predictions** on `test_set` with the repaired model, validate the output format
   (remember: whitespace `" "` not empty string for no-detection rows) against `sample_submission.csv`,
   and sanity-check your local maCADD estimate against the three official reference notebooks
   (fine-tuning baseline / poisoned-model-as-is / empty-predictions) before submitting.
6. Still worth checking the Rules tab directly for submission-frequency limits and prize eligibility
   specifics not covered in the Overview text.

## Appendix: rendered sample previews

Saved to `eda/sample_renders/` (percentile 1-99.5% linear stretch to 8-bit, for viewing 16-bit source
images in a normal viewer):

- `0_stretched.png`, `1_stretched.png`, `10_stretched.png`, `100_stretched.png` - from `test_set/`
- `unlearn_15_stretched.png`, `unlearn_255_stretched.png` - from `unlearn_set/`
