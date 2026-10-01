# Experiment log

Append-only record of everything tried, **including what failed**. The negative results are the
most valuable part of this file: they stop a future session from re-running a dead end.

**Convention.** Newest session at the bottom. Every entry states the change, the measurement,
and the verdict. Never delete an entry - if something is later proven wrong, append a
correction rather than editing history.

**Numbers.** "local CV" = `scripts/run_local_cv.py` on the placeholder test videos that have
ground truth, scored with the organizers' own metric via `src/local_eval.py`. Unless stated
otherwise it is **n=2 videos with ~50 GT edges each** - a single edge moves the Jaccard by
~0.02, so differences below ~0.01 are noise.

---

## Session 1 - 2026-08-06

### Reconnaissance

| Finding | Evidence |
|---|---|
| Code competition (`isKernelsSubmissionsOnly=true`), 5 subs/day, max team 5, deadline 2026-09-29 | Kaggle API `competitions/list` |
| 199 train videos (`.zarr` + `.geff`, 85.7 GB), 4 test videos (1.9 GB), 87.6 GB total | full paginated file listing |
| Two embryos only: `44b6` (71), `6bba` (128) | dataset name prefixes |
| Volumes are `(T,Z,Y,X) = (100,64,256,256)` uint16, one blosc2 chunk per timepoint | `0/zarr.json` |
| Per-dataset intensity quantiles ship inside `{name}.zarr/zarr.json` | available at test time too - used for normalisation |
| **The 4 visible test videos are byte-identical copies of 4 training videos** | all 100 chunk sizes match exactly; GT for them sits in `train/` |
| GT is ~50 nodes/video against `estimated_number_of_nodes` ≈ 25k  **0.20% annotated** | `read_geff` on `44b6_0113de3b` |
| Every GT edge spans exactly Δt = 1 | `np.unique(dt) == [1]` |
| Only **5 annotated divisions across 65 videos** | scan of downloaded `.geff` files |
| GT frame-to-frame displacement: p50 1.72 µm, **p99 7.20 µm**, p100 11.06 µm | 5,454 annotated steps |
| Cell density varies hugely: **74 to 786 cells/frame** | `estimated_number_of_nodes` across videos | ### The metric - two structural facts

1. **Most predictions are invisible.** In the organizers' `metrics.py`, a predicted edge is only
   eligible to be a FP if an endpoint matches an annotated GT node
   (`pred_valid = out_valid | in_valid`). Edges among unannotated cells are neither TP nor FP.
2. **Under-predicting nodes is rewarded.** `adj = max(0, J·(1 − 0.1·(N_pred−N_true)/N_true))`.
   Scoring GT against itself yields **1.0998**, not 1.0 (`local_eval.oracle_upper_bound`).
   Ceiling with divisions ≈ 1.2; leaderboard leader is 0.949.

### Experiments

| # | Change | Result | Verdict |
|---|---|---|---|
| 1 | Initial classical pipeline, `max_link_um=8`, `min_track_len=6` | 0.7955 (n=1) | baseline |
| 2 | **Ablation** on `44b6_0113de3b`: linking only  +gap closing  +short-track pruning | 0.715  **0.829**  0.831 | gap closing is the single biggest post-processing win (+0.115) |
| 3 | `min_track_len` 6  4 | 0.7955  0.831 | 6 deleted a real lineage; 4 is safer |
| 4 | Linking gate sweep 3-10 µm | best at 7.0 (= measured p99) | 10 µm cost ~0.09 via identity swaps |
| 5 | DoG scales (2.0,6.0),(3.0,9.0)  **(1.5,4.5),(2.5,7.5)** | node recall 0.33  0.67 on the hard video | nuclei are ~5-8 µm, not 15; big kernels merge neighbours |
| 6 | 2-dataset CV with the above | 0.6171  **0.6973** | |
| 7 | **162-config joint sweep** (`scripts/sweep.py`) | best 0.7329 | `min_distance_um=3.5` is the one large effect: 0.733 vs 0.671 (5.5), 0.656 (4.5), consistent across all linking settings. `rel_threshold=0.02` best of {0.01,0.02,0.04} but margin small |
| 8 | Disable geometric `safe_divisions` | 0.7329  **0.7395** | produced 0 division TPs and 1 FP. With only 5 annotated divisions in 65 videos a geometric rule has nothing to find. **Now off by default** | ### Negative results - do not retry without a new idea

* **Adaptive contrast-invariant detection threshold.** Motivated by the real 74-786 cells/frame
  spread. Implemented as `threshold = rel_threshold × percentile(dog[dog>0], p)`; tried
  `p ∈ {99.0, 99.5, 99.9}` × `rel ∈ {0.15, 0.3, 0.5, 0.7}`. **No setting improved node recall**
  over the absolute cut on either measurable video. Survives as `Config.adaptive_percentile`,
  defaulting to `None`. A per-*dataset* calibration targeting an estimated cell count is a
  different and still-untested idea.
* **`xy_downsample=2` instead of 4.** Finer in XY but leaves the anisotropy that distorts the
  DoG kernels and the NMS footprint. Node recall 0.650 vs 0.750; median localisation error 4.56
  vs 3.86 µm. **4 is correct** - it makes the grid isotropic at 1.625 µm.
* **Centroid refinement is near-neutral on the hard video.** ds4 refine vs raw: recall
  identical (0.750), median distance 3.86 vs 3.41 µm. Kept because it clearly helps on
  `44b6_0113de3b` (median error 0.64 µm) and costs ~2% of runtime.

### Leaderboard

| submission | config | local CV | public LB |
|---|---|---|---|
| v2 | `rel_threshold=0.01`, divisions on | 0.697 | 0.785 |
| v5 | `rel_threshold=0.02`, `min_distance_um=3.5`, divisions off | 0.7395 | **0.824** |
| | delta | +0.042 | +0.039 | **Calibration (important for every future session):** the LB sits **~0.085 above** local CV, but
**deltas transfer almost exactly**. Use local CV to *rank* configurations; never quote it as a
score estimate.

### Per-dataset diagnostics worth remembering

Peaks/frame on the four test videos under the final config: **270, 369, 62, 671**.
`6bba_05b6850b` is genuinely sparse, not a detector failure - the true density range confirms it.

`44b6_0b24845f` is the hard one: node recall 0.82 vs 1.00, and detections carry a **systematic
−2.1 µm bias in Z** that a fixed band-pass cannot correct. This is the clearest single argument
for the learned detector.

### Environment gotchas (cost real time)

* **Kaggle API rate-limits hard (HTTP 429)** when fetching thousands of small files. Two
  `zarr.json` files stayed unreachable for over an hour, which is why local CV is n=2 and not
  n=4. Prefer bulk download.
* **Code-competition submissions need an explicit `-v N`**, and the CLI cannot list versions.
  `kaggle_ops.py push` now records the number it prints in `.kaggle_versions.json`.
* **Kaggle derives the notebook URL slug from the *title*, not the `id`.** If they disagree it
  warns and silently uses the title.
* **A notebook's `%%writefile biohub_ct.py` cell writes into the CWD.** Running a notebook
  locally from the repo root left a stale copy there that **shadowed `src/`** for anything
  launched with `python -c` / `python -`, silently producing wrong measurements once. Both
  names are now in `.gitignore`; scripts insert `src/` at `sys.path[0]` explicitly.
* **Kaggle discussion pages render client-side** and cannot be fetched programmatically. The
  public-solution survey in `docs/public_solutions.md` is built from downloaded notebooks plus
  search snippets, not the threads themselves.

### Bug found by the first training run

`biohub-03-train-unet` **v1 failed** after ~1 min with
`AttributeError: module 'biohub_unet' has no attribute 'index_training_frames'`.

Cause: `nbbuild.library_cell(target)` read a single hard-coded source path
(`src/biohub_ct.py`) and only varied the *output* filename - so the training notebook wrote
`biohub_ct.py`'s contents into `biohub_unet.py`. Both notebooks that inline two modules (03 and
04) were affected.

Fix: `library_cell(target, source=None)` now resolves `src/<source or target>`, and `build()`
asserts every `%%writefile` cell byte-matches its `src/` module before writing the notebook.
A mismatch used to surface only as a runtime error minutes into a paid GPU run.

**Lesson for future sessions:** verify the *content* of generated artifacts, not just that
generation succeeded. The syntax check passed happily - the file was valid Python, just the
wrong file.

### Second training failure - `zarr` on Kaggle cannot read the ground truth

**v2 failed** with `ValueError: num_samples should be a positive integer value, but got 0`,
preceded by `annotated points: 0 train`. The DataLoader error was a symptom; the cause was that
**every one of the 199 `.geff` reads failed** and `index_training_frames` swallowed the
exceptions with a bare `except Exception: continue`.

Root cause: `read_geff` used the `zarr` package. **Correction, confirmed later by the smoke
test: `zarr` is not installed on Kaggle at all** - nor is `numcodecs`. (My first guess was
"zarr 2.x cannot read v3"; the truth is simpler and worse.) The *image* data was unaffected
because it is read manually with blosc2 on the raw chunk - the same trick the official starter
notebook uses, and for the same reason.

Kaggle package reality, measured (`00_smoke_test`, 2026-08-06):
`numpy 2.0.2`, `scipy 1.16.3`, `torch 2.10.0`, `blosc2 4.1.2`, `zstandard 0.25.0`,
**`zarr` MISSING, `numcodecs` MISSING, `pyzstd` MISSING**.

This is why the decompressor tries *four* zstd backends. `numcodecs` - the obvious first
choice, and the only one I would have written if I had not been guessing - is absent;
`zstandard` is what actually carries the geff reads.

Fix, in `src/biohub_ct.py`:

* `_read_zarr_v3_array()` - reads a single-chunk Zarr v3 array straight from `zarr.json` +
  the `c/0[/0]` chunk file, no `zarr` import.
* `_decompress()` - geff arrays are `bytes` + `zstd`; tries `numcodecs`, `zstandard`, `pyzstd`,
  then the 3.14 stdlib, and raises listing what it tried rather than failing opaquely.
* `index_training_frames` now **reports** skipped datasets and raises if the sample set is
  empty, instead of handing an empty dataset to the DataLoader.

Verified byte-identical to the `zarr`-package read on `44b6_0113de3b`, and local CV is unchanged
at 0.7395 (behaviour-preserving).

**Lesson:** `except Exception: continue` over a whole dataset directory turns a total failure
into a silent empty result. Any loop that skips bad inputs must count and report what it
skipped. Also: assume nothing about Kaggle package versions - the image predates Zarr v3.

### Third training failure - P100 has no CUDA kernels in Kaggle's torch build

**v3 got all the way to the first forward pass** (data loading fixed: 16,682 train / 2,251 val
annotated frames, 118,649 annotated points) and then died:

```
AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

Kaggle assigned a **Tesla P100** (compute capability 6.0) and its `torch 2.10` build ships no
`sm_60` kernels, so every CUDA op fails. Nothing to do with this code.

Fix: set `"machine_shape": "NvidiaTeslaT4"` in the kernel metadata. T4 is `sm_75` and is what
**every** strong public notebook in this competition uses - visible in the
`kernel-metadata.json` of each one downloaded into `public_kernels/`. `kaggle_ops.py push --gpu`
now sets this by default (`--machine-shape` to override).

**Lesson:** when a Kaggle GPU run fails inside a framework call, check the *accelerator type*
before suspecting the code. And read the metadata of public notebooks that work - the answer
was sitting in files already on disk.

### Testing added after these failures

Three runs were burned on environment problems, so:

* **`notebooks/00_smoke_test.ipynb`** - ~2 min, CPU, validates the whole stack on Kaggle:
  package inventory, module API (catches the run-1 bug), image read speed, geff reads across 20
  files (catches run-2), training index non-empty, three real training steps, both detectors,
  an end-to-end submission write + validate, and a runtime extrapolation.
  **Run this before any long push.**  passes as of 2026-08-06.
* **`tests/test_biohub.py`** - 27 local unit tests on synthetic fixtures, no data or network
  needed. Includes explicit regression tests for both failures: one asserts every notebook's
  `%%writefile` cell matches its `src/` module, the other reads a synthetic `.geff` with
  `import zarr` monkeypatched to raise.

The division of labour matters: every failure this repo actually hit was an *environment*
problem that local unit tests cannot see. Unit tests guard logic; the smoke test guards the
platform.

### In flight at end of session

* `biohub-03-train-unet` **v4** running on Kaggle **T4**. Saves the best checkpoint by
  validation node recall @ 7 µm; `TIME_BUDGET_S = 6.5 h` stops it cleanly.
  Split: 199 datasets  175 train / 24 val, stratified by embryo.
  Scale: 16,682 annotated training frames, 2,085 steps/epoch at batch 8.

---

## Session 2 - 2026-08-07

### The trained detector lost to the classical baseline - checkpoint-selection bug

Training v4 completed on a T4: 24 epochs, 241 min, loss 0.120  0.046. But the saved checkpoint
is from **epoch 1**, because validation recall hit **1.000 in epoch 1** and never moved, so
"save the best" never fired again.

Scored on the two local videos with the official metric:

| read-out | score |
|---|---|
| classical DoG (LB 0.824) | **0.7395** |
| U-Net, threshold 0.3 / 0.5 / 0.7 / 0.9 | 0.164 / 0.166 / 0.167 / 0.169 |
| U-Net, top-K = 200 / 260 / **320** / 400 / 500 | 0.243 / 0.307 / **0.349** / 0.237 / 0.204 | **Not submitted.** LB stays at 0.824 from the classical pipeline.

#### Why thresholding is the wrong read-out (worth internalising)

The detection loss gives positives a *total* weight of 1.0 and all negatives combined 0.1, so
the network is pushed 10× harder toward "cell". The heatmap is therefore a **ranking, not a
probability**. Measured on the trained model:

```
heatmap median 0.13, mean 0.23;  14.5% of voxels > 0.5,  7.2% > 0.9,  3.2% > 0.99
peaks/frame:  thr0.5  813    thr0.999  620     (true density ~258-328)
even at NMS radius 8 µm and thr 0.999: 556 peaks
```

Raising the threshold barely helps because the plateaus are broad. **Take the top-K peaks by
score instead.** That alone doubled the score (0.17  0.35), though the epoch-1 model is too
undertrained to be competitive regardless.

#### Fixes applied

* `validate_topk_recall()` caps peaks at each video's own `estimated_number_of_nodes / T`, so a
  diffuse heatmap cannot score well by brute force.
* **Recall alone still did not discriminate** - with only ~16 GT points a random-init model
  scored the same 0.688 as the trained one. Median localisation distance did separate them
  (10.49 µm vs 2.08 µm), so selection is now
  `recall + 0.1·(1 − min(median_dist, 7)/7)` - measured 0.375 vs 0.758 on that pair.
* Validation widened to 24 datasets × 16 frames.
* **`unet_detector_last.pt` is always saved** alongside the best, so a saturating metric can
  never again discard an entire run.
* `detect_frames_unet(budget_from_dog=True)` sets K per frame from the classical detector's
  count, which tracks true density well (270 vs 258, 369 vs 328) and adapts to the 74-786
  cells/frame spread.

**Lesson:** a selection metric that saturates is worse than none - it silently discards the
run while reporting success. Always check that the metric separates a *random-init* model from
a trained one, and always keep the last checkpoint.

### The "second Kaggle account" does not exist

GPU quota on `boltuzamaki` ran out. `~/.kaggle-profiles/` holds two profiles, but:

| profile | `credentials.json` label | actually authenticates as |
|---|---|---|
| `boltuzamaki` | boltuzamaki | boltuzamaki |
| `divyanshuboltuzamaki` | divyanshuboltuzamaki | **boltuzamaki** | Both tokens belong to the same account; `~/.kaggle-uza/` is empty. Pushing with an `id` of
`divyanshuboltuzamaki/...` while authenticated as `boltuzamaki` fails with a bare
**`409 Conflict`**, which is a uselessly opaque way to say "wrong owner".

`kaggle_ops.kaggle_username()` now resolves identity from `kaggle config view` (the authenticated
truth) instead of the `username` field in credentials.json (a stale label). `--profile` selects
a profile dir via `KAGGLE_CONFIG_DIR`.

Net effect: **no additional GPU quota is available on this machine.** Real credentials for a
different account would have to be added as a new profile directory.

### CORRECTION - detection, not linking, is the bottleneck

Sessions 1 and 2 both asserted "node recall is 0.91 while edge Jaccard is 0.75, so the loss is
in *linking*, and a learned edge scorer is the highest-value next step." **That was wrong**, and
it was an inference from two aggregate numbers rather than a measurement.

`scripts/error_attribution.py` walks every GT edge and classifies why it was missed:

| cause | count | % of GT edges |
|---|---|---|
| TP | 82 | 82.8% |
| **detection_miss_target** | 5 | 5.1% |
| **detection_miss_source** | 4 | 4.0% |
| **detection_miss_both** | 4 | 4.0% |
| wrong_link | 3 | 3.0% |
| not_linked | 1 | 1.0% | **Detection misses outnumber linking errors 13 to 4.** All 13 are on `44b6_0b24845f`;
`44b6_0113de3b` loses only 2 edges, both to identity swaps.

Also measured: `(of those, present before pruning) = 0` - post-processing is *not* discarding
GT-matched nodes, so `min_track_len` and the gap-closer are not the problem.

Why the aggregate numbers misled: node recall counts *nodes*, but one missed node destroys up
to **two** edges, and misses cluster on the hard video. Recall of 0.91 therefore maps to a much
larger edge loss than it appears to.

**Consequence:** the learned edge scorer drops down the priority list, and detector quality
(better scale-space, U-Net, or a DoG∪U-Net union) moves to the top. Ranked next steps below are
updated accordingly.

### Ranked next steps

*Re-ranked after the error attribution above.*

1. **Detector recall on hard videos.** 13.1% of GT edges are lost to detection misses, all on
   the denser video. Levers: DoG scale-space geometry, the trained U-Net, and a **DoG ∪ U-Net
   union** (the two miss different cells, so the union should beat either).
2. **Per-dataset detection budget** so the node count lands near that video's expected cell
   count. The 74-786 cells/frame spread makes any single global threshold wrong for most
   videos; `detect_frames_unet(budget_from_dog=True)` is a first cut.
3. **Learned edge scorer.** Still worth doing - 4.0% of edges are linking errors - but it is
   now the *second* lever, not the first. Every public solution above ~0.90 uses one.
4. **ILP global linking** with asymmetric appearance/disappearance costs (public notebooks use
   appearance 0.0, disappearance 1.575 - cheap to start a track, expensive to end one).
5. **Two-seed heatmap logit blending before NMS** (blending after NMS is much weaker).
6. Divisions - lowest priority despite the 0.1 weight, see experiment 8.

---

## Session 3 - 2026-08-07

### The metric is now reproducible without `tracksdata`

`src/biohub_metric.py` reimplements the organizers' scorer in numpy + scipy.
`tests/test_metric_parity.py` asserts **exact** agreement on TP/FP/FN and 1e-9 agreement on
the adjusted Jaccard, across three configurations on both scorable videos - and it passes.

This is the session's most useful artifact. It removes the constraint that forced every earlier
decision onto an n=2 sample: `notebooks/05_full_cv.ipynb` now scores against **60 training
videos** inside Kaggle, where all 199 are already mounted.

Quirks that had to be reproduced exactly (each changes the number materially): the
`pred_valid = out_valid | in_valid` false-positive rule, merge collapsing, out-degree cap of 2,
the >1 adjustment factor, and the fact that run-level adjusted Jaccard is weighted by
`TP+FP+FN` while plain Jaccard is micro-averaged.

### Detection union - higher recall, lower score

Independently-NMS'd DoG runs merged in physical space (`detect_dog_union`). On the hard video
node recall went **0.725  0.961**. End to end it *lost*:

| configuration | recall | adj |
|---|---|---|
| baseline (max over scales) | 0.902 | **0.7395** |
| union of 2 scale sets | 0.902 | 0.6516 |
| union of 3 scale sets | 0.990 | 0.5383 |
| union of 3, capped 400/frame | 0.931 | 0.6172 |
| union of 3, capped 350/frame | 0.902 | 0.6213 | Tighter linking gates did not rescue it (4/5/6/7 µm  0.456/0.545/0.601/0.617).

**Lesson: detection *recall* is not the objective - edge Jaccard is.** Extra candidates confuse
the assignment more than the recovered nodes gain. What is needed is a detector that *ranks*
true cells above spurious ones, which is a learned detector, not a wider band-pass. Kept as a
documented function with the numbers in its docstring.

### Learned edge scorer - no measured benefit over plain distance

`src/biohub_link.py` builds 13 features per candidate pair (geometry, motion residual,
forward/backward distance ranks, runner-up margins, mutual-nearest-neighbour, local density,
patch NCC) and fits a gradient-boosted classifier.

On 318 labelled pairs from the two available videos:

```
4-fold ROC-AUC (learned) : 0.965 ± 0.038
distance alone           : 0.969
mutual-NN alone          : 0.945
```

**The learned scorer does not beat the single distance feature.** Within an already-gated
candidate set the geometry is nearly sufficient, and permutation importance is dominated by
`rank_fwd`, `dxy_um`, `dz_um_abs` - all geometric.

Caveats before discarding the idea: 318 pairs from 2 videos is a very small sample; the pairs
are anchored on annotated nodes, which may be systematically easier; and AUC on gated pairs is
not the same as end-to-end score. The infrastructure is kept
(`scripts/build_edge_dataset.py`) so it can be retrained on all 199 videos if the detector
improves enough for linking to become the binding constraint. **But it should not be assumed
to help - every public solution uses one, and here it measurably did not.**

### Housekeeping

* `scripts/error_attribution.py` - classifies every missed GT edge by cause.
* `scripts/kernel_log.py` - decodes Kaggle's JSON log format into readable text.
* Local per-file download of more training videos was abandoned: ~42 of 1600 chunks in 25
  minutes under rate limiting. Evaluating on Kaggle is strictly better.

### DoG scale-space sweep - 36 configs, end-to-end scored

`scripts/sweep_detector.py` (n=2, so treat as a ranking only):

| scale set | best min_dist | rel | adj | node recall | recall (hard video) | peaks/frame |
|---|---|---|---|---|---|---|
| **current 2-scale** | **3.0** | 0.02 | **0.7552** | 0.931 | 0.863 | 329 |
| current 2-scale | 3.5 | 0.02 | 0.7395 | 0.912 | 0.824 | 319 |
| single small | 3.5 | 0.01 | 0.7062 | 0.951 | 0.902 | 391 |
| fine 2-scale | 4.5 | 0.02 | 0.7024 | 0.882 | 0.765 | 373 |
| 3-scale fine | 3.0 | 0.02 | 0.6728 | **0.961** | **0.922** | 431 |
| 4-scale wide | 3.5 | 0.02 | 0.6701 | 0.892 | 0.784 | 401 | Two things worth keeping:

1. **`min_distance_um` 3.5  3.0 is worth +0.0157** on this sample. Held back from the default
   until the 30-video Kaggle CV confirms it - the whole point of building that instrument was
   to stop tuning on n=2.
2. **The recall/score inversion appears again, independently.** The scale sets with the *best*
   node recall (`3-scale fine` at 0.961, `single small` at 0.951) score *worst* (0.673, 0.706).
   The current 2-scale set has middling recall and the best score. This is the same effect the
   union experiment found, from a different direction, so it is a property of the metric and
   the linker rather than an artifact of one experiment.

### Why higher recall lowers the score - the mechanism, measured

`error_attribution.py` at three detector densities (n=2, totals across both videos):

| `min_distance_um` | TP | detection misses | **wrong_link** | not_linked |
|---|---|---|---|---|
| 2.0 | 60 | 9 | **26** | 4 |
| 3.0 | 84 | 11 | **3** | 1 |
| 3.5 | 82 | 13 | **3** | 1 | Going from 3.0  2.0 leaves detection misses essentially unchanged (11  9) while wrong links
explode **3  26**. The recall/score inversion is therefore *entirely* a linking failure under
dense candidates, not a node-count-penalty effect.

This is the key structural fact about the pipeline: **the linker is what caps how much
detection can be exploited.** Any future detector improvement has to come with a linker that
survives the extra candidates, or it will lose score.

### NMS radius - 3.0 is a true optimum

| `min_distance_um` | 2.0 | 2.5 | 3.0 | 3.25 | 3.5 | 4.5 |
|---|---|---|---|---|---|---|
| adj | 0.370 | 0.668 | **0.755** | 0.740 | 0.740 | ~0.67 | Not a grid edge - it falls off sharply on both sides. `rel_threshold` 0.02 beats 0.03 at every
radius. Pending confirmation on the 30-video Kaggle CV before changing the default.

### Learned edge scorer, dense regime - still not worth it

Rebuilt the pair dataset at `min_distance_um=2.0`, where wrong links are abundant:

| regime | pairs | learned AUC | distance-only | gain |
|---|---|---|---|---|
| sparse (md 3.5) | 318 | 0.9755 | 0.9686 | +0.0069 |
| dense (md 2.0) | 443 | 0.9405 | 0.9341 | +0.0063 | The gain is real but tiny, and nowhere near enough to repair 26 wrong links. Feature importance
in the dense regime is dominated by **`margin_fwd_um`** - the distance gap to the runner-up
target - which is an ambiguity signal rather than an appearance one.

### Ambiguity-margin guard - derived from the above, and it does not work

Acting on that feature directly: refuse a link when the runner-up is within `min_margin_um` of
the chosen target, leaving a hole for `close_gaps` to repair.

| margin (µm) | 0.0 | 0.5 | 1.0 | 1.5 | 2.5 |
|---|---|---|---|---|---|
| adj @ md 2.0 | 0.370 | 0.382 | 0.366 | 0.381 | 0.385 |
| adj @ md 3.0 | **0.755** | 0.740 | 0.739 | 0.739 | 0.681 | It barely helps in the dense regime and actively *hurts* at the good operating point. Kept as
`Config.min_margin_um`, defaulting to **0.0 (off)**.

**Standing conclusion:** dense detection is not usable with a per-frame Hungarian linker,
whether the cost is distance, a learned score, or distance with an ambiguity veto. Making it
usable needs a genuinely different linker - multi-frame/global (ILP) rather than greedy
frame-by-frame. That is now the highest-value untried idea.

### 60-video CV on Kaggle - and it invalidates most of the n=2 tuning

`notebooks/05_full_cv.ipynb`, 60 training videos, ~42,000 GT edges:

```
BASELINE  n=60  SCORE=0.7516  edge_J=0.7513  node_recall=0.8858  TP/FP/FN=31448/3632/6778
```

Sweep on 30 videos, all deltas vs a 0.7260 baseline:

| change | delta | | change | delta |
|---|---|---|---|---|
| min_distance 3.0 | **+0.0012** | | min_track_len 6 | −0.0025 |
| rel_threshold 0.04 | +0.0008 | | link 6 µm | −0.0027 |
| no motion model | +0.0004 | | rel_threshold 0.01 | −0.0062 |
| min_track_len 2 | −0.0022 | | max_gap 2 | −0.0076 |
| | | | scales 3-set | −0.0131 |
| | | | **no gap closing** | **−0.0166** | **`min_distance_um = 3.0` measured +0.0157 on n=2 and +0.0012 on n=30.** The n=2 result was
noise. Default left at 3.5 - the difference is not real. This is the clearest possible
vindication of building the large-CV instrument, and a warning about every earlier delta.

Two things that *are* real: **gap closing is worth +0.017**, confirming the session-1 ablation;
and the **motion model contributes nothing** (+0.0004 to remove it) - a candidate for
simplification.

### What actually predicts the score: cell density

Per-video correlations across 60 videos: **node_recall +0.900**, density −0.392,
peaks/frame −0.377, node-count error **−0.051** (i.e. the count penalty is nearly irrelevant
in practice).

By density quartile:

| quartile | cells/frame | mean spacing | node recall | adj |
|---|---|---|---|---|
| sparse | 55 | 27.4 µm | 0.947 | **0.884** |
| med-low | 106 | 22.0 µm | 0.848 | 0.696 |
| med-high | 223 | 17.2 µm | 0.904 | 0.712 |
| dense | 521 | 12.9 µm | 0.844 | **0.656** | By embryo: `44b6` 0.650 vs `6bba` 0.784. **20% of videos score below 0.6**, and lifting the
worst 10 to the median would add ~0.055 to the aggregate - far more than any hyperparameter.

Note this does *not* contradict the recall/score inversion: across *videos*, recall predicts
score (easy videos are easy); within a video, adding candidates still hurts. Different things.

### Automatic per-video scale selection - sound theory, bad result

Cells shrink as the embryo divides, so a fixed band-pass should not fit both a 27 µm-spacing
and a 13 µm-spacing video. `estimate_scale_um()` implements Lindeberg scale selection: sweep
sigma, take the argmax of the sigma²-normalised DoG response.

It chose the **largest** candidate (3.8 µm) for both test videos and collapsed the score:

| video | fixed scales | auto scales |
|---|---|---|
| 44b6_0113de3b | 0.9025 (recall 1.000) | 0.3591 (recall 0.635) |
| 44b6_0b24845f | 0.5880 (recall 0.824) | 0.1905 (recall 0.216) | The scale-normalised response is maximised by tissue-level structure, not by individual cells,
so the criterion measures the wrong thing on this data. Kept as `Config.auto_scale`, default
**False**. The underlying observation - density drives the score - remains the most promising
lead; the fix needs a criterion tied to cell *count* or nearest-neighbour spacing rather than
response magnitude.

### The U-Net detector does not work - decisive, after the selection bug was fixed

Training v5 ran 24 epochs on a T4 (272 min) with the corrected, non-saturating selection metric
and a 2,138-point validation set. The metric now discriminates properly. What it shows:

| epoch | loss | top-K recall | median dist |
|---|---|---|---|
| **untrained (random init)** | - | **0.377** | 8.79 µm |
| 1 | 0.1204 | 0.304 | 17.51 µm |
| 3 (best selection) | 0.0942 | 0.527 | 3.92 µm |
| 14 | 0.0592 | 0.507 | 6.12 µm |
| 24 | 0.0460 | 0.476 | 8.95 µm | **A randomly-initialised network already scores 0.377**, and 24 epochs of training reach only
~0.48. Meanwhile the classical DoG detector achieves **node recall 0.886** on the 60-video CV.
The training loss falls smoothly (0.120  0.046) while the detection metric does not improve - the objective is being optimised but it is not aligned with the task.

End-to-end on the two scorable videos, with the correct top-K / DoG-budget read-out:

| detector | adj | node recall | edge TP/FP/FN |
|---|---|---|---|
| **classical DoG** | **0.7395** | 0.912 | 82/11/17 |
| U-Net epoch 3 | 0.2969 | **0.942** | 52/74/47 |
| U-Net epoch 24 | 0.1252 | 0.785 | 25/99/74 | Two things stand out. **Training makes it worse** (0.297  0.125). And at epoch 3 the U-Net's
*node recall is higher than DoG's* (0.942 vs 0.912) while its edge Jaccard is a quarter of DoG's - false positives go 11  74. So the network finds cells but **localises them too imprecisely
to track**: median error 4-9 µm against a 7 µm matching gate and ~1.7 µm of real per-frame
motion, so frame-to-frame jitter swamps the actual displacement.

**Not submitted. The classical pipeline remains the best model.**

Diagnosis for anyone retrying (do not simply rerun this recipe):

1. The positive target is a 1-voxel-radius cube on a 1.625 µm grid, which caps achievable
   localisation - and the observed 4-9 µm error is far worse than that cap, so the peaks are
   not landing on cell centres at all.
2. The count-normalised BCE (positives total 1.0, all negatives total 0.1) appears to admit a
   degenerate solution: a smooth low-frequency map that scores well on the loss and badly on
   detection.
3. Worth trying instead: **Gaussian heatmap regression with an MSE/focal loss on a soft target**
   rather than weighted BCE on a near-binary one; and **crop-based sampling centred on
   annotations**, so positives are not 1 in 262,144 voxels.

This is the second failed attempt at the learned detector (the first was the checkpoint-selection
bug). The recipe is from the organizers' baseline and public solutions reach ~0.908 with it, so
it *can* work - but two multi-hour runs have not reproduced that, and the honest position is
that the classical detector is better until a fundamentally different formulation is tested.

### Corrected detector loss - the learned detector finally beats DoG

`local_heatmap_loss` replaces the count-normalised BCE: a CenterNet-style penalty-reduced focal
loss on a **soft Gaussian target**, computed **only inside a ±4-voxel (~6.5 µm) cube around each
annotation**, plus a small global negative term. Outside those cubes the label is genuinely
unknown (most cells are unannotated), so contributing nothing there is both honest and removes
the degenerate "smooth low-frequency map" solution.

Validated on synthetic blobs *before* spending GPU: the new loss puts peaks on the **exact**
voxel (0.0 error) within 120 steps; the old one never localised.

Training v6, same 24 epochs / same data / same architecture - only the loss changed:

| | top-K recall | median localisation |
|---|---|---|
| random init | 0.377 | 8.79 µm |
| **old loss**, 24 epochs | 0.476 | 8.95 µm |
| **new loss**, epoch 2 | **0.957** | **1.82 µm** |
| **new loss**, epoch 24 | 0.940 | **1.68 µm** | End-to-end on the two scorable videos:

| detector | adj | node recall | TP/FP/FN |
|---|---|---|---|
| classical DoG | 0.7395 | 0.912 | 82/11/17 |
| U-Net epoch 2 (best selection) | 0.7697 | 0.952 | 87/13/12 |
| **U-Net epoch 24 (last)** | **0.7740** | 0.942 | 86/11/13 | **+0.0345 over DoG**, and node recall exceeds it. Two lessons:

* The diagnosis was right: the failure was **localisation**, not capacity or data. Peaks 4-9 µm
  from centres are useless against a 7 µm gate with ~1.7 µm of real motion; 1.68 µm works.
* **Saving `unet_detector_last.pt` paid off** - the last epoch beats the best-selected one
  end-to-end, so the selection metric is still not perfectly aligned with the score. Keep both.

Pending confirmation on the 40-video detector CV (`notebooks/07_detector_cv.ipynb`) before
submitting; n=2 has produced a false positive before.

### Density-adaptive NMS radius - a physical law, not a fitted constant

`06_dense_rescue` asked whether dense and sparse videos want different parameters. They do:

| group | cells/frame | spacing | best NMS | delta vs baseline |
|---|---|---|---|---|
| sparse | 55 | 27.3 µm | **5.5 µm** | **+0.0186** |
| mid | 106 | 22.0 µm | 3.5-4.5 µm | +0.0000 |
| dense | 521 | 12.9 µm | 3.0 µm | +0.0057 | Converting density to mean cell spacing, the optimal radius is a near-constant fraction of it:
ratios **0.201 / 0.205 / 0.232**. So the whole relationship reduces to

    NMS radius = 0.20 x (video_volume / cells_per_frame)^(1/3)

and it is usable at test time: observed peaks/frame correlates with the true count at
**r = 0.999**, so `estimated_number_of_nodes` is not needed. Implemented as
`Config.adaptive_nms` via `adaptive_min_distance()`.

Locally 0.7395 -> **0.7556**, with accurate density estimates (260 vs 258 true; 369 vs 328).
Running on the 60-video CV before adoption - `min_distance 3.0` looked worth +0.016 on n=2 and
turned out to be +0.001 on n=30, so n=2 agreement is not evidence.

### CORRECTION - adaptive NMS is worth ~+0.003, not +0.016

The 30-video CV verdict on the density-adaptive radius:

| config | delta vs baseline |
|---|---|
| adaptive NMS, ratio 0.17 | **+0.0032** |
| adaptive NMS, ratio 0.20 | +0.0014 |
| adaptive NMS, ratio 0.24 | −0.0054 |
| (min_distance 3.0, for scale) | +0.0012 | n=2 said +0.016. **That was noise again** - the third time an n=2 result has failed to survive
the large CV (after `min_distance 3.0` and the earlier `rel_threshold` deltas).

The underlying physics still holds - the per-group optima really are ~0.2 x spacing, and the
density proxy really is r=0.999 - but the *aggregate* benefit is small because the evaluation
distribution is dominated by mid/dense videos where the adaptive radius lands close to the fixed
3.5 µm anyway. The +0.0186 measured on the sparse group is real but sparse videos are a minority
of the sample.

Not adopted as a default. `Config.adaptive_nms` stays available and defaults to False; ratio
0.17 is the better setting if it is ever turned on.

**Standing rule for this project: no parameter change is adopted on fewer than ~30 videos.**
Every single n=2 "improvement" this session has evaporated at scale, while the two changes that
*did* survive (gap closing, and the detector loss rewrite) were large enough to see anywhere.

### CONFIRMED on 40 videos - the learned detector is the session's real gain

`notebooks/07_detector_cv.ipynb`, 40 videos stratified across the density range, same videos for
every arm:

| detector + read-out | adj | node recall | TP/FP/FN |
|---|---|---|---|
| classical DoG | 0.7814 | 0.899 | 23002/2180/4275 |
| **U-Net, top-K from DoG budget** | **0.8221** | **0.942** | 24109/2086/3168 |
| U-Net, fixed global K=300 | 0.6148 | 0.897 | 22688/2442/4589 |
| U-Net, threshold 0.5 | 0.4464 | 0.395 | 11825/331/15452 | **+0.0407**, winning on **31/40 videos** (mean delta +0.0488, median +0.0271) and **+0.0344 on
the densest quartile** - precisely where the score was being lost.

Two predictions from the earlier analysis were confirmed quantitatively:

* **Thresholding is catastrophic (0.4464).** The heatmap is a *ranking*, not a probability.
* **A fixed global K fails (0.6148).** The 74-786 cells/frame spread defeats any single budget;
  the per-frame DoG-derived budget is what makes the learned detector usable. Note this arm has
  a *higher* edge Jaccard (0.7634) than its adjusted score (0.6148) - the gap is the node-count
  penalty, the one place where that penalty actually bites.

This is the only change all session whose margin sits far above the noise floor, and it was
predicted by an independently-measured mechanism (localisation 8.95 µm -> 1.68 µm, verified on
synthetic blobs) *before* the score confirmed it.

### Session scoreboard - what survived

| change | small-sample estimate | large CV | adopted |
|---|---|---|---|
| gap closing | +0.115 (n=1) | **+0.017** (n=30) | |
| **detector loss rewrite** | +0.035 (n=2) | **+0.041** (n=40) | |
| adaptive NMS | +0.016 (n=2) | +0.003 (n=30) | |
| min_distance 3.0 | +0.016 (n=2) | +0.001 (n=30) | |
| union detection | - | −0.10 to −0.20 | |
| learned edge scorer | +0.006 AUC | - | |
| ambiguity veto | - | −0.015 | |
| auto scale selection | - | −0.4 | | **The pattern: every parametric gain evaporated at scale; both architectural gains held.** The
two survivors were 10-100x larger than any tuning delta and visible at any sample size. Three
sweeps (162, 36 and 13 configurations) between them found nothing worth adopting.

### Leaderboard: 0.824 -> 0.852 with the learned detector

| submission | config | CV | public LB |
|---|---|---|---|
| v2 | classical, untuned | 0.697 (n=2) | 0.785 |
| v5 | classical, tuned | 0.7395 (n=2) / 0.7516 (n=60) | 0.824 |
| **U-Net** | learned detector + top-K DoG budget | **0.8221 (n=40)** | **0.852** | Pre-submission diagnostics on the four placeholder videos were the strongest of the project:
node recall **1.000 / 0.933 / 1.000 / 0.983**, and node-count multipliers **0.997 / 0.984 /
1.007 / 1.003** across an 11x density range (62-671 peaks/frame). The DoG-derived per-frame
budget lands the node count almost exactly on the expected cell count without ever seeing
`estimated_number_of_nodes` - which is the mechanism that makes the learned detector usable.

#### Calibration update - the delta transferred at ~70%, not ~95%

| | CV delta | LB delta | ratio |
|---|---|---|---|
| classical tuning (session 1) | +0.042 | +0.039 | 0.93 |
| **learned detector** | **+0.041** | **+0.028** | **0.68** | Earlier this log claimed deltas "transfer almost exactly". On this change they transferred at
about two thirds. The predicted LB was ~0.86 and the actual is 0.852 - the direction and rough
magnitude were right, the precision was not.

Plausible reason: the CV set is drawn from *training* videos while the LB is a different sample,
and the detector was trained on 175 of those training videos. Even with a held-out split inside
training, the CV is measured on the same two embryos and imaging sessions the model saw, so some
optimism is expected for a *learned* component in a way it is not for a parameter change.

**Revised rule: treat CV deltas as an upper bound for learned components, and roughly 1:1 for
parameter changes.** Still the right instrument for ranking; not a predictor of the LB number.

---

## Session 4 - 2026-09-03

### Public-field survey - the landscape changed

Public notebooks are now at **0.926-0.936** and the LB leader is **0.963** (we were at 0.852).
The important structural fact: **every 0.90+ public notebook is the same lineage** - the
organizers' `tracking_cellmot` repo driven by ~60 `BIOHUB_*` environment variables, plus three
shared Kaggle datasets:

* `pilkwang/biohub-tracking-support-pack-50ep-v1` (349 MB, **12,799 downloads**) - the repo plus
  a trained **U-Net + transformer edge predictor** (`weights/unet_transformer/split_0/edge_predictor_best.pth`)
* `pilkwang/biohub-deepcenter-unet3d-center-prior-v1` - a centre-prior model used as a *veto*
* `pilkwang/biohub-temporal-unet3d-seed314159-v1` - a second seed for logit blending

So the public 0.93 is a shared pretrained stack anyone can attach, tuned by env var - not 0.93
of independent modelling. Their recipe: ILP (appearance 0.0 / disappearance 2.0 / division 1.2),
learned edge scoring, gap2 recovery, wide safe divisions (7.0 / 12.0 µm), short-track rescue,
line-fit smoothing, dual-seed detection fusion, and harmonic bidirectional association.

### Line-fit trajectory smoothing - adopted, +0.0135

Every top notebook runs `OUTPUT_LINEFIT_SMOOTH`. Implemented as `smooth_tracks`: fit a line to
each node's own track over ±window frames and move the node `weight` of the way toward it,
stopping the chain at branch points so a division's two daughters are never averaged together.

30-video CV with U-Net detections cached (all configs see identical candidates):

| config | adj | delta |
|---|---|---|
| **smooth 0.8, window 3** | **0.8229** | **+0.0135** |
| bidir 0.15 + smooth 0.8 | 0.8219 | +0.0125 |
| smooth 0.8, window 2 | 0.8213 | +0.0119 |
| smooth 0.5 | 0.8194 | +0.0100 |
| baseline | 0.8094 | - | Rationale that makes it work here: matching uses a hard 7 µm radius while real motion is smooth
at ~1.7 µm/frame, so per-frame detection jitter is nearly pure noise. **Adopted as the default**
(0.8 / window 3) - the largest post-processing gain since gap closing.

### Harmonic bidirectional fusion - +0.0006, i.e. noise

The headline technique of the 0.936 notebook, implemented as `link_harmonic`:
`p = 1/((1-w)/p_fwd + w/p_rev)`, swept over w ∈ {0.10, 0.15, 0.25} and temperature ∈ {1, 2, 4} µm.
Every variant landed within 0.0006 of baseline.

**Why it works for them and not here:** their fusion combines a *learned* edge model's logits,
which encode appearance and context, so the forward and reverse passes are genuinely different
estimators and mutual agreement is informative. Fusing a **symmetric distance matrix** gives two
softmax views of the same numbers - the reverse pass carries almost no information the forward
pass lacked. The technique is sound; it needs a learned asymmetric scorer to have anything to
fuse.

Kept as `Config.bidir_weight`, default 0.0.

### Wide division gates - still no help

Public notebooks use `SAFE_DIV_MAX_UM 7.0` / `SISTER 12.0`, far wider than the 4.7 / 7.5 tested
in session 1. At their settings: **−0.0007**. Divisions remain unreachable by geometry alone in
this pipeline; theirs works because the ILP predicts divisions natively and the geometric rule
only *confirms* them.

Also re-confirmed as harmful at scale: `min_track_len 6` (−0.0035) and `max_gap 2` (−0.0103),
both of which the public notebooks use - further evidence that their settings are tuned to
*their* pipeline and do not transfer piecemeal.

### Leaderboard: 0.852 -> 0.866 with line-fit smoothing

| submission | change | CV | public LB |
|---|---|---|---|
| classical tuned | - | 0.7516 (n=60) | 0.824 |
| U-Net detector | learned detection | 0.8221 (n=40) | 0.852 |
| **+ smoothing** | line-fit 0.8 / window 3 | **0.8229 (n=30)** | **0.866** | **Delta transfer, third data point:** +0.0135 CV -> +0.014 LB, i.e. ~1.0. This confirms the
revised rule from session 3:

| change type | CV -> LB transfer |
|---|---|
| parameter / post-processing | **~1.0** (+0.042->+0.039, +0.0135->+0.014) |
| learned component | **~0.68** (+0.041->+0.028) | The learned detector transfers at a discount because the CV is measured on training videos the
model was fitted on (same two embryos, same imaging sessions); a post-processing rule has no
such optimism. Use ~1.0 for algorithmic changes and ~0.7 for anything trained.

### Session 4 scoreboard

Four public techniques ported, **one transferred**:

| technique | result | verdict |
|---|---|---|
| line-fit smoothing | **+0.0135** | adopted, LB +0.014 |
| harmonic bidirectional fusion | +0.0006 | needs a learned asymmetric scorer |
| wide division gates 7/12 | −0.0007 | |
| `min_track_len 6` | −0.0035 | |
| `max_gap 2` | −0.0103 | | The last three are settings the 0.93+ notebooks actively *use*. They make this pipeline worse
because they are tuned to a pipeline whose ILP predicts divisions natively and whose linker is a
learned scorer. **Public settings do not transfer piecemeal** - only the technique that was
independent of their architecture (smoothing) carried over.

### ITEC-style evidence-based track repair - no gain, and the per-embryo split rejects it

Implemented `repair_tracks_with_evidence` + `build_graph_iterative`: instead of interpolating a
node across a dropout, predict where the cell should be from the trajectory, inspect the U-Net
heatmap there, and insert the **real local peak** when one clears a threshold; recovered nodes
are fed back into the candidate pool and the graph is re-linked.

Verified on a synthetic case first: a cell present in the heatmap at x=116 but dropped from the
top-K list is recovered at **x = 116.0 exactly**, where interpolation places a chord midpoint.

16 videos (8 per embryo), U-Net detections + heatmaps cached:

| config | adj | 44b6 | 6bba | evidence / interp |
|---|---|---|---|---|
| **baseline (interpolated)** | **0.8607** | **0.7803** | 0.8775 | - |
| search 3 µm, resp 0.2 | 0.8615 | 0.7825 | 0.8783 | 16439 / 3591 |
| 2 rounds | 0.8618 | 0.7759 | **0.8814** | 2015 / 370 |
| max_gap 2, 2 rounds | 0.8393 | 0.7510 | 0.8561 | 11717 / 2977 |
| max_gap 3, 2 rounds | 0.8150 | 0.7260 | 0.8357 | 22043 / 6971 | Best is **+0.0011** - noise. And the per-embryo split kills it outright: the best configuration
is **worse on 44b6** (0.7759 vs 0.7803) and better only on 6bba. A gain present in one embryo
and absent in the other is an imaging-condition artifact, and the hidden test is embryo-disjoint.
**Not adopted.**

**Why it does not help, despite working.** The evidence is genuinely there - 16,439 recoveries
against 3,591 fallbacks, so the detector really did see those cells below its top-K budget. The
problem is that *position was never the binding constraint at one-frame gaps*: cells move
~1.72 µm/frame, so a chord midpoint already lands well inside the 7 µm matching gate.
Improving a position that was already good enough changes no match.

Where ITEC should pay is longer gaps and dense regions - and `max_gap` 2 and 3 are exactly where
the score collapses (−0.021, −0.046) **even with real evidence**, because more nodes means the
per-frame Hungarian linker makes more wrong links. This is the same wall as the detection-union
and dense-NMS experiments.

**Standing conclusion: every structural improvement attempted so far is gated behind the
linker.** Better detection, more candidates, and better-placed repairs all fail for the same
reason. That is now three independent lines of evidence that the *graph construction* - not
detection, not post-processing - is the binding constraint, and it argues for putting the
learned edge scorer + global ILP in place before anything else is tried on top.

### ILP ablation - the learned linker works, but our nodes are off-distribution for it

6 videos (3 per embryo), detection frozen, divisions at pack defaults in every arm.

| arm | detector | edge scoring | assignment | adj | 44b6 | 6bba | edge FP | edge FN |
|---|---|---|---|---|---|---|---|---|
| **A** | public | public learned | **ILP** | **0.8883** | 0.8844 | 0.9071 | 158 | 192 |
| B | ours | distance | Hungarian | 0.8238 | 0.7953 | 0.8569 | 274 | 356 |
| C | ours | public learned | **ILP** | 0.8144 | 0.8142 | 0.8523 | 217 | 476 |
| D | ours | public learned | greedy | 0.7859 | 0.7914 | 0.8230 | 276 | 480 | **First, a methodology correction.** The initial run had C and D byte-identical because
`predict_video()` returns the *pre-ILP* graph - the solver lives in their `predict()`, which was
never called. Every arm was greedy. Fixed, plus a guard that warns if C ever equals D again.

Three readings, in order of importance:

1. **The complete public stack beats ours by +0.065** (0.8883 vs 0.8238), and wins on *both*
   embryos. That is by far the largest single delta measured in this project.
2. **The ILP itself is worth +0.0285** (C 0.8144 vs D 0.7859, identical edge probabilities,
   only the assignment differs). Global consistency is a real, separable contribution.
3. **The learned scorer under-links our nodes.** With our detections it cuts false positives
   274  217 (it *is* discriminating) but pushes false negatives 356  **476**. It refuses links
   it would happily make between its own detector's nodes.

C vs B is **+0.019 on 44b6 and −0.005 on 6bba** - split, so it fails the stated bar and no
thresholds were tuned.

**Diagnosis: feature-domain mismatch, exactly as predicted.** The edge model indexes encoder
features at *integer voxels on the downsampled grid*, at locations its own detection head
produced. Our centroids quantise onto slightly different voxels and come from a different
detector with different density, so the features it looks up are off-distribution - and it
responds by withholding edges.

**Consequence for the plan:** the fastest credible jump is not "our detector + their linker" but
the untouched public stack, then repairing the domain mismatch to put our detector back in.

---

## Session 5 - 2026-09-18

Eleven days to the deadline. Opened with a leaderboard reality check, which
changed the plan more than any measurement did.

### The leaderboard is a cliff, and 0.947 is now worth nothing

3,674 teams. Medal cutoffs and the score needed to clear them:

| medal | rank | score at that rank |
|---|---|---|
| gold | 15 | **0.9580** |
| silver | 183 | 0.9470 |
| bronze | 367 | 0.9470 | Bronze and silver have the **same cutoff score** because **622 teams sit on exactly
0.9470** - a single public notebook, forked. That block spans ranks 171-792. Kaggle
breaks ties by earliest submission, so arriving at 0.947 *today* lands at ~rank 790:
**not silver, not even bronze.** Our 0.866 is rank 2,615.

What each score is actually worth right now:

| score | rank |
|---|---|
| 0.866 (us) | 2,615 |
| 0.947 | ~790 (back of the tie block) |
| 0.948 | ~170 - silver, barely |
| 0.950 | ~70 |
| 0.958 | ~15 - gold | **Consequence: reproducing the public stack is table stakes, not progress.** The
target is 0.950+ for a silver that survives more teams piling in, 0.958+ for gold.

### The public frontier is one shared stack, configured by environment variable

Every notebook at 0.94+ attaches the same three datasets and differs only in
`BIOHUB_*` env vars:

* `pilkwang/biohub-tracking-support-pack-50ep-v1` (18,659 downloads) - the organizers'
  repo plus a trained U-Net + transformer edge predictor
* `pilkwang/biohub-deepcenter-unet3d-center-prior-v1` - a centre-prior model used as a veto
* `pilkwang/biohub-temporal-unet3d-seed314159-v1` - a second seed for logit blending

The verified progression is documented inside the notebooks themselves:
0.933  0.934 (harmonic fusion)  0.939 (wider divisions)  0.941 (repair budget)
 0.946 (primary edge-feature TTA)  0.947 (secondary feature TTA + DeepCenter TTA).
A newer variant (`haideptry`, unverified by us) claims 0.948 by adding a DivNet 3D
mitosis gate, `MOTION_RELINK_TIGHT_UM` 6.0  5.5 and `UNET_BATCH_SIZE` 8.

This matches session 4's ILP ablation, which already measured the complete public
stack at 0.8883 against our 0.8238 on the same six videos.

### Where the remaining score is: divisions, and it is a *ranking* problem

The score is `adj_edge_jaccard + 0.1 * division_jaccard`. Decomposing a 0.9508
held-out run from the public stack: `0.9383 + 0.1 * (1/8)`. So the division term
contributes **0.0125 of a possible 0.100**, and the arithmetic for the medals is:

| target | needed division Jaccard, at edge 0.9383 |
|---|---|
| 0.958 (gold) | 0.197 |
| 0.970 (LB leader) | 0.317 | The top of the board is almost certainly collecting divisions; the 622-team block is not.

Three facts, each independently sourced, explain why nobody in the block collects them:

1. **The 0.963-0.966 public notebooks are fossils.** They exploited a pre-patch rule
   where a GT division counted as recovered if its parent and daughters shared *any*
   weakly connected component containing *any* fork. One out-of-volume hub node wired
   to every track root satisfied it for every division at once. Patched 2026-07-17
   (`aa65e90`). Those scores stand; the method does not reproduce. Verified: all of
   the current top 200 submitted **after** the patch, so the live frontier is legitimate.
2. **The shipped gates are the wrong suspect.** Of 151 real divisions in training,
   the gate stack leaves 35 reachable - but opening the gates end-to-end *lowers* the
   score (0.9508  0.9341). Opening them takes one video's candidate pool from 141 to
   1,586 against a per-frame budget of ~5 slots, and the ranking key is
   `score = parent_dist + 0.15 * sister_dist`, ascending - **pure geometry, tightest
   pair first**. The tightest pairs are duplicate detections, not divisions, so the
   budget is spent before a real division is reached.
3. **Everyone in the block is tuning divisions against a broken metric.** The public
   stack's own held-out validator still scores divisions with the *pre-patch*
   weakly-connected-component rule. On the same prediction it reports division Jaccard
   0.2500 where the official scorer says **0.1250**. Every safe-division threshold in
   that stack was selected by a compass reading double.

**So the lever is the fork ranker, and the instrument to steer it did not exist.**

### `src/biohub_div_metric.py` - the patched division metric, without `tracksdata`

The organizers' `division_metrics.py` needs `tracksdata` + `polars`, neither of which
is in the Kaggle image, so the division half of the score has never been measurable
inside a notebook. Ported to plain dicts/sets over `biohub_metric.match_nodes`.

`tests/test_div_metric_parity.py` checks it against the real implementation on ten
synthetic graphs, each targeting one clause. **Exact agreement on TP/FN/FP in all ten**,
including the hub exploit (official 0/1/0, ours 0/1/0).

Details that had to be reproduced, each of which changes the count:

* Matching runs **per GT division window** against a six-node subgraph, not against the
  full GT graph - a fresh matching per division, so a pred node may partner differently
  in each.
* Candidate forks are restricted to the matched parent side and its immediate successors.
* A fork whose two direct-child branches carry evidence in two different GT weakly
  connected components is rejected; so is one with merged local branches.
* Direct-child evidence beats grandchild evidence; grandchildren are a fallback only.
* Bipartite pairing stops one fork claiming two divisions.
* **FPs are unioned, not summed.** `fp = |considered ∪ evaluable ∪ invalid| − |tp|`,
  where `evaluable` is any predicted fork matched to an *annotated* GT node with a child.

Two properties worth carrying forward, both found by the parity tests:

* **A spurious fork on an unannotated cell is invisible**, exactly as an unannotated
  edge is. The metric tolerates a wide candidate pool far better than the shipped
  gates assume - which is further evidence that gating is the wrong lever.
* **Matching is a global per-timepoint assignment, not nearest-neighbour.** A duplicate
  detection of one daughter can be handed to the *other* daughter and complete a
  division whose own node was never predicted (`duplicate_detection` case: scores TP).
  Over-proposing forks *near* a real division is therefore not symmetric with
  proposing them elsewhere.

### Division inventory on all 199 films (`13_div_inventory`, CPU, ~3 min)

Graph arrays only, no images, no model. **151 labelled divisions**, and the shipped gate
stack leaves **35 of them reachable (23.2%)** - an independent reproduction of the public
claim, which is also a real-data check on `biohub_div_metric`.

Where each gate sits in the distribution of *real* divisions:

| gate | shipped | real divisions passing | percentile it sits at |
|---|---|---|---|
| `SAFE_DIV_MAX_UM` | 9.0 | 80.1% | 80th |
| `SAFE_DIV_SISTER_MAX_UM` | 14.0 | 87.4% | 87th |
| **`SAFE_DIV_DIVERGE_UM`** | **2.25** | **51.7%** | **48th - the median** |
| **`SAFE_DIV_SISTER_SYMMETRY_TAU`** | **0.6** | **62.3%** | **62nd** | Cumulative survival: 151  121  114  **61**  **35**. The two distance gates are nearly
free; `diverge` and `symmetry` between them throw away 79 of the 114 that survive the
distance gates. Both are described in the source as precision filters and both sit at
roughly the middle of the true distribution, so against real divisions they are coin flips.

#### The bar for gold is far lower than it looks

Division Jaccard as a function of (recall, precision), D = 151:

| recall \ precision | 20% | 30% | 50% | 70% | 90% |
|---|---|---|---|---|---|
| 30% | 0.136 | 0.176 | 0.231 | 0.266 | 0.290 |
| 50% | 0.167 | 0.231 | 0.333 | 0.412 | 0.474 |
| 80% | 0.190 | 0.279 | 0.444 | 0.596 | 0.735 | What each target demands, holding edge Jaccard at 0.9383:

| target J | score | precision needed at 40% recall | at 60% recall |
|---|---|---|---|
| 0.200 | **0.9583 (gold)** | **29%** | **23%** |
| 0.300 | 0.9683 | 55% | 37% |
| 0.400 | 0.9783 | 100% | 55% | **Gold needs ~30% precision at ~40% recall.** That is not a hard classification problem - it is a gate problem plus a mediocre ranker. Dropping `diverge` and `symmetry` alone takes
reachability from 23% to 75%, and at 75% recall a J of 0.20 needs only ~22% precision.

This reframes the effort: the expensive part is *not* building a great division classifier.
It is opening the gates without letting the geometry-only ranker spend the budget on
duplicate detections - which is why the ranking key is the thing to replace.

#### Why a wide pool is cheaper than the gates assume

The metric charges a false positive only for forks that are local to a GT division,
`evaluable` (matched onto an annotated node with a child), or invalid. Median labelled
fraction is **0.036**, so of order 28 forks can be proposed per one the metric can even see - a loose upper bound, since real proposals cluster where annotations do.

#### The transfer risk to write down now

| embryo | films | divisions | labelled fraction |
|---|---|---|---|
| `44b6` | 71 | 26 | 0.0077 |
| `6bba` | 128 | 125 | 0.0971 | Divisions are **5× more numerous** in `6bba` and its labelling is **12× denser**. Reachability
under the shipped gates also differs (34.6% vs 20.8%). The hidden test is a *third* embryo,
so any FP policy calibrated on the pooled set is calibrated mostly on `6bba`. Every division
result from here on gets reported per embryo, and a gain present in only one is rejected - the same rule that correctly killed the ITEC repair in session 4.

Divisions are also concentrated: 50% of them live in the 27 richest films, 80% in 57. A
random held-out split measures the division term on almost nothing.

### The public fork ranker has the wrong *sign*, not the wrong tuning

The line that spends the fork budget (`add_safe_divisions_postlink`):

```python
score = parent_dist + 0.15 * sister_dist
...
proposals.sort(key=lambda item: item[0])     # ascending - tightest pair first
```

Against the measured geometry of all 151 real divisions:

| `parent_dist_um` | p0 | p10 | p50 | p90 | p100 |
|---|---|---|---|---|---|
| real divisions | 2.84 | 4.49 | **7.13** | 10.05 | 13.53 | * real divisions with `parent_dist` < 2 µm: **0 / 151**
* real divisions with `parent_dist` < 3 µm: **1 / 151 (0.7%)**

So an ascending key puts the 0-3 µm band at the **front** of a queue with ~5 slots per
frame, and no real division lives there. This is not a threshold that wants tuning; the key
is anti-correlated with the target. It explains every earlier observation at once: why
opening the gates lowers the score, why `cap_skipped` explodes, and why the one true
positive that existed is lost when the pool widens.

**Replacement** (`15_div_rank`): a Gaussian log-likelihood over the measured division
geometry - log space for the two distances, linear for the symmetry ratio, each term clipped
at 12 so one wild feature cannot dominate - used as a negative log-likelihood so
`proposals.sort()` is unchanged. Fitted parameters come from the 151 divisions:

```
parent_dist  log N(1.94503, 0.31430)
sister_dist  log N(2.30300, 0.33479)
symmetry         N(0.55760, 0.42911)
```

Ordering check against simulated duplicate detections (200 positives drawn from the measured
geometry, 2,000 duplicates at 0.2-3.0 µm offsets):

| key | AUC | real divisions in the top 5 slots |
|---|---|---|
| stock, ascending | **0.349** | 0 |
| likelihood | 0.999 | 5 | **An AUC of 0.349 for a key used ascending means it prefers the negatives**, which is the
budget failure stated as a number.

 **Do not quote the 0.999.** Those negatives are simulated, and simulated to have small
`parent_dist`, so that figure confirms the mechanism and says nothing about the size of the
win. The real candidate pool contains harder negatives - genuinely separated neighbouring
cells, not just duplicates. The honest number has to come from the held-out run.

**Why this should survive the embryo change**, where a tuned threshold might not: the two
training embryos differ 12× in labelling density but their division geometry is nearly
identical (median `parent_dist` 6.17 vs 7.37 µm, symmetry 0.457 vs 0.469). A likelihood over
geometry is anchored to cell biology; a cap or threshold is anchored to a false-positive rate
driven by the density that does differ. The hidden test is a third embryo.

`tests/test_div_ranker.py` pins the sign, the two-sidedness (the property a monotone distance
key cannot have), NaN/absurd-input handling, and that
`BIOHUB_SAFE_DIV_RANK_MODE=geometric` reproduces the stock key exactly so the change stays
A/B-able.

### Session 5 infrastructure

* `src/biohub_div_metric.py` + `tests/test_div_metric_parity.py` - the patched division
  metric without `tracksdata`, exact parity on 10 synthetic graphs.
* `tests/test_notebook_div_metric.py` - re-checks the copy **as injected into the notebook**,
  against the reference. The port passing is not evidence the injected transcription does.
* `scripts/pubfork.py` - derives our variants from the pinned public 0.947 notebook as
  ordered patches, **asserting each applies exactly the expected number of times**. A
  silently-unapplied patch would complete the run and report a number answering a different
  question - session 1's lesson, in a new place.
* `scripts/score_dumps.py` - scores public prediction dumps with the patched vs pre-patch
  rule. Ran locally; the four films we hold ground truth for contain **1 division between
  them**, which is itself the finding: the division term cannot be measured on the local
  sample and all of this work has to happen on Kaggle.

### In flight at end of session

| notebook | machine | what it decides |
|---|---|---|
| `11_pub_repro` | GPU | does the public 0.947 stack reproduce under our account |
| `12_div_probe` | GPU | the true division Jaccard of the stock stack, and what its own post-process sweep selects once the compass is fixed |
| `14_div_biology` | CPU | whether image evidence (volume down, peak flat) separates dividers, as an upper bound from annotated negatives |
| `15_div_rank` | GPU (queued) | gates open + likelihood ranker, end to end | ### Image evidence for divisions replicates, and it is usable (`14_div_biology`, CPU, 9 min)

1,359 samples (151 labelled divisions + annotated single-child controls from the same films),
5,625 image frames read. Median value relative to lag −3, divider minus paired control:

| lag | −2 | −1 | 0 | **+1** | **+2** | **+3** | +4 |
|---|---|---|---|---|---|---|---|
| **volume** | −0.049 | −0.078 | +0.080 | **−0.266** | **−0.352** | **−0.267** | −0.124 |
| **peak** | 0.000 | −0.017 | −0.049 | −0.063 | −0.039 | −0.058 | −0.097 |
| mean | −0.013 | −0.029 | −0.033 | −0.134 | −0.151 | −0.137 | −0.103 | **A dividing nucleus loses ~35% of its half-max volume at +2 while its peak moves ~4%.**
The published result replicates on our own measurement, and `mean` sits between the two
exactly as the fixed-probe artefact predicts - which is the reason to measure volume and peak
separately and never mean alone.

Practical consequence: the discriminative window is **after** the proposed split (+1 to +3),
not before it. At inference the whole video is available, so a ranker may look forward.

Feature AUC for "does this node divide":

| feature | AUC |
|---|---|
| `sister_dist` (larger  division) | 0.737 |
| `mean_drop` | 0.705 |
| `volume_rel_lag+2` | 0.678 |
| `volume_post_min` | 0.648 |
| geometric key | 0.582 | **The geometry AUCs on this sample are confounded and must not be carried forward.**
Negatives here are *annotated* single-child nodes, and because only ~3.6% of cells carry a
label, their nearest annotated neighbour is far away. So `parent_dist` appears to favour
*small* distances (AUC 0.645), the opposite of what the 151-division inventory shows. The
image features are measured at the parent node and are unaffected by this; the geometry
features need the real predicted candidate pool, which is what `16_div_rank_offline` and the
held-out runs supply. This is the same trap as the rest of the project: a negative set that
is convenient rather than representative.

### The candidate pool is enormous, which is why the ranking key matters so much

Regenerating `add_safe_divisions_postlink`'s proposal loop over the public prediction dumps - with all distance gates *and* the mutual-nearest-neighbour requirement still on - yields
**~13,000 proposals per film** (156,788 across 12 film-configs). The global cap is
`0.00375 × n_edges`, i.e. a few dozen. So roughly **0.2% of the pool is ever spent**, and
which 0.2% is decided entirely by a key whose AUC against real divisions is 0.349.

This is the clearest statement of the opportunity: the pool already contains the divisions,
the budget is tiny, and the selection is anti-correlated with the target.

### Generative image term, fitted on positives only (for a later variant)

The geometry likelihood is a *generative* model of real divisions, so it is fitted on the 151
positives and never needs a negative set - which is exactly why it dodges the
unrepresentative-negatives trap that invalidated the geometry AUCs in `14_div_biology`. The
same construction works for the image features:

| feature | positives | controls | Cohen's d |
|---|---|---|---|
| `volume_drop` | N(0.569, 0.375) | N(0.819, 0.848) | **0.381** |
| `volume_post_min` | N(0.659, 0.539) | N(1.009, 1.239) | 0.366 |
| `volume_rel_lag+2` | N(0.904, 0.697) | N(1.516, 2.348) | 0.353 |
| `peak_post_min` | N(0.836, 0.242) | N(0.936, 0.607) | 0.217 | `d ≈ 0.38` for the best single feature - real but modest, and the volume features are
mutually correlated so combining them adds little. Note the controls' spread is 2-3× wider,
so a likelihood fitted on positives naturally penalises them.

**Deliberately not added to `15_div_rank` yet.** The geometry fix is a sign error worth a
large delta; the image term is worth perhaps a few points of AUC. Bundling them would make an
end-to-end result uninterpretable, and this project's log is a long record of exactly that
mistake. Geometry alone is measured first; the image term becomes a second variant only if
geometry alone falls short. The frames are already in memory at that point in the pipeline
(the DeepCenter veto holds a `frame_cache`), so the marginal cost when it is wanted is small.

### `16_div_rank_offline` - inconclusive, 2 positives in 468,454 candidates

The plan was to replace the simulated negatives with real ones by regenerating the
safe-division proposal loop over the public `zhincez/biohub-diagnostic-dumps` predictions.
It ran cleanly and produced **468,454 candidates containing 2 positives** (base rate
4.3e-6). The reported AUCs - stock 0.543, likelihood 0.915 - rest on **two** positive
examples and are worth nothing. **Do not quote them.**

Cause: the dumps cover 8 distinct films, and those films hold ~4 labelled divisions between
them (`44b6_12dfb391` 1, `6bba_05db0fb1` 3, the rest 0). The films were chosen by their
author to be the four visible test copies, not for division content. The earlier local run
found the same wall from the other side - the four films with local ground truth contain one
division.

**This is the third time this project has been bitten by the same thing**: an evaluation set
chosen for availability rather than for containing the phenomenon under test. Sessions 1-3
tuned on n=2 videos and every parametric "gain" evaporated at n=30; here the sample is
plentiful in rows and empty in *events*. For a rare-event metric the sample size that matters
is the number of **divisions**, not the number of candidates or films.

#### The one thing worth keeping from the run

On `6bba_05db0fb1`, `division_targets` resolved the parent side and both daughter lineages
for **3** divisions, yet the regenerated candidate pool contained **0-1** proposals that would
complete any of them - out of ~58,000 candidates in that film. So for those divisions the
proposal that would score is **not in the pool at all**, and no ranker can help.

That splits the problem in two, and the split was not visible before:

* divisions whose completing proposal **is** in the pool but never reaches the budget  the
  ranking fix (`15_div_rank`) addresses these;
* divisions whose completing proposal is **absent** from the pool  blocked earlier, by the
  `len(kids) != 1` precondition, the mutual-nearest-neighbour requirement, or the distance
  gates.

Which of those dominates decides whether the ranking fix is worth a lot or a little, and it
has to be measured on a properly powered sample. n=3 here is anecdote.

**Next**: re-run this analysis against the held-out validator predictions from `12_div_probe`
(24 division-rich films, chosen division-first) rather than the public dumps, and report the
pool-absent vs budget-starved split explicitly.

### A working model of the division score, written down *before* the results land

Stated in advance so the incoming numbers can confirm or refute it rather than be
rationalised after the fact.

The only published decomposition available (gates as shipped) is **145 forks proposed  1 TP,
1 FP, 6 FN** on a held-out set. One FP per 145 proposals is a **0.7%** charge rate, far below
the 3.6% labelled-cell fraction, because a fork is only charged when it is `evaluable`
(matched onto an annotated GT node that has a child) or `considered` (local to a GT division
window). Everything else is invisible.

That gives a two-parameter model over the whole 199-film set, with `N` forks proposed and
recall `r` against the 151 divisions:

```
TP  = 151 r
FP ~ 0.007 N
FN  = 151 (1 - r)
J   = TP / (TP + FP + FN) = 151 r / (151 + 0.007 N)
```

The `TP` terms cancel in the denominator, which is the useful part: **J depends on recall and
on the total fork count, not on precision per se.** At the current budget (~145 forks/film,
~29,000 over the set):

| recall | forks proposed | predicted J | predicted score |
|---|---|---|---|
| 0.10 | 29,000 | 0.043 | 0.943 |
| 0.30 | 29,000 | 0.129 | 0.951 |
| **0.46** | **29,000** | **0.198** | **0.958 - gold** |
| 0.46 | 14,500 | 0.276 | 0.966 |
| 0.46 | 7,000 | 0.345 | 0.973 | Two consequences that change what to tune:

1. **Halving the fork count is worth as much as a large recall gain.** Going from 29,000 to
   7,000 forks at fixed recall takes J from 0.198 to 0.345. So once the ranker works, the cap
   should come *down*, not up - the opposite of the instinct that a wider pool needs a bigger
   budget.
2. `SAFE_DIV_FRAME_FRAC_CAP` and `SAFE_DIV_GLOBAL_FRAC_CAP` are both already in
   `PP_SWEEP_KEYS`, so the notebook's own sweep will find this **provided the metric it
   sweeps against is the corrected one** - which is precisely what `12`, `15` and `18` change.
   Against the stock 2× division metric the sweep was being told false positives cost half
   what they do.

**Falsifiable prediction:** the corrected-metric pp-sweep in `12_div_probe` should select a
*smaller* `SAFE_DIV_*_FRAC_CAP` than the shipped 0.0076 / 0.00375. If it selects a larger one,
this model is wrong and the FP charge rate is not ~0.7%.

### Real-data parity for `biohub_div_metric` - and one claim we have *not* yet verified ourselves

Parity had only been shown on synthetic graphs. Running our port, the official
`tracking_cellmot.division_metrics`, and the pre-patch rule over the public prediction dumps
against real ground truth: **12 of 12 (config, film) pairs agree exactly** on TP/FN/FP.

 On this sample the pre-patch and patched rules give **identical** answers, because these
films carry 0-1 divisions and no fork was predicted in the right place - with no true
positives to inflate, the two rules cannot diverge. So the "public validator reads ~2× high"
figure (0.2500 vs 0.1250) is still the *published* measurement, not one we have reproduced.
It is consistent with the patch's stated purpose and with our synthetic hub-exploit case
(pre-patch scores the hub, patched does not), but it should be labelled second-hand until
`12_div_probe` returns a held-out set with real division true positives in it.

---

## Session 6 - 2026-09-21

Eight days left. Two GPU runs from session 5 completed; `15_div_rank` never got a slot and
had to be re-pushed.

### CORRECTION - the cap was never binding, so the ranking key was never the active constraint

Session 5 concluded the fork budget was "spent by a key with the wrong sign". The sign error
is real and provable, but it was **not** what limited the shipped stack. Measured across 42
film-runs of `12_div_probe`:

| stage | count |
|---|---|
| geometric candidates | 6,267 |
| rejected by the DeepCenter veto | **5,063 (81%)** |
| survived the veto | 1,204 |
| actually added | 996 |
| **skipped by the cap** | **0 - in every single film** | `cap_skipped = 0` everywhere. Everything surviving the veto was added, so the order it was
added in could not matter. The ranking key only becomes active once the pool exceeds the cap.

This does not retract the sign error - it re-dates when it bites. It is a good example of the
failure this log keeps recording: a mechanism was inferred from one published row (145 forks
1 TP/1 FP) and generalised before the funnel was measured.

### The true division numbers, patched metric, 24 held-out films

From `ppsweep_results.csv` (base config):

```
div_tp = 7    div_fp = 14    div_fn = 27     division_jaccard = 0.146
adjusted_edge_jaccard = 0.9124        proxy = 0.9270
```

34 GT divisions in the held-out set; we recover **7 - recall 21%**.

**And the shipped gates leave 23% of divisions reachable** (`13_div_inventory`, all 199 films).
23% of 34 ≈ 8. So the stack is already recovering **essentially every division its gates
permit**. The gates are the binding constraint, exactly as the inventory predicted - not the
veto, not the cap, not the ranker.

Confirmation of the 2× metric claim, previously second-hand:

| run | division rule | films | adj edge | implied division Jaccard |
|---|---|---|---|---|
| `11_pub_repro` | pre-patch | 8 | 0.9260 | **0.231** |
| `12_div_probe` | patched | 24 | 0.9124 | **0.146** | Different film sets so it is not an exact ratio, but the direction and size match.

### Fixing the metric bought nothing, because the sweep has no division candidates

The pp-sweep tries exactly seven configs: `base`, `tight55`, `bonus125`, `gap45`, `dcgap035`,
`gap2step40`, `reuse28`, `relaxed9`. **Not one touches a division parameter.** `PP_SWEEP_KEYS`
advertises 20 sweepable keys including all the `SAFE_DIV_*` ones, but the candidate list is a
hand-written shortlist of edge/relink/gap knobs.

So correcting the division metric changed the *reported* number and could not change the
*selection*. This is plausibly why 704 teams share one score to four decimals: they are all
sweeping the same seven non-division configs.

Both sweeps (stock metric on 8 films, patched metric on 24) selected **`tight55`
(`MOTION_RELINK_TIGHT_UM` 6.0  5.5)** - the same change the public "0.948" notebook makes.

### Opening the gates does what the inventory predicted

`19_div_rank_fast` (diverge and symmetry gates off, likelihood ranker, validator off):

| film | candidates | forks added | cap_skipped |
|---|---|---|---|
| `44b6_0113de3b` | 172  **716** | 55  94 | 0  **29** |
| `44b6_0b24845f` | 232  **1137** | 25  79 | 0  **35** |
| `6bba_05b6850b` | 22  **105** | 10  23 | 0  **21** |
| `6bba_05db0fb1` | 304  **2954** | 34  236 | 0 | Pool 4-10× larger and **the cap now binds on three of four films**, so from here the ranking
key is live.

### Relaxing the DeepCenter veto changes composition, not count

`20_div_open_fast` additionally drops `DEEPCENTER_SAFE_DIV_THRESHOLD` 0.20  0.08. Rejections
fall from ~81% to 15-36%, so post-veto candidates roughly double (432603, 8811885) - but
**`added` barely moves** (9494, 7979, 236258) because the cap absorbs the difference
(`cap_skipped` 2945, 035).

With the gates open the veto is no longer a gate on the *number* of forks, only on *which*
ones. Submission count is nearly identical (241,882 vs 241,935 rows), so `20` is close to a
duplicate of `19` and does not deserve its own submission slot until `19` is scored.

### Leaderboard, 2026-09-21 (3,779 teams)

| medal | rank | cutoff |
|---|---|---|
| gold | 15 | 0.9630 |
| silver | 188 | 0.9490 |
| bronze | 377 | 0.9480 | **704 teams now sit on 0.9470**, spanning ranks 239-942 - so landing *on* 0.947 is worth
nothing, 0.948 is bronze and 0.949 is silver. Our 0.866 is rank ~2,721.

### Submissions today

1. `11_pub_repro` - stock 0.947 stack + its own sweep selection (`tight55`). Pending.
2. `19_div_rank_fast` - gates open + likelihood ranker. Pending.

`15_div_rank` (gates + ranker **with** the validator) re-pushed: it is the run that says
whether division recall actually rose from 7/34, which neither fast submission can show.

---

## Session 7 - 2026-09-28 (deadline 2026-09-29)

### DECISIVE - opening the division gates fails, and the whole division thesis with it

Two submissions from 2026-09-21 scored:

| submission | LB |
|---|---|
| `11_pub_repro` - stock 0.947 stack + its own sweep pick (`tight55`) | **0.947** |
| `19_div_rank_fast` - gates open + likelihood ranker | **0.916** | **−0.031.** The held-out run (`15_div_rank`, patched metric, same 24 films as `12_div_probe`)
explains it exactly:

| config | div TP | div FP | div J | adj edge | proxy |
|---|---|---|---|---|---|
| stock gates | **7** | 14 | **0.146** | 0.9124 | 0.9270 |
| gates open | **6** | **57** | **0.066** | 0.9093 | 0.9159 | **Opening the gates recovered not one extra division - it lost one - while quadrupling false
positives.** Division Jaccard *halved*, and the adjusted edge Jaccard fell 0.003 on top.

This refutes the central prediction of sessions 5-6. `13_div_inventory` measured that the
`diverge` and `symmetry` gates block 79 of 114 real divisions, and inferred that opening them
would take reachability from 23% to 75%. Reachability rose; **recovery did not move**. The
inference was wrong in a specific, identifiable way:

> *A division being geometrically admissible is not the same as the completing proposal
> existing in the candidate pool.*

`16_div_rank_offline` had actually seen this and it was dismissed as anecdote: on
`6bba_05db0fb1`, three divisions had their parent side and both daughter lineages resolved,
yet **0-1** of ~58,000 candidates would have completed any of them. That was the real signal,
at n=3, and it was right. The gates were never the binding constraint - the **candidate
generator** is. A proposal only exists where the parent already has exactly one predicted
child and the true daughter is among the next frame's detections; for 27 of 34 divisions it
is not.

**7 of 34 is close to this generator's ceiling.** Recovering more needs a different generator
(the `len(kids) != 1` precondition and the mutual-NN requirement are the suspects), not
different thresholds - and that is not a one-day change.

#### Why the loss was −0.031 rather than −0.008

Division Jaccard is worth 0.1, so halving it from 0.146 costs ~0.008. The other ~0.023 is
edge damage: the run added ~108 forks/film against the stock ~31, and every added fork brings
nodes and edges that are counted as false positives and inflate the node-count penalty.
**The harm was volume, not selection.**

#### The sign error, finally placed

Across three sessions the `parent_dist + 0.15 * sister_dist` ascending key was called the
binding constraint, then re-dated to "only once the cap binds", and now has to be demoted
again. It is a genuine defect - 0 of 151 real divisions sit below 2 µm and the key sorts that
band first - but with the stock gates the cap never binds, and with the gates open the pool
contains almost no completing proposals to re-rank. **A correct ranker over a pool that lacks
the right candidates buys nothing.** The lesson is the one this log keeps re-learning from a
new angle: measure the funnel end to end before attributing a loss to one stage.

### Standing after this

Best is **0.947**, rank ~942 of 3,779 - no medal, because 704 teams share that exact score and
ties break by earliest submission. Bronze needs 0.948, silver 0.949.

In flight, both submission-only (~25 min each):

* `21_div_budget_fast` - gates open + ranker with **both caps cut 3.5×** so the fork count
  matches stock. Isolates selection from volume, the confound that sank `19`. Expected to land
  near 0.947 given TP did not rise with a wider pool; run because it is nearly free and it
  cleanly closes the question.
* `22_pub948` - a direct fork of the public `haideptry` 0.948 recipe (DivNet 3D mitosis gate
  on `giorgosi/biohub-divnet-v2`, `MOTION_RELINK_TIGHT_UM` 5.5, `DEEPCENTER_SAFE_DIV_THRESHOLD`
  0.25, validator off). With one day left, reproducing a **known** 0.948 is a better bet than
  inventing one, and 88 teams already sit on that score.

### Endgame decision - stop inventing, reproduce

With ~31 hours left and the division thesis refuted, the remaining value is in *reproducing*
public recipes that already score above the 0.947 block, not in new modelling. Scanning the
newest public kernels turned up two that claim more than the 0.948 we were chasing:

| notebook | claim | extra dataset |
|---|---|---|
| `haideptry/biohub-0-951-sota-deepcenter-fast-ilp-19m` | 0.951 | `giorgosi/biohub-divnet-v2` |
| `anvithpothula/biohub-0-953-lb-original` | 0.953 | `anvithpothula/biohub-v1284-head-s075` | Both were screened before running, on two axes:

* **the patched metric exploit** - searched for the out-of-volume hub (`t = -1000`,
  `(z,y,x) = -10000`), `FORKS`/`MAX_COMPONENTS` fake-fork chains, and `augment_dataset`.
  Zero hits in either.
* **ground-truth leakage** - `23_pub953` does read `train/`, which is the pattern that scores
  brilliantly in development and worthlessly on the hidden-test rerun. Inspected: it uses
  `train/` only for its held-out validator, and explicitly removes the test stems
  (`candidates = [s for s in train_stems_all if s not in test_stem_set]`). Same structure as
  the 0.947 stack. Clean.

Both ran cleanly (configuration guard PASS; 238,260 and 245,195 submission rows).

### Submissions, 2026-09-28 (4 of 5 used)

| # | notebook | what it is |
|---|---|---|
| 1 | `22_pub948` | public 0.948 recipe, DivNet 3D mitosis gate |
| 2 | `21_div_budget_fast` | gates open + likelihood ranker, caps cut 3.5× so fork count matches stock (131 vs 124) - isolates fork *selection* from fork *volume*, the confound that cost `19` its 0.031 |
| 3 | `23_pub953` | public 0.953 recipe |
| 4 | `24_pub951` | public 0.951 recipe | Three are independent shots at clearing 0.948. `21` is the only one that still tests our own
idea, and it is the honest version of the experiment: same number of forks, chosen by the
likelihood key instead of the ascending geometric one. If it beats 0.947 the ranker is worth
something after all; if it ties, the ranker is neutral and the whole division line is closed.

Standing: best **0.947**, rank **1368 of 3969** - the tie block keeps absorbing new teams, so
the same score is worth less each day. Bronze 0.948, silver 0.949.
