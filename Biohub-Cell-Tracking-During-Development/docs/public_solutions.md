# What the strong public solutions actually do

Survey of the top public notebooks (by votes) plus the organizers' reference implementation.
Leaderboard context: top is **0.949**, a clean public baseline is **~0.908**, a pure classical
pipeline reaches **~0.72-0.73**.

Note on sourcing: Kaggle discussion threads render client-side and are not fetchable
programmatically, so the discussion-level findings below come from search snippets and from
what the notebooks themselves state. Everything attributed to code was read directly from the
downloaded notebooks in `public_kernels/`.

## The consensus architecture

Every notebook above ~0.90 is the same four-stage pipeline:

```
3D U-Net detector    learned edge scorer    ILP global linking    post-processing
   (heatmap +          (cross-attention        (pyscipopt; explicit    (gap closing, short-track
    local-max NMS)      transformer over        appearance /            pruning, motion relink,
                        candidate pairs)        disappearance /         conservative divisions)
                                                division costs)
```

Frequency of key terms across the top notebooks - `ILP`, `GAP_CLOSE`, `transformer`, `tta`,
`division` dominate. Nobody in the top tier links by distance alone.

### 1. Detection - 3D U-Net heatmap

Organizer baseline uses `TemporalUNet3D`: per-voxel features plus a single-channel detection
map, cell centres recovered by local-maximum suppression. Trained with **sparse supervision** - only annotated points are positives, everything else is a heavily down-weighted negative
(`neg_weight = 0.1`, positive and negative terms each count-normalised). This is the
non-obvious part and it is what makes learning from ~1 label per frame possible.

### 2. Linking - learned, not geometric

`SimpleNodeTransformer`: features pooled at detected centres, cross-attention scores every
`(t, t+1)` node pair. Loss is focal-weighted BCE over the pair matrix, masked to rows/columns
that contain an annotation. **This is the single biggest differentiator** - our own measurements
put node recall at ~0.92 while edge Jaccard lags well behind, confirming that linking, not
detection, is the bottleneck.

### 3. Global optimisation - ILP

Per-frame Hungarian assignment is greedy in time. The top solutions build one integer program
over the whole candidate graph with explicit costs. Typical calibrated values from
`yusuketogashi/clean-approach-lightweight-local-cv-no-hack`:

| parameter | value |
|---|---|
| detection threshold | 0.96875 |
| ILP appearance / disappearance weight | 0.0 / 1.575 |
| ILP edge weight | −1.0 |
| motion gate tight / relaxed | 6.0 / 9.5 µm |
| gap-close max gap / distance | 2 / 5.8 µm |
| min retained track length | 6 |
| safe-division parent / sister radius | 4.66 / 8.5 µm | Note the asymmetry: appearance costs **0**, disappearance costs **1.575**. Tracks are cheap to
start and expensive to end - which biases toward long, continuous tracks. That is exactly the
node-count-bonus logic showing up in a different guise.

### 4. Post-processing - where a surprising amount of score lives

In rough order of measured value:

1. **Gap closing.** Insert interpolated nodes across 1-2 frame dropouts. In our own ablation
   this was worth **+0.115** - larger than any other single step.
2. **Short-track pruning.** Drop components below ~4-6 nodes. Helps twice: cleaner graph and a
   smaller node count, which raises the `1 − 0.1·(N_pred−N_true)/N_true` multiplier.
3. **Motion-aware relinking** with tight/relaxed gates and a learned-probability bonus.
4. **Trajectory smoothing** (line fit over a ±2 frame window).
5. **Conservative division insertion**, capped as a fraction of nodes per frame.

Several notebooks expose all of this as environment variables, which makes A/B testing across
notebook versions cheap - a genuinely good pattern worth copying.

### Ensembling and TTA

`pilkwang/biohub-cell-tracking-two-seeds-logit-blend` averages **heatmap logits from two seeds
before NMS** (not after), plus flip TTA. Blending detections after NMS is much weaker because
the peak-picking has already discretised the evidence.

## The metric exploit, and its patch

A scoring exploit existed in the **division Jaccard**: fabricated fork structures pushed it from
~0 to ~1.0, adding nearly the full `+0.1` even though no added node or edge corresponded to a
real cell or a real division. Organizers patched it - a division now requires a genuine,
strongly-connected parent  two-daughter structure matched within the 7 µm radius.

Since the patch, `0.908` is treated as the legitimate clean baseline, and several top notebooks
explicitly advertise "no hack": no artificial hubs, no fake forks, no negative-time nodes, no
out-of-volume coordinates, no cross-clip edges, and - tellingly - **no hidden-test labels**.

That last one confirms what we verified independently: the four visible `test/` videos are
byte-identical copies of training videos whose ground truth ships in `train/`. Reading it scores
brilliantly in development and worthlessly on the rerun, since Kaggle substitutes the real
hidden test.

## What this repo takes, and what it leaves

**Taken:**
- XY downsample of 4 for an isotropic 1.625 µm grid
- multi-scale DoG with small kernels for the training-free baseline
- sparse-supervision detection loss (`neg_weight = 0.1`, count-normalised)
- motion-aware Hungarian linking with physical gates
- gap closing by node interpolation, short-track pruning, capped safe divisions
- dataset-level intensity quantiles from the zarr metadata for stable normalisation

**Deliberately not taken yet** - these are the ranked next steps:
1. **Learned edge scorer.** The measured gap between node recall (~0.92) and edge Jaccard says
   this is worth the most.
2. **ILP global linking** with appearance/disappearance asymmetry.
3. **Two-seed logit blending** before NMS.
4. **Per-dataset threshold calibration** so the node count lands near the expected cell count
   for each video rather than using one global threshold.

## Sources

- [royerlab/kaggle-cell-tracking-competition](https://github.com/royerlab/kaggle-cell-tracking-competition) - reference implementation and the authoritative metric
- `inversion/cell-tracking-getting-started-w-nearest-neighbor` - official starter, exact Zarr read path
- `pilkwang/biohub-cell-tracking-data-model-eda-baseline` - the clearest data-model write-up
- `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery`, `.../two-seeds-logit-blend`
- `yusuketogashi/clean-approach-lightweight-local-cv-no-hack` - calibrated parameter tables
- `romanrozen/strong-start-dog-band-pass-lb-0-73` - classical DoG reference point
