# Biohub: Cell Tracking During Development

Kaggle [`biohub-cell-tracking-during-development`](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development),
a $60k research competition. Reconstruct cell lineage graphs from 3D+time light-sheet
microscopy of a developing zebrafish embryo, including cell divisions.

Notebook-only submissions, 5 per day, 4,020 teams.

**Result: 0.953 public / 0.917 private, rank 972 of 4,020. No medal.**

Data, virtual environments and third-party notebooks are excluded from the repo
(see [Section 8](#8-what-is-not-here)).

## 1. The problem

**Input.** One movie per dataset, `(T, Z, Y, X) = (100, 64, 256, 256)` uint16, OME-Zarr v3,
one blosc2 chunk per timepoint. Voxels are anisotropic at `(1.625, 0.40625, 0.40625)` um,
so the field of view is a 104 um cube. 199 training movies, 4 visible test movies.

**Output.** A lineage graph flattened to CSV: node rows carry `(t, z, y, x)` in voxel units,
edge rows carry `(source_id, target_id)`. A division is one node with two outgoing edges.
Edges must satisfy `t_target = t_source + 1`.

**Score.**

```
score = adjusted_edge_jaccard + 0.1 * division_jaccard
adjusted_edge_jaccard = max(0, J * (1 - 0.1 * (N_pred - N_true) / N_true))
```

Node matching is optimal bipartite assignment on centroid distance with a hard 7 um cutoff.

## 2. Three structural facts

These came out of the first session and shaped everything after it.

**Ground truth is radically sparse.** Each label file holds roughly 50 annotated nodes,
usually a single hand-traced lineage, against an `estimated_number_of_nodes` near 25,000.
Grading happens on about 0.2 percent of the cells. Across all 199 movies there are only
**151 labelled divisions**.

**Most predictions are invisible to the metric.** A predicted edge can only be a false
positive if one endpoint matches an annotated ground-truth node. Edges among unannotated
cells are neither true nor false positives. The only brake on over-prediction is the
node-count term.

**Under-predicting nodes is rewarded.** When `N_pred < N_true` the multiplier exceeds 1.
Scoring the ground truth against itself gives 1.0998, not 1.0.

## 3. What was built

```
src/
  biohub_ct.py          detection, linking, post-processing, Zarr v3 and .geff readers
  biohub_unet.py        3D U-Net detector and its training loop
  biohub_metric.py      edge half of the official metric, numpy only
  biohub_div_metric.py  division half of the official metric, numpy only
  biohub_link.py        learned edge-scoring features (tested, not adopted)
  local_eval.py         thin wrapper over the organizers' scorer, for parity checks
scripts/                notebook builders, sweeps, Kaggle push/submit, log readers
notebooks/              the Kaggle notebooks, generated from scripts/build_*.py
tests/                  unit tests on synthetic fixtures, plus parity tests vs the reference
docs/experiment_log.md  append-only record of every experiment, including failures
results/                raw sweep outputs
```

Two pieces are worth calling out.

**A dependency-free reimplementation of the official metric.** The organizers' scorer needs
`tracksdata`, `geff` and `polars`, none of which exist in the Kaggle image. That confined
honest evaluation to two locally downloadable movies, a sample so small that a single edge
moved the Jaccard by 0.02. `biohub_metric.py` and `biohub_div_metric.py` reproduce both
halves in numpy, so the pipeline can be scored against all 199 training movies inside a
Kaggle notebook. Both are pinned to the reference by parity tests.

**The division metric was patched mid-competition and most of the field never noticed.**
On 2026-07-17 the organizers closed an exploit where a single out-of-volume hub node, wired
to the root of every track, made any merely-detected division count as recovered. The widely
forked public stack still scores divisions with the pre-patch rule in its own validator, so
it reports roughly double the official value. `biohub_div_metric.py` implements the patched
rule, verified against the reference on ten synthetic graphs that each target one clause,
plus twelve real predicted graphs.

## 4. The pipeline

| stage | approach |
|---|---|
| detection | 3D U-Net heatmap, peaks taken as top-K per frame with K set from a classical detector's count |
| linking | motion-aware Hungarian assignment with physical gates |
| post-processing | gap closing, short-track pruning, line-fit trajectory smoothing, conservative division insertion |

The final submissions ran on the public `tracking_cellmot` stack rather than this pipeline.
Section 6 explains why.

## 5. What worked

Measured on 30 to 60 movie cross-validation inside Kaggle, not on the two-movie local sample.

| change | effect | notes |
|---|---|---|
| Gap closing across 1-frame dropouts | +0.017 | largest post-processing gain |
| Rewriting the detector loss | +0.041 | see below |
| Line-fit trajectory smoothing (weight 0.8, window 3) | +0.0135 | transferred to the leaderboard at +0.014 |

The detector loss rewrite is the one result that generalised cleanly. Two multi-hour training
runs produced a U-Net that lost to a classical difference-of-Gaussians detector. The diagnosis
was that a count-normalised binary cross-entropy on a near-binary target admits a degenerate
smooth solution: the network found cells but localised them 4 to 9 um from their centres,
against a 7 um matching gate and 1.7 um of real per-frame motion, so jitter swamped the actual
displacement. Replacing it with a CenterNet-style penalty-reduced focal loss on a soft Gaussian
target, computed only inside a small cube around each annotation, moved median localisation
from 8.95 um to 1.68 um and node recall from 0.476 to 0.940. The new loss was validated on
synthetic blobs before any GPU time was spent on it.

## 6. What did not work

The failures are the more useful half of this repo and are recorded in full in
[`docs/experiment_log.md`](docs/experiment_log.md).

**Every parametric gain evaporated at scale.** Three sweeps of 162, 36 and 13 configurations
found nothing worth adopting. A density-adaptive suppression radius measured +0.016 on two
movies and +0.003 on thirty. A nearest-neighbour spacing change measured +0.016 on two and
+0.001 on thirty. The standing rule that came out of this is that no parameter change is
adopted on fewer than about 30 movies.

**Higher detection recall lowered the score.** Merging independently suppressed detector runs
raised node recall from 0.725 to 0.961 and dropped the score from 0.74 to 0.54. Error
attribution showed why: going from a sparse to a dense candidate set left detection misses
unchanged at 9 to 11 but exploded wrong links from 3 to 26. The per-frame linker, not the
detector, was the cap.

**The division work, which was the main thesis for three sessions, failed.** The reasoning
chain was: the division term is worth 0.1 and the public stack collects 0.0125 of it; the
shipped gates leave only 23 percent of real divisions reachable; two of those gates sit at the
median of the real distribution; therefore opening them should roughly triple what is
reachable. Opening them recovered **zero** extra divisions on a 24-movie held-out set, lost
one, and quadrupled false positives. Leaderboard cost: 0.947 to 0.916.

The error was specific and worth stating plainly. A division being geometrically admissible is
not the same as the completing proposal existing in the candidate pool. An earlier offline
check had actually seen this (three divisions fully resolved, yet zero to one of 58,000
candidates would have completed any of them) and it was dismissed as too small a sample. It
was the real signal. The candidate generator, not the gates, is the ceiling.

A follow-up that held the number of inserted forks fixed and changed only which forks were
chosen scored 0.915, the same loss. That ruled out the volume explanation and showed the
replacement ranking key itself was worse than the geometric one it replaced.

**A learned edge scorer did not beat plain distance.** Thirteen features over gated candidate
pairs reached 4-fold ROC-AUC 0.965 against 0.969 for distance alone. Within an already-gated
candidate set the geometry is close to sufficient.

## 7. Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy scikit-image pandas torch blosc2 zstandard

PYTHONPATH=src python -m pytest tests/ -q        # 29 pass, 31 skip without data
python scripts/build_13_div_inventory.py          # regenerate a notebook
python scripts/kaggle_ops.py push 13_div_inventory
python scripts/kaggle_ops.py submit 13_div_inventory -m "..."
```

Notebooks are generated rather than hand-edited so that `src/` stays the single source of
truth. `scripts/nbbuild.py` asserts that every inlined module matches its `src/` original
before writing, after an early run was lost to a notebook that was valid Python and the wrong
file.

Tests that need the competition data, the organizers' reference stack or a forked public
notebook skip loudly rather than silently passing, so a clean checkout reports 29 passed and
31 skipped.

`scripts/pubfork.py` derives variants of the public stack as ordered textual patches and
requires each patch to apply exactly the expected number of times. A patch that silently fails
to apply would produce a run that completes, reports a number, and answers a different
question.

## 8. What is not here

| excluded | why |
|---|---|
| `data/` | 87 GB of competition movies, download from Kaggle |
| `.venv/` | virtual environment |
| `third_party/` | the organizers' reference repo, clone from `royerlab/kaggle-cell-tracking-competition` |
| `public_kernels/` | other people's published notebooks, not ours to redistribute |
| Patched and copied public notebooks | the final submissions were forks of public work; `scripts/pubfork.py` records exactly what was changed, which is the part that is ours |
| `*.pt`, `submission*.csv` | model weights and generated predictions |

## 9. Environment notes

Things that cost real time and are not obvious from the outside.

- `zarr`, `numcodecs` and `tracksdata` are **not installed** in the Kaggle image. Label files
  are read directly from their zstd chunks using `zstandard`, and image chunks with `blosc2`.
  `src/biohub_ct.py` tries four zstd backends and reports which it used rather than failing
  opaquely.
- Kaggle may assign a P100, whose compute capability the stock PyTorch build has no kernels
  for. Every CUDA op then fails with an error that looks like a code bug. Setting
  `machine_shape` to a T4 avoids it.
- Kaggle derives a notebook's URL slug from its title, not its id, and silently uses the title
  when they disagree.
- Code-competition submissions require an explicit version number and the CLI cannot list
  versions, so `scripts/kaggle_ops.py` records the number each push prints.
