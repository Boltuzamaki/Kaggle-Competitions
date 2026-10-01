"""Build notebooks/01_explainer_eda.ipynb - the "understand the competition" notebook."""

from nbbuild import REPO, build, code, library_cell, md

CELLS = [
    md(r"""
# Biohub - Cell Tracking During Development
## A complete walk-through: what the data is, what you predict, and how you are scored

Biohub filmed **zebrafish embryos** with a light-sheet microscope during the first hours of
development. Your job is to take a raw 3D movie of a cube of embryo tissue and produce a
**cell lineage graph**: where every cell is at every timepoint, which cell at time *t*
becomes which cell at *t+1*, and where cells divide.

This notebook is the map of the territory. By the end you will know:

1. **What the input is** - the Zarr layout, the physical scale, how to read a frame fast.
2. **What the output is** - the node/edge CSV schema and what a "division" looks like in it.
3. **What the ground truth is** - and why its extreme sparsity changes everything.
4. **How the metric really works** - including the two quirks that decide the leaderboard.
5. **What a reasonable target looks like** - measured, not guessed.

---
### The one-paragraph summary

> Input: a `(T=100, Z=64, Y=256, X=256)` `uint16` volume - a ~104 µm cube of tissue over 100
> frames. Output: a list of cell centres `(t, z, y, x)` plus directed `t  t+1` links between
> them. Score: `adjusted_edge_jaccard + 0.1 × division_jaccard`, where node matching uses a
> hard **7 µm** radius and the "adjustment" *rewards you for predicting fewer nodes*.
"""),

    md("## 0. Setup\n\nThe shared library is written to disk here so this notebook runs identically on Kaggle and locally."),
    library_cell(),

    code(r"""
import sys, json, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
import biohub_ct as B

warnings.filterwarnings("ignore")

COMP = B.find_comp_dir()
TRAIN, TEST = COMP / "train", COMP / "test"
print("competition root :", COMP)
print("train exists     :", TRAIN.is_dir())
print("test  exists     :", TEST.is_dir())
print("voxel scale (z,y,x) µm/voxel :", B.SCALE)
"""),

    md(r"""
## 1. Inventory - what is actually shipped

Two kinds of thing live side by side, keyed by dataset name:

| Path | What it is |
|---|---|
| `train/{name}.zarr` | the image volume |
| `train/{name}.geff` | the ground-truth track graph |
| `test/{name}.zarr` | image volume only - no `.geff` | Dataset names are `{embryo_id}_{field_of_view}`. There are only **two embryos** in the whole
competition (`44b6` and `6bba`), so a field-of-view split is *not* an independent split - neighbouring crops of the same embryo share cells and imaging conditions. Split by embryo
if you want an honest estimate of generalisation, and be aware that with n=2 even that is coarse.
"""),

    code(r"""
train_names = B.list_datasets(COMP, "train")
test_names  = B.list_datasets(COMP, "test")

print(f"train datasets : {len(train_names)}")
print(f"test  datasets : {len(test_names)}  -> {test_names}")

from collections import Counter
print("\nvideos per embryo (train):", dict(Counter(B.embryo_of(n) for n in train_names)))
print("videos per embryo (test) :", dict(Counter(B.embryo_of(n) for n in test_names)))
"""),

    md(r"""
### The visible `test/` folder is a placeholder

Every name in `test/` **also appears in `train/`**, and the files are byte-identical - same
sizes, same chunk count. They are there so your notebook has something to run against while
you develop. At scoring time Kaggle swaps in the real, hidden test videos.

Two practical consequences:

* **Good:** you can score the exact public-test datasets locally, because their ground truth
  is sitting in `train/`.
* **Bad:** never tune against those four alone, and never read `train/{test_name}.geff` at
  inference time. It would look brilliant during development and score nothing on the rerun.
"""),

    code(r"""
overlap = sorted(set(test_names) & set(train_names))
print("test names that also exist in train:", overlap)

for n in overlap[:2]:
    try:
        a = B.open_volume(TEST / f"{n}.zarr")
        b = B.open_volume(TRAIN / f"{n}.zarr")
        print(f"  {n}: test shape {a.shape} == train shape {b.shape} -> {a.shape == b.shape}")
    except FileNotFoundError as e:
        print(f"  {n}: skipped ({Path(e.filename).name} not present in this copy of the data)")
"""),

    md(r"""
## 2. The image data

### Physical geometry

The array axes are `(T, Z, Y, X)` but the voxels are **anisotropic**:

```
z : 1.625   µm per voxel      64 voxels -> 104.0 µm
y : 0.40625 µm per voxel     256 voxels -> 104.0 µm
x : 0.40625 µm per voxel     256 voxels -> 104.0 µm
```

So the field of view is a **cube in physical space** even though it is not a cube in voxels - Z is 4× coarser than XY. Every distance you compute (linking gates, NMS radius, the 7 µm
matching threshold) must be in **micrometres**, never in voxels. Multiply by `B.SCALE` first.

A pleasant consequence: downsampling XY by exactly **4** makes the grid perfectly isotropic
at 1.625 µm, and shrinks a frame from 4.2 M to 262 K voxels. Almost every strong public
solution does this.
""" ),

    code(r"""
name = test_names[0]
vol = B.open_volume(TEST / f"{name}.zarr")

print("dataset :", name)
print("shape   :", vol.shape, "(T, Z, Y, X)")
print("dtype   :", vol.dtype)
print("physical size (z,y,x) µm :", np.array(vol.shape[1:]) * B.SCALE)
print("\nper-dataset intensity quantiles stored in the zarr metadata:")
for k, v in sorted(vol.quantiles.items(), key=lambda kv: float(kv[0])):
    print(f"   q={k:>6} -> {v:.1f}")
"""),

    md(r"""
Those quantiles are worth noticing: they are computed over the **whole movie** and shipped
inside `{name}.zarr/zarr.json`. That means you get a stable, per-dataset normalisation for
free - including on the hidden test set - instead of re-estimating percentiles per frame.
`B.normalize_frame` uses them when present.

### How a frame is stored, and how to read one quickly

The array is chunked as `(1, 64, 256, 256)` - **one chunk per timepoint** - blosc-compressed,
at the path `{name}.zarr/0/c/{t}/0/0/0`. You can therefore read frame *t* by decompressing a
single file, with no Zarr machinery in the loop. That is what `Volume.frame()` does, falling
back to `zarr` only if the fast path fails.
"""),

    code(r"""
import time

meta = json.loads((vol.path / "0" / "zarr.json").read_text())
print("chunk shape :", meta["chunk_grid"]["configuration"]["chunk_shape"])
print("codecs      :", [c["name"] for c in meta["codecs"]])

t0 = time.time()
frame = vol.frame(vol.n_t // 2)
print(f"\nread one frame in {time.time()-t0:.3f}s -> {frame.shape} {frame.dtype}")
print(f"intensity: min={frame.min()} median={np.median(frame):.0f} max={frame.max()}")
"""),

    md("### Looking at it\n\nThree orthogonal maximum-intensity projections plus the intensity histogram."),

    code(r"""
import matplotlib.pyplot as plt

lo, hi = np.percentile(frame, [1.0, 99.7])
fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

ax[0, 0].imshow(np.clip(frame.max(0), lo, hi), cmap="magma")
ax[0, 0].set(title=f"Z-projection (XY view) - {name}, t={vol.n_t//2}", xlabel="x voxel", ylabel="y voxel")

ax[0, 1].imshow(np.clip(frame.max(1), lo, hi), cmap="magma", aspect=B.SCALE[0]/B.SCALE[2])
ax[0, 1].set(title="Y-projection (XZ view)", xlabel="x voxel", ylabel="z voxel")

ax[1, 0].imshow(np.clip(frame.max(2), lo, hi), cmap="magma", aspect=B.SCALE[0]/B.SCALE[1])
ax[1, 0].set(title="X-projection (ZY view)", xlabel="y voxel", ylabel="z voxel")

ax[1, 1].hist(frame.ravel()[::37], bins=100, log=True, color="#444")
ax[1, 1].set(title="intensity distribution", xlabel="uint16 value", ylabel="voxels (log)")
plt.show()
"""),

    md(r"""
Note the aspect-ratio correction on the side views (`aspect=SCALE[0]/SCALE[2]`). Without it
the tissue looks squashed and you will misjudge how far cells move in Z.

### The movie

Cells are packed, roughly spherical, and drift coherently - neighbouring cells move together.
That coherence is exactly what a motion-aware linker exploits.
"""),

    code(r"""
from matplotlib import animation
from IPython.display import HTML

STRIDE = 5
ts = list(range(0, vol.n_t, STRIDE))
projections = [vol.frame(t).max(0) for t in ts]

fig, ax = plt.subplots(figsize=(5.5, 5.5))
im = ax.imshow(np.clip(projections[0], lo, hi), cmap="magma")
title = ax.set_title(f"{name} - t=0")
ax.set_axis_off()

def _update(i):
    im.set_data(np.clip(projections[i], lo, hi))
    title.set_text(f"{name} - t={ts[i]}")
    return im, title

anim = animation.FuncAnimation(fig, _update, frames=len(ts), interval=200, blit=False)
plt.close(fig)
HTML(anim.to_jshtml())
"""),

    md(r"""
## 3. The ground truth - and the fact that reframes the whole problem

Ground truth lives in `.geff` files (Graph Exchange File Format): a Zarr group holding

```
nodes/ids                    (N,)   node identifiers
nodes/props/{t,z,y,x}/values (N,)   coordinates, in VOXEL units
edges/ids                    (E,2)  (source_id, target_id)
```

plus metadata under `zarr.json  attributes.geff`, including one number you should care about
a great deal: **`estimated_number_of_nodes`**.
"""),

    code(r"""
# Pick a dataset that actually has ground truth - prefer one of the placeholder
# test videos so the same dataset can be scored later in this notebook.
gt_candidates = [n for n in test_names if (TRAIN / f"{n}.geff").is_file()] \
                or sorted(p.stem for p in TRAIN.glob("*.geff"))
gt_name = gt_candidates[0]
gt = B.read_geff(TRAIN / f"{gt_name}.geff")

print("dataset                    :", gt_name)
print("annotated nodes            :", gt.n_nodes)
print("annotated edges            :", gt.n_edges)
print("estimated real cell count  :", gt.meta.get("estimated_number_of_nodes"))
print("t range                    :", gt.t.min(), "->", gt.t.max())

frac = gt.n_nodes / float(gt.meta.get("estimated_number_of_nodes", np.nan))
print(f"\n>>> annotated fraction: {frac:.4%} of the cells in this video")
"""),

    md(r"""
### Read that number again

Roughly **one cell in five hundred** is annotated. A `.geff` typically contains a *single
hand-traced lineage* followed through the movie - around 50 nodes - while the video itself
holds ~25,000 cell instances.

This is not a nuisance detail; it is the central fact of the competition:

* You cannot train a dense detector with "everything unlabelled is background". The
  unannotated cells look exactly like the annotated ones.
* You are evaluated on your ability to follow **those specific lineages**, not on segmenting
  the whole embryo.
* Most of what your model outputs is never directly compared to anything.
"""),

    code(r"""
import collections

out_deg = collections.Counter(int(s) for s in gt.edges[:, 0])
in_deg  = collections.Counter(int(d) for d in gt.edges[:, 1])

print("nodes per timepoint :", dict(collections.Counter(collections.Counter(gt.t.tolist()).values())))
print("out-degree histogram:", dict(collections.Counter(out_deg.values())))
print("in-degree  histogram:", dict(collections.Counter(in_deg.values())))

pos = {int(i): int(t) for i, t in zip(gt.ids, gt.t)}
dt = np.array([pos[int(d)] - pos[int(s)] for s, d in gt.edges])
print("\nedge Δt values present:", np.unique(dt), " <- always exactly 1")
"""),

    md(r"""
**Every ground-truth edge spans exactly one frame.** The scorer explicitly drops any predicted
edge where `t_target − t_source ≠ 1`, so there is no point emitting "skip" edges to bridge a
gap - you must insert an intermediate node instead. (`B.close_gaps` does precisely this.)

A **division** is one node with **two** outgoing edges. Out-degree 3+ is biologically invalid
and the scorer silently discards the surplus edges.

### How far does a cell move between frames?

This sets your linking gate. Too tight and you break tracks; too loose and you swap identities
in dense regions.
"""),

    code(r"""
def step_distances(g):
    idx = {int(i): k for k, i in enumerate(g.ids)}
    p = g.physical()
    return np.array([
        np.linalg.norm(p[idx[int(d)]] - p[idx[int(s)]])
        for s, d in g.edges if int(s) in idx and int(d) in idx
    ])

geffs = sorted(TRAIN.glob("*.geff"))
print(f"scanning {len(geffs)} ground-truth graphs...")

all_steps, rows = [], []
for p in geffs:
    try:
        g = B.read_geff(p)
    except Exception:
        continue
    d = step_distances(g)
    all_steps.append(d)
    rows.append({
        "dataset": p.stem,
        "embryo": B.embryo_of(p.stem),
        "nodes": g.n_nodes,
        "edges": g.n_edges,
        "divisions": sum(1 for v in collections.Counter(int(s) for s in g.edges[:, 0]).values() if v >= 2),
        "est_total": g.meta.get("estimated_number_of_nodes", np.nan),
        "median_step_um": float(np.median(d)) if len(d) else np.nan,
    })

steps = np.concatenate([d for d in all_steps if len(d)]) if all_steps else np.array([])
print(f"\n{len(steps)} annotated frame-to-frame steps")
for q in [50, 90, 95, 99, 99.9, 100]:
    print(f"  p{q:<5} = {np.percentile(steps, q):6.2f} µm")
"""),

    code(r"""
import pandas as pd
df = pd.DataFrame(rows)

fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
ax[0].hist(steps, bins=60, color="#3b7dd8")
ax[0].axvline(np.percentile(steps, 99), color="crimson", ls="--",
              label=f"p99 = {np.percentile(steps,99):.1f} µm")
ax[0].set(title="cell displacement per frame", xlabel="µm"); ax[0].legend()

ax[1].hist(df["nodes"], bins=40, color="#3b7dd8")
ax[1].set(title="annotated nodes per video", xlabel="nodes")

ax[2].hist(df["est_total"].dropna(), bins=40, color="#3b7dd8")
ax[2].set(title="estimated_number_of_nodes per video", xlabel="cells")
plt.show()

display(df.groupby("embryo")[["nodes", "edges", "divisions", "est_total", "median_step_um"]]
          .agg(["mean", "median", "count"]).round(2))
print("\ntotal annotated divisions across all training videos:", int(df['divisions'].sum()))
"""),

    md(r"""
Two numbers to carry forward:

* **p99 of the per-frame step** tells you where to put `max_link_um`. Anything much larger
  just invites identity swaps.
* **Divisions are rare.** Across the whole training set there are only a handful of annotated
  forks. Combined with the 0.1 weight, this means divisions are a *garnish*, not the main
  course - and an aggressive division predictor will cost you more in edge false positives
  than it earns.

### Seeing a lineage on top of the pixels

Let's overlay the annotated track on the image and confirm it really does sit on a cell.
"""),

    code(r"""
order = np.argsort(gt.t)
tt, zz, yy, xx = gt.t[order], gt.z[order], gt.y[order], gt.x[order]
picks = np.linspace(0, len(tt) - 1, 6).astype(int)

# The image is identical in either split for a placeholder dataset, so fall
# back to the test copy when only that one is present locally.
try:
    gt_vol = B.open_volume(TRAIN / f"{gt_name}.zarr")
except FileNotFoundError:
    gt_vol = B.open_volume(TEST / f"{gt_name}.zarr")

R = 40  # crop half-width in XY voxels

fig, axes = plt.subplots(1, len(picks), figsize=(3 * len(picks), 3.4), constrained_layout=True)
for ax_, i in zip(axes, picks):
    t = int(tt[i]); zc, yc, xc = zz[i], yy[i], xx[i]
    f = gt_vol.frame(t)
    zs = slice(max(0, int(zc) - 3), int(zc) + 4)
    sub = f[zs].max(0)
    y0, x0 = max(0, int(yc) - R), max(0, int(xc) - R)
    ax_.imshow(np.clip(sub[y0:y0 + 2 * R, x0:x0 + 2 * R], lo, hi), cmap="magma")
    ax_.plot(xc - x0, yc - y0, "o", mfc="none", mec="#4ade80", ms=16, mew=2.2)
    ax_.set_title(f"t={t}", fontsize=10); ax_.set_axis_off()
fig.suptitle(f"{gt_name} - the annotated lineage, local Z-max projection", fontsize=12)
plt.show()
"""),

    code(r"""
fig = plt.figure(figsize=(11, 4.5), constrained_layout=True)

ax1 = fig.add_subplot(1, 2, 1)
p = gt.physical()[order]
ax1.plot(p[:, 2], p[:, 1], "-o", ms=3, color="#3b7dd8")
ax1.scatter(p[0, 2], p[0, 1], c="#4ade80", s=70, zorder=3, label="start")
ax1.scatter(p[-1, 2], p[-1, 1], c="crimson", s=70, zorder=3, label="end")
ax1.set(title="trajectory in XY", xlabel="x µm", ylabel="y µm"); ax1.legend(); ax1.set_aspect("equal")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot(p[:, 2], p[:, 1], p[:, 0], "-o", ms=2.5, color="#3b7dd8")
ax2.set(xlabel="x µm", ylabel="y µm", zlabel="z µm", title="trajectory in 3D")
plt.show()
"""),

    md(r"""
## 4. The metric, precisely

From the organizers' `metrics.py`:

```
score = adjusted_edge_jaccard + 0.1 × division_jaccard
```

**Step 1 - node matching.** Predicted nodes are matched to GT nodes within the same timepoint
by an optimal bipartite assignment on centroid distance, with a hard cutoff of **7 µm**.
Each predicted node matches at most one GT node.

**Step 2 - edge classification.** A predicted edge is a **TP** when both endpoints match GT
nodes that are themselves joined by a GT edge.

**Step 3 - the false-positive rule (the important one).** An edge is only *eligible* to be a
false positive if at least one endpoint matches a GT node that has a GT edge:

```python
pred_valid = out_valid | in_valid       # source matches a GT node with out-degree > 0, or
                                        # target matches a GT node with in-degree > 0
edge_fp = n_valid_pred_edges - edge_tp
```

Every edge among the ~99.8% of unannotated cells is **neither TP nor FP - it is invisible**.

**Step 4 - the node-count adjustment.**

```
adjusted = max(0, J × (1 − 0.1 × (N_pred − N_true) / N_true))
```

`N_true` is the `estimated_number_of_nodes` from the GEFF metadata. Note the sign: if you
predict **fewer** nodes than `N_true`, the ratio is negative and the multiplier goes **above
one**, up to 1.1 in the limit.

### Proving it: score the ground truth against itself
"""),

    code(r"""
try:
    sys.path.insert(0, str(Path("../src").resolve()))
    from local_eval import oracle_upper_bound, summarise_rows, format_summary
    row = oracle_upper_bound(TRAIN / f"{gt_name}.geff")
    print("Scoring the ground truth against itself:\n")
    for k in ["edge_tp", "edge_fp", "edge_fn", "num_pred_nodes",
              "total_node_ratio", "edge_jaccard", "adj_edge_jaccard"]:
        print(f"  {k:<18} {row[k]}")
    print("\n" + format_summary(summarise_rows([row])))
except Exception as e:
    print(f"local_eval unavailable here ({type(e).__name__}: {e}).")
    print("It needs `pip install tracksdata geff polars` plus the organizer repo.")
    print("Expected result: edge_jaccard = 1.0, adj_edge_jaccard ≈ 1.0998")
"""),

    md(r"""
### A perfect prediction scores **≈ 1.0998**, not 1.0

The oracle emits only the 52 annotated nodes, so `total_node_ratio ≈ −0.998` and the
multiplier is `1 + 0.1 × 0.998 ≈ 1.0998`. Add a perfect division Jaccard and the theoretical
ceiling is about **1.2**.

This is not a loophole to be exploited so much as a **gradient telling you what to optimise**:

> Emitting fewer, better nodes is rewarded twice - a cleaner graph *and* a larger multiplier.

That is why every strong public solution ends with aggressive filtering: prune isolated nodes,
drop connected components shorter than ~6 frames, and keep only confident detections. The
annotated lineages are long, so length-based filtering removes spurious detections far faster
than it removes real ones. The current leaderboard tops out around **0.949**, comfortably
inside the 1.2 ceiling - there is real headroom left.

### Division Jaccard

A predicted fork must be a genuine parent  two-daughters structure, matched within 7 µm, with
a ±1 frame tolerance on *when* the split happens. It contributes only `0.1 ×` its value, and
since divisions are rare, a wrong fork usually costs more in edge FPs than the division TP is
worth. Be conservative.

## 5. The submission format

One flat CSV for all datasets, two row types sharing a schema. Unused fields are `-1`.
"""),

    code(r"""
print(",".join(B.SUBMISSION_COLUMNS))
print()
print("0,44b6_0113de3b,node,1,0,32,128,128,-1,-1    <- cell centre: node 1 at t=0, (z,y,x)=(32,128,128)")
print("1,44b6_0113de3b,node,2,1,32,128,128,-1,-1    <- cell centre: node 2 at t=1")
print("2,44b6_0113de3b,edge,-1,-1,-1,-1,-1,1,2      <- node 1 becomes node 2")
"""),

    md(r"""
Rules that are easy to get wrong:

| Rule | Consequence if broken |
|---|---|
| `node_id` unique **within a dataset** | edges resolve to the wrong cell |
| coordinates in **voxel** units | the scorer applies `SCALE` itself; passing µm mis-places everything by ~2.5× |
| edges must satisfy `t_target = t_source + 1` | silently dropped |
| out-degree ≤ 2 | surplus edges silently dropped |
| `id` column strictly increasing from 0 | malformed CSV | `B.validate_submission()` checks all of these before you burn one of your 5 daily submissions.

## 6. Where this leaves you

**The task.** Detect ~250 cells per frame in a 104 µm cube, 100 frames, and link them through
time - with essentially no dense supervision.

**What is graded.** Your accuracy on a handful of hand-traced lineages, plus a penalty (or
bonus) on your total node count.

**The shape of a strong solution**, as visible in the public code:

1. **Detect** - 3D U-Net heatmap (or multi-scale Difference-of-Gaussians for a training-free
   baseline)  local-maximum suppression with a physical radius.
2. **Refine** - sub-voxel centroid placement. The 7 µm gate is generous, but Z is coarse.
3. **Link** - Hungarian assignment on physical distance, made motion-aware by predicting each
   track forward from its velocity.
4. **Repair** - bridge one-frame dropouts with interpolated nodes (each bridge converts 0
   matchable edges into 2).
5. **Prune** - drop short components. This is where the node-count multiplier is won.
6. **Divisions** - add forks only where the geometry is unambiguous.

Notebook `02` implements exactly that with no training and is submittable as-is. Notebooks
`03`/`04` replace step 1 with a learned detector.
"""),
]

if __name__ == "__main__":
    out = build(REPO / "notebooks" / "01_explainer_eda.ipynb", CELLS)
    print("wrote", out)
