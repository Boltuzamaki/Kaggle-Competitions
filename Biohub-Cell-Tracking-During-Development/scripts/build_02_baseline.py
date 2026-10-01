"""Build notebooks/02_baseline_classical.ipynb - training-free, submittable end-to-end."""

from nbbuild import REPO, build, code, library_cell, md

CELLS = [
    md(r"""
# Baseline - classical detect  link  repair  prune

**No training. No model weights. No external dataset.** Attach the competition data, run top
to bottom, submit. Roughly 30 s per test video.

This exists for three reasons: it establishes a floor, it exercises the whole submission path
before any model is involved, and it is the fallback if a learned pipeline misbehaves on the
rerun.

### The pipeline

| Stage | What it does | Why |
|---|---|---|
| 1. **Detect** | multi-scale Difference-of-Gaussians  local-max NMS | training-free blob finder; XY is downsampled 4× so the grid is isotropic at 1.625 µm |
| 2. **Refine** | background-subtracted centroid at full resolution | peaks land on a coarse lattice; the 7 µm match gate deserves sub-voxel placement |
| 3. **Link** | Hungarian assignment, motion-aware, 7 µm gate | 7 µm is the p99 of the annotated per-frame displacement |
| 4. **Repair** | insert interpolated nodes across 1-frame dropouts | every GT edge spans exactly one frame, so a bridge converts 0 matchable edges into 2 |
| 5. **Prune** | drop connected components shorter than 4 nodes | removes detector noise *and* shrinks the node count, which raises the score multiplier |
| 6. **Divide** | add a second daughter only where geometry is unambiguous | divisions are worth just 0.1×, a wrong fork costs more than it earns | Measured on `44b6_0113de3b` against its ground truth: **score ≈ 0.918**
(edge Jaccard 0.904, node recall 0.98).

> **Reminder:** the visible `test/` datasets are copies of training videos. The local score
> cell below is a development sanity check on n=1-4 videos with ~50 GT edges each - treat it
> as a smoke test, not as a leaderboard estimate.
"""),

    md("## Setup"),
    library_cell(),

    code(r"""
import sys, time, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
import biohub_ct as B

warnings.filterwarnings("ignore")

COMP = B.find_comp_dir()
TEST = COMP / "test"
test_names = B.list_datasets(COMP, "test")

print("competition root :", COMP)
print(f"test datasets    : {len(test_names)}")
print(test_names)
"""),

    md(r"""
## Configuration

Every constant is in physical units (µm), because the anisotropic voxels make voxel-space
thresholds meaningless. The two that matter most:

* `max_link_um = 7.0` - the p99 of annotated frame-to-frame displacement. Wider gates start
  swapping identities in dense regions; the sweep showed 10 µm costs ~0.09 score.
* `min_track_len = 4` - long enough to kill detector noise, short enough to keep a real
  lineage the detector saw intermittently. Going to 6 dropped a genuine track in testing.
"""),

    code(r"""
# Defaults live in biohub_ct.Config, which is the single source of truth --
# restating them here is how a notebook silently drifts from the tuned values.
cfg = B.Config()
for k, v in vars(cfg).items():
    print(f"  {k:<18} {v}")
"""),

    md("## Run inference over the test set\n\nRows are streamed to disk as each dataset finishes - a full submission can reach millions of rows, and holding it all in a DataFrame is a reliable way to hit the memory ceiling."),

    code(r"""
from tqdm.auto import tqdm

SUBMISSION = "submission.csv"
t_start = time.time()

with B.SubmissionWriter(SUBMISSION) as writer:
    for name in test_names:
        t0 = time.time()
        vol = B.open_volume(TEST / f"{name}.zarr")
        frames = B.detect_frames(vol, cfg, progress=lambda r: tqdm(r, desc=f"{name} detect", leave=False))
        g = B.build_graph(frames, cfg)
        writer.add(name, g)
        print(f"{name}: T={vol.n_t}  peaks/frame={np.mean([len(f) for f in frames]):6.1f}  "
              f"nodes={g.n_nodes:6d}  edges={g.n_edges:6d}  ({time.time()-t0:.1f}s)")

print(f"\ntotal {time.time()-t_start:.1f}s -> {writer.n_nodes} nodes, {writer.n_edges} edges")
"""),

    md(r"""
## Validate before submitting

A malformed submission scores zero and still costs one of your five daily attempts. This
checks the header, that every edge references a node that exists, that all edges span exactly
one frame, and that no node has out-degree above 2.
"""),

    code(r"""
stats = B.validate_submission(SUBMISSION, expected_datasets=test_names)
for k, v in stats.items():
    print(f"  {k:<22} {v}")

assert stats["edges_wrong_dt"] == 0, "edges must connect t -> t+1"
assert stats["nodes_with_outdeg_gt2"] == 0, "a cell may have at most two daughters"
print("\n submission.csv is well-formed")

import pandas as pd
display(pd.read_csv(SUBMISSION, nrows=6))
print("file size: %.1f MB" % (Path(SUBMISSION).stat().st_size / 1e6))
"""),

    md(r"""
## Development-only: local score

This runs **only** when a visible test dataset also exists in `train/` - i.e. during
development against the placeholder test folder. On the real rerun the intersection is empty
and the cell prints nothing.

Nothing here touches the submission; it is read-only diagnostics.
"""),

    code(r"""
TRAIN = COMP / "train"
overlap = [n for n in test_names if (TRAIN / f"{n}.geff").exists()]

if not overlap:
    print("No ground truth available for the test datasets - this is the real test set.")
else:
    print(f"Development check on {len(overlap)} placeholder dataset(s): {overlap}\n")
    try:
        sys.path.insert(0, str(Path("../src").resolve()))
        from local_eval import score_dataset, summarise_rows, format_summary

        rows = []
        for name in overlap:
            vol = B.open_volume(TEST / f"{name}.zarr")
            g = B.build_graph(B.detect_frames(vol, cfg), cfg)
            r = score_dataset(g, TRAIN / f"{name}.geff")
            rows.append(r)
            print(f"  {name}: J={r['edge_jaccard']:.4f}  adj={r['adj_edge_jaccard']:.4f}  "
                  f"recall={r['node_recall']:.3f}  TP/FP/FN={r['edge_tp']}/{r['edge_fp']}/{r['edge_fn']}")
        print("\n" + format_summary(summarise_rows(rows)))
    except Exception as e:
        print(f"local_eval unavailable here ({type(e).__name__}: {e}).")
        print("It needs tracksdata + the organizer repo - run scripts/run_local_cv.py locally.")
"""),

    md(r"""
## Where the remaining score is

Diagnostics from the development run point at three specific gaps:

1. **Node recall ≈ 0.98, edge Jaccard ≈ 0.90.** Detection is nearly saturated; the losses are
   in *linking*, not finding. A learned edge scorer beats pure distance here.
2. **The DoG detector finds ~220 peaks/frame against ~257 estimated cells.** The misses are
   dim and clustered cells that a fixed band-pass filter cannot separate - exactly what a
   learned detector fixes. That is notebook `03`.
3. **Divisions contribute nothing yet.** They are worth up to +0.1, but only with a real
   division classifier; geometric rules mostly add false positives.

Notebook `03` trains the detector; notebook `04` swaps it into stage 1 of this same pipeline.
"""),
]

if __name__ == "__main__":
    print("wrote", build(REPO / "notebooks" / "02_baseline_classical.ipynb", CELLS))
