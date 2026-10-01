"""Build notebooks/04_inference_submit.ipynb - U-Net inference + submission."""

from nbbuild import REPO, build, code, library_cell, md

CELLS = [
    md(r"""
# Inference & submission - U-Net detector + graph tracker

**Attach two inputs before running:**

1. `Biohub - Cell Tracking During Development` (the competition data)
2. the dataset containing `unet_detector.pt` produced by notebook `03`

Writes `submission.csv` to `/kaggle/working`. Submit this notebook's version to the leaderboard.

### Pipeline

```
frame ──▶ U-Net heatmap ──▶ local-max NMS ──▶ sub-voxel refine
                                                      │
                     motion-aware Hungarian linking ◀──┘
                                   │
              gap closing ──▶ short-track pruning ──▶ safe divisions ──▶ CSV
```

Only stage 1 differs from notebook `02`. Everything downstream is shared, so any gain here is
attributable to the detector.

### Fail-safe

If the checkpoint is missing or fails to load, the notebook falls back to the classical DoG
detector rather than crashing. A submission notebook that throws on the rerun scores nothing;
one that degrades gracefully still scores.
"""),

    md("## Setup"),
    library_cell(),
    library_cell("biohub_unet.py"),

    code(r"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, ".")
import biohub_ct as B

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMP = B.find_comp_dir()
TEST = COMP / "test"
OUT  = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")

test_names = B.list_datasets(COMP, "test")
print("device        :", DEVICE)
print("test datasets :", len(test_names), test_names)
"""),

    md("## Locate and load the checkpoint\n\nThe search is deliberately broad so the notebook does not depend on the exact name you gave the attached dataset."),

    code(r"""
import biohub_unet as U

# Notebook 03 saves two checkpoints: the one selected by validation
# (`unet_detector.pt`) and the final epoch (`unet_detector_last.pt`). The
# selected one is the defensible default; the last epoch measured marginally
# better end-to-end on two videos (0.7740 vs 0.7697), which is inside the noise
# of that sample, so it is not preferred without larger evidence.
CHECKPOINT_NAME = "unet_detector.pt"

def find_checkpoint():
    roots = [Path("/kaggle/input"), Path("."), Path("..")]
    for root in roots:
        if not root.is_dir():
            continue
        for pat in (f"**/{CHECKPOINT_NAME}", "**/*.pt", "**/*.pth"):
            for p in sorted(root.glob(pat)):
                if "input/competitions" in str(p):
                    continue
                return p
    return None

CKPT = find_checkpoint()
model, meta = None, {}
if CKPT is None:
    print("  no checkpoint found - falling back to the classical DoG detector")
else:
    try:
        model, meta = U.load_detector(CKPT, device=DEVICE)
        print("loaded :", CKPT)
        print("  trained epochs :", meta.get("epoch"))
        print("  val top-K recall:", meta.get("val_topk_recall"))
        print("  val median dist :", meta.get("val_median_dist_um"), "µm")
    except Exception as e:
        print(f"  failed to load {CKPT}: {type(e).__name__}: {e}")
        print("   falling back to the classical DoG detector")
        model = None
"""),

    md(r"""
## Configuration

**Do not threshold this heatmap.** The detection loss gives positives a total weight of 1.0
against 0.1 for *all* negatives combined, so the output saturates - ~7% of voxels exceed 0.9 - and thresholding yields ~800 peaks/frame against ~260-330 real cells. Measured end to end: a
0.5 threshold scored **0.166**, top-K=320 scored **0.349**.

So `DET_THRESHOLD` stays at 0 and the *budget* selects. `BUDGET_FROM_DOG` sets K per frame from
the classical detector's own count, which tracks the true density closely and adapts to the
74-786 cells/frame spread across videos - a single global K cannot.

 Note the counter-intuitive result from `docs/experiment_log.md`: raising node recall by
proposing more candidates **lowers** the score, because the linker degrades faster than recall
improves. Prefer a smaller, better-ranked candidate set.
"""),

    code(r"""
# The heatmap is a RANKING, not a probability: the loss weights positives 1.0
# against 0.1 for all negatives combined, so ~7% of voxels exceed 0.9 and
# thresholding returns ~800 peaks/frame against ~260-330 real cells. Measured:
# threshold read-out scored 0.17, top-K scored 0.35. Always take top-K.
DET_THRESHOLD  = 0.0     # keep at 0 - the budget does the selecting
BUDGET_FROM_DOG = True   # per-frame K from the classical detector's count,
                         # which tracks true density (270 vs 258, 369 vs 328)
                         # and adapts to the 74-786 cells/frame spread
FLIP_TTA       = False   # 4x slower, typically a small recall gain
MIN_DIST_UM    = 3.5
MAX_PEAKS      = 4000    # fallback cap when BUDGET_FROM_DOG is off

# Linking/post-processing defaults come from biohub_ct.Config so this notebook
# cannot drift from the tuned values; only the detector knobs are set here.
cfg = B.Config()
for k, v in vars(cfg).items():
    print(f"  {k:<18} {v}")
print(f"\n  DET_THRESHOLD      {DET_THRESHOLD}\n  BUDGET_FROM_DOG    {BUDGET_FROM_DOG}\n  FLIP_TTA           {FLIP_TTA}")
"""),

    md("## Run"),

    code(r"""
from tqdm.auto import tqdm

SUBMISSION = OUT / "submission.csv"
t_start = time.time()
summary = []

with B.SubmissionWriter(SUBMISSION) as writer:
    for name in test_names:
        t0 = time.time()
        vol = B.open_volume(TEST / f"{name}.zarr")
        bar = lambda r: tqdm(r, desc=f"{name}", leave=False)

        if model is not None:
            frames = U.detect_frames_unet(
                vol, model, device=DEVICE, threshold=DET_THRESHOLD,
                min_distance_um=MIN_DIST_UM, max_peaks=MAX_PEAKS,
                budget_from_dog=BUDGET_FROM_DOG, dog_cfg=cfg,
                flip_tta=FLIP_TTA, refine=True, progress=bar)
        else:
            frames = B.detect_frames(vol, cfg, progress=bar)

        g = B.build_graph(frames, cfg)
        writer.add(name, g)
        summary.append({"dataset": name, "T": vol.n_t,
                        "peaks_per_frame": float(np.mean([len(f) for f in frames])),
                        "nodes": g.n_nodes, "edges": g.n_edges,
                        "seconds": round(time.time() - t0, 1)})
        print(f"{name}: peaks/frame={summary[-1]['peaks_per_frame']:6.1f}  "
              f"nodes={g.n_nodes:6d}  edges={g.n_edges:6d}  ({summary[-1]['seconds']}s)")

print(f"\ntotal {time.time()-t_start:.1f}s -> {writer.n_nodes} nodes, {writer.n_edges} edges")

import pandas as pd
display(pd.DataFrame(summary))
"""),

    md("## Validate"),

    code(r"""
stats = B.validate_submission(SUBMISSION, expected_datasets=test_names)
for k, v in stats.items():
    print(f"  {k:<22} {v}")

assert stats["edges_wrong_dt"] == 0, "edges must connect t -> t+1"
assert stats["nodes_with_outdeg_gt2"] == 0, "a cell may have at most two daughters"
assert stats["n_nodes"] > 0 and stats["n_edges"] > 0, "empty submission"
print("\n submission.csv is well-formed")
print("file size: %.1f MB" % (Path(SUBMISSION).stat().st_size / 1e6))
display(pd.read_csv(SUBMISSION, nrows=6))
"""),

    md(r"""
## Development-only diagnostics

Runs only when a visible test dataset also has ground truth in `train/` - i.e. against the
placeholder test folder. Empty on the real rerun. Read-only; it does not touch the submission.
"""),

    code(r"""
TRAIN = COMP / "train"
overlap = [n for n in test_names if (TRAIN / f"{n}.geff").exists()]

if not overlap:
    print("No ground truth for the test datasets - this is the real test set.")
else:
    print(f"Development check on {overlap}\n")
    for name in overlap:
        vol = B.open_volume(TEST / f"{name}.zarr")
        g_gt = B.read_geff(TRAIN / f"{name}.geff")
        by_t = {}
        for tt, zz, yy, xx in zip(g_gt.t, g_gt.z, g_gt.y, g_gt.x):
            by_t.setdefault(int(tt), []).append([zz, yy, xx])

        hit = tot = 0
        for t in sorted(by_t)[:15]:
            raw = vol.frame(t)
            dog_coords, _ = B.detect_dog(raw, vol.quantiles, cfg.xy_downsample, cfg.dog_scales,
                                         cfg.min_distance_um, cfg.rel_threshold, cfg.max_peaks)
            if model is not None:
                hm = U.predict_heatmap(model, raw, vol.quantiles, device=DEVICE)
                K = len(dog_coords) if BUDGET_FROM_DOG and len(dog_coords) else MAX_PEAKS
                coords, _ = U.peaks_from_heatmap(hm, MIN_DIST_UM, DET_THRESHOLD, K)
            else:
                coords = dog_coords
            gt = np.array(by_t[t])
            if len(coords):
                d = np.sqrt((((coords[:, None, :] - gt[None, :, :]) * B.SCALE) ** 2).sum(2)).min(0)
                hit += int((d <= 7.0).sum())
            tot += len(gt)
        est = g_gt.meta.get("estimated_number_of_nodes", float("nan"))
        n_pred = next(s["nodes"] for s in summary if s["dataset"] == name)
        print(f"  {name}: node recall@7µm = {hit/max(tot,1):.3f}   "
              f"nodes {n_pred} vs estimated {est}  "
              f"(multiplier {1 - 0.1*(n_pred-est)/est:.4f})")
"""),

    md(r"""
## Reading the diagnostics

* **node recall @ 7 µm** - the ceiling on your edge Jaccard. A cell the detector never finds
  can never be linked.
* **multiplier** - `1 − 0.1·(N_pred − N_true)/N_true`. Above 1 means you are under the
  estimated cell count and being rewarded; below 1 means you are over-predicting.

Getting both above ~0.95 and ~1.0 respectively is what separates a 0.8 from a 0.9+.

### Where to go next

1. **Learned edge scoring.** Distance-based linking is the biggest remaining loss. Train a
   small model on candidate `(node_t, node_{t+1})` pairs using local image features plus
   displacement; feed its probabilities as the assignment cost.
2. **Global optimisation.** Replace the per-frame Hungarian with an ILP over the whole
   candidate graph, with explicit appearance/disappearance/division costs. This is what the
   top public solutions do and it is worth several points.
3. **Threshold calibration per dataset.** Pick `DET_THRESHOLD` per video so the node count
   lands near the expected cell count, instead of using one global value.
4. **Ensembling.** Average heatmaps from two seeds before NMS.
"""),
]

if __name__ == "__main__":
    print("wrote", build(REPO / "notebooks" / "04_inference_submit.ipynb", CELLS))
