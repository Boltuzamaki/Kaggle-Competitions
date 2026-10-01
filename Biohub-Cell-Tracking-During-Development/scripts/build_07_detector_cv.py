"""Build notebooks/07_detector_cv.ipynb - DoG vs U-Net on the same many-video CV.

The point is a like-for-like comparison. Notebook 04 reports the U-Net on the four
placeholder videos, which is far too small to decide anything; this scores both
detectors on the same 40 training videos, stratified by density, using the metric
verified to match the organizers' scorer.
"""

from nbbuild import REPO, build, code, library_cell, md

CELLS = [
    md(r"""
# Detector comparison - DoG vs U-Net, on a real sample

**Needs GPU + the `unet_detector.pt` dataset from notebook `03`.**

### What this decides

Whether the learned detector is actually worth using. Prior evidence says detection is the
binding constraint - node recall correlates **+0.900** with score across 60 videos, and the
dense quartile scores 0.656 against 0.884 for sparse - but the only U-Net numbers so far come
from the two placeholder videos, which is not enough to decide anything.

Both detectors are scored on the **same** videos, stratified by density, so the difference is
attributable to the detector alone.

### Read-outs compared

The heatmap is a **ranking, not a probability**: the loss weights positives 1.0 against 0.1 for
all negatives combined, so ~7% of voxels exceed 0.9 and thresholding returns ~800 peaks/frame
against ~260-330 real cells. Measured on the placeholder videos, thresholding scored **0.166**
and top-K scored **0.349**. So three read-outs are compared:

| read-out | K per frame |
|---|---|
| `threshold` | unbounded above 0.5 - expected to be poor, included as the control |
| `topk_dog` | the classical detector's own peak count for that frame |
| `topk_fixed` | a single global K - expected to fail on the 74-786 cells/frame spread |
"""),

    library_cell(),
    library_cell("biohub_metric.py"),
    library_cell("biohub_unet.py"),

    code(r"""
import sys, time, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
import biohub_ct as B
import biohub_metric as M
import biohub_unet as U

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMP = B.find_comp_dir()
TRAIN = COMP / "train"
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
print("device:", DEVICE)

def find_checkpoint(prefer="unet_detector.pt"):
    hits = []
    root = Path("/kaggle/input")
    if root.is_dir():
        for p in sorted(root.rglob("*.pt")):
            if "input/competitions" not in str(p):
                hits.append(p)
    hits.sort(key=lambda p: (p.name != prefer, str(p)))
    return hits

CKPTS = find_checkpoint()
print("checkpoints found:", [p.name for p in CKPTS] or "NONE")
assert CKPTS, "attach the dataset containing unet_detector.pt from notebook 03"

model, meta = U.load_detector(CKPTS[0], device=DEVICE)
print(f"\nloaded {CKPTS[0].name}: epoch={meta.get('epoch')} "
      f"top-K recall={meta.get('val_topk_recall')} median_dist={meta.get('val_median_dist_um')}")
"""),

    md("## Evaluation set - stratified by density, seeded so it is reproducible"),

    code(r"""
N_VIDEOS = 40
SEED = 11

names = sorted(p.stem for p in TRAIN.glob("*.geff") if (TRAIN / f"{p.stem}.zarr").is_dir())
dens = []
for n in names:
    try:
        g = B.read_geff(TRAIN / f"{n}.geff")
        e = g.meta.get("estimated_number_of_nodes")
        if e:
            dens.append((n, float(e) / 100.0))
    except Exception:
        pass
dens.sort(key=lambda x: x[1])

# even coverage across the density range rather than a uniform random draw
idx = np.linspace(0, len(dens) - 1, N_VIDEOS).astype(int)
eval_set = [dens[i][0] for i in idx]
eval_dens = {dens[i][0]: dens[i][1] for i in idx}
print(f"{len(eval_set)} videos, density {min(eval_dens.values()):.0f}-{max(eval_dens.values()):.0f} cells/frame")
"""),

    code(r"""
cfg = B.Config()
MIN_DIST_UM = 3.5
FIXED_K = 300

def run_one(name, mode):
    vol = B.open_volume(TRAIN / f"{name}.zarr")
    gt = B.read_geff(TRAIN / f"{name}.geff")
    if mode == "dog":
        frames = B.detect_frames(vol, cfg)
    else:
        frames = U.detect_frames_unet(
            vol, model, device=DEVICE,
            threshold=0.5 if mode == "threshold" else 0.0,
            min_distance_um=MIN_DIST_UM,
            max_peaks=None if mode == "threshold" else (FIXED_K if mode == "topk_fixed" else 4000),
            budget_from_dog=(mode == "topk_dog"), dog_cfg=cfg,
            refine=True)
    g = B.build_graph(frames, cfg)
    r = M.evaluate(g, gt, n_total=gt.meta.get("estimated_number_of_nodes"))
    return r, float(np.mean([len(f) for f in frames]))

MODES = ["dog", "topk_dog", "topk_fixed", "threshold"]
rows, per_video = {m: [] for m in MODES}, []

for i, name in enumerate(eval_set):
    rec = {"dataset": name, "density": eval_dens[name]}
    for m in MODES:
        try:
            r, peaks = run_one(name, m)
        except Exception as e:
            print(f"  {name} [{m}] failed: {type(e).__name__}: {e}")
            continue
        rows[m].append(r)
        rec[f"{m}_adj"] = r.adj_edge_jaccard
        rec[f"{m}_recall"] = r.node_recall
        rec[f"{m}_peaks"] = peaks
    per_video.append(rec)
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(eval_set)} videos")

pv = pd.DataFrame(per_video)
pv.to_csv(OUT / "detector_cv_per_video.csv", index=False)

print()
for m in MODES:
    if rows[m]:
        print(f"{m:<12} {M.format_summary(M.summarise(rows[m]))}")
"""),

    md("## Where does the U-Net win or lose?\n\nAggregates hide the thing that matters: whether it rescues the dense videos, which is the entire reason for training it."),

    code(r"""
best_unet = "topk_dog"
pv["delta"] = pv[f"{best_unet}_adj"] - pv["dog_adj"]
pv["quartile"] = pd.qcut(pv["density"], 4, labels=["sparse", "med-low", "med-high", "dense"])

print("mean score by density quartile:\n")
display(pv.groupby("quartile", observed=True).agg(
    videos=("dataset", "size"),
    cells_per_frame=("density", "median"),
    dog=("dog_adj", "mean"),
    unet=(f"{best_unet}_adj", "mean"),
    delta=("delta", "mean"),
    dog_recall=("dog_recall", "mean"),
    unet_recall=(f"{best_unet}_recall", "mean"),
).round(4))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
ax[0].scatter(pv["dog_adj"], pv[f"{best_unet}_adj"], s=26)
lim = [0, 1]; ax[0].plot(lim, lim, "k--", lw=0.8)
ax[0].set(xlabel="DoG adj", ylabel="U-Net adj", title="per video")
ax[1].scatter(pv["density"], pv["delta"], s=26, color="#e07b39")
ax[1].axhline(0, color="k", lw=0.8, ls="--"); ax[1].set_xscale("log")
ax[1].set(xlabel="cells/frame (log)", ylabel="U-Net − DoG", title="does it rescue dense videos?")
ax[2].scatter(pv["dog_recall"], pv[f"{best_unet}_recall"], s=26, color="#4ade80")
ax[2].plot(lim, lim, "k--", lw=0.8)
ax[2].set(xlabel="DoG node recall", ylabel="U-Net node recall", title="node recall")
plt.show()

wins = int((pv["delta"] > 0).sum())
print(f"\nU-Net beats DoG on {wins}/{len(pv)} videos; mean delta {pv['delta'].mean():+.4f}, "
      f"median {pv['delta'].median():+.4f}")
print(f"on the densest quartile: mean delta {pv[pv['quartile']=='dense']['delta'].mean():+.4f}")
"""),

    md(r"""
## Decision rule

Adopt the U-Net for the submission only if it beats DoG on this sample by more than the
sampling noise - with 40 videos, roughly **±0.01**. A win concentrated in the dense quartile is
the outcome worth having, since that is where the score is lost.

If it loses, the checkpoint is undertrained or the read-out is wrong; check
`val_topk_recall` in the checkpoint metadata and confirm it is not saturated near 1.0 from the
first epoch - that failure has already cost one 4-hour run, and it is recorded in
`docs/experiment_log.md`.

A **hybrid** is also worth testing if the U-Net wins only on dense videos: use DoG where the
observed peak count is low and the U-Net where it is high.
"""),
]

if __name__ == "__main__":
    print("wrote", build(REPO / "notebooks" / "07_detector_cv.ipynb", CELLS))
