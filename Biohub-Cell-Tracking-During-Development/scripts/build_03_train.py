"""Build notebooks/03_train_unet.ipynb - trains the detector, saves weights."""

from nbbuild import REPO, build, code, library_cell, md

CELLS = [
    md(r"""
# Training a 3D U-Net cell detector under sparse point supervision

**Run this on a GPU notebook.** It trains a detector and writes
`unet_detector.pt` to `/kaggle/working`. Save the output as a Kaggle Dataset and attach it to
notebook `04`, which does inference and submits.

### The problem this has to solve

A training frame holds ~250 cells and typically **one** annotated node. Treating everything
unlabelled as background would teach the network to suppress ~249 real cells per frame - precisely the objects we need it to find.

### The first formulation failed - read this before changing the loss

The organizers' recipe is an asymmetric count-normalised BCE (positives total weight 1.0, all
negatives total 0.1). Trained for 24 epochs it produced a detector **worse than a random
network**:

| | top-K node recall | end-to-end score |
|---|---|---|
| random init | 0.377 | - |
| trained 24 epochs | ~0.48 | 0.125 |
| **classical DoG** | **0.886** | **0.7395** | The loss fell smoothly (0.120  0.046) while detection did not improve, and the peaks landed
**4-9 µm from cell centres** - hopeless against a 7 µm matching gate when real motion is only
~1.7 µm/frame. A near-binary target on a 1.6 µm grid gives no gradient saying *which* voxel is
the centre, and the weighting admits a degenerate smooth low-frequency solution.

### The current formulation: locally-masked CenterNet loss

`local_heatmap_loss` fixes both problems:

* **Soft Gaussian target + penalty-reduced focal loss** - a real gradient toward the exact
  centre, with easy background down-weighted.
* **Loss computed only inside a ±4-voxel (≈6.5 µm) cube around each annotation.** Outside those
  cubes the label is genuinely unknown, since most cells are unannotated, so contributing
  nothing there is the honest choice - and it removes the degenerate solution. The cube is
  comfortably inside the 13-27 µm cell spacing, so it does not swallow neighbours.
* A small global negative term stops the background saturating.

Sanity check on synthetic blobs: the old loss never localised; this one places peaks at
**exactly** the right voxel (0.0 error) within 120 steps.

### Why the classical detector needs replacing

Measured on the two placeholder datasets with ground truth:

| dataset | DoG node recall | median localisation error |
|---|---|---|
| `44b6_0113de3b` | 1.00 | 0.64 µm |
| `44b6_0b24845f` | 0.84 | 3.82 µm | The second video is denser and has a systematic ~2 µm bias in Z. A fixed band-pass filter
cannot adapt to that; a learned detector can.
"""),

    md("## Setup"),
    library_cell(),
    library_cell("biohub_unet.py"),

    code(r"""
import sys, time, math, json, random, warnings
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
import biohub_ct as B
import biohub_unet as U

warnings.filterwarnings("ignore")

SEED = 1234
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMP  = B.find_comp_dir()
TRAIN = COMP / "train"
OUT   = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")

print("device :", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
print("train  :", TRAIN)
"""),

    md(r"""
## Data split

There are only **two embryos** in the whole competition. Fields of view from the same embryo
share cells near their borders and share imaging conditions, so a random split leaks. Splitting
by embryo is the honest option but leaves n=1 for validation - so this uses a **grouped split
by dataset** and reports validation per embryo, which at least surfaces cross-embryo failure.
"""),

    code(r"""
VAL_FRACTION = 0.12

all_names = sorted(p.stem for p in TRAIN.glob("*.geff"))
rng = random.Random(SEED)

by_embryo = {}
for n in all_names:
    by_embryo.setdefault(B.embryo_of(n), []).append(n)

val_names = []
for emb, names in by_embryo.items():
    names = sorted(names); rng.shuffle(names)
    k = max(1, int(round(VAL_FRACTION * len(names))))
    val_names += names[:k]
val_names = set(val_names)
train_names = [n for n in all_names if n not in val_names]

print(f"{len(all_names)} datasets -> {len(train_names)} train / {len(val_names)} val")
for emb, names in by_embryo.items():
    print(f"  {emb}: {sum(n in val_names for n in names)} val of {len(names)}")
"""),

    code(r"""
t0 = time.time()
train_samples = U.index_training_frames(TRAIN, sorted(train_names))
val_samples   = U.index_training_frames(TRAIN, sorted(val_names))
print(f"indexed in {time.time()-t0:.1f}s")
print(f"annotated frames: {len(train_samples)} train / {len(val_samples)} val")
print(f"annotated points: {sum(len(s.coords) for s in train_samples)} train")
"""),

    md(r"""
## Model and optimiser

A small isotropic U-Net (3.1 M parameters) is plenty: the input is only 64³ and the target is a
sparse point map, not a fine segmentation. **InstanceNorm** rather than BatchNorm, because
batches are small and intensity statistics differ between embryos - per-sample normalisation is
steadier here.
"""),

    code(r"""
EPOCHS           = 24
BATCH_SIZE       = 8
LR               = 2e-3
LOSS_KIND        = "local_focal"   # "local_focal" (current) or "count_bce" (failed, see above)
WINDOW           = 4      # supervise +/-4 voxels ~ +/-6.5 um around each annotation
SIGMA            = 1.5    # Gaussian target width, voxels
GLOBAL_NEG_W     = 0.02   # keeps background from saturating
NEG_WEIGHT       = 0.1    # count_bce only
POS_RADIUS       = 1      # count_bce only
# Training is I/O bound, not compute bound: each sample is one blosc2 chunk
# decompress (~0.15 s) against a ~0.02 s forward+backward on a 64^3 volume.
# Workers matter far more than model size here.
NUM_WORKERS      = 4
TIME_BUDGET_S    = 6.5 * 3600   # stop cleanly before the Kaggle session limit

model = U.UNet3D(base=24, depth=3).to(DEVICE)
print("parameters: %.2f M" % (sum(p.numel() for p in model.parameters()) / 1e6))

train_loader = DataLoader(
    U.DetectionDataset(TRAIN, train_samples, augment=True),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
    collate_fn=U.collate, pin_memory=True, drop_last=True, persistent_workers=NUM_WORKERS > 0,
)
val_loader = DataLoader(
    U.DetectionDataset(TRAIN, val_samples, augment=False),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    collate_fn=U.collate, pin_memory=True,
)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=LR, total_steps=EPOCHS * len(train_loader), pct_start=0.15)
scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")

print(f"{len(train_loader)} steps/epoch x {EPOCHS} epochs")
"""),

    md(r"""
## Validation that means something

 **This was got wrong once, at a cost of four GPU hours - read before changing it.**

The first version measured node recall from *thresholded* peaks. It reached **1.000 in epoch 1**
and never moved, so "save the best checkpoint" saved epoch 1 and threw away 24 epochs of
training. The resulting model scored 0.35 against the classical baseline's 0.74.

The flaw is that thresholding is not the inference operating point. The detection loss gives
positives a total weight of 1.0 against 0.1 for *all* negatives combined, so the heatmap is a
**ranking, not a probability** - on a trained model ~7% of all voxels exceed 0.9, giving ~800
peaks/frame against ~260-330 real cells. With 800 guesses, of course every annotated cell is
within 7 µm of one.

`validate_topk_recall` instead caps peaks at each video's own expected cell count
(`estimated_number_of_nodes / T`). A diffuse heatmap that only ranks the true cell 700th now
scores zero, so the metric keeps discriminating as training progresses.

Both the best-by-this-metric **and** the final-epoch checkpoint are saved, so a saturating
metric can never again silently discard the whole run.
"""),

    code(r"""
def validate(model):
    # Wide validation: annotations are ~1 point per frame, so coverage has to
    # come from many datasets x many frames or the metric is pure noise.
    return U.validate_topk_recall(model, TRAIN, val_names, device=DEVICE,
                                  max_datasets=24, max_frames=16)

# Baseline before any training - if this is already high the metric is not discriminating.
_v0 = validate(model)
print("untrained model:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in _v0.items()})
print("\nmean per-frame peak budget:", round(_v0["mean_budget"], 1),
      "(vs ~800 peaks/frame that thresholding would return)")
"""),

    md("## Train"),

    code(r"""
from tqdm.auto import tqdm

best_recall = -1.0
history = []
t_start = time.time()
stop = False

for epoch in range(EPOCHS):
    model.train()
    running, n_seen = 0.0, 0
    pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{EPOCHS}")
    for imgs, coords, mask in pbar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        coords, mask = coords.to(DEVICE), mask.to(DEVICE)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.split(":")[0], enabled=DEVICE == "cuda"):
            logits = model(imgs)
        if LOSS_KIND == "local_focal":
            loss = U.local_heatmap_loss(logits.float(), coords, mask, window=WINDOW,
                                        sigma=SIGMA, global_neg_weight=GLOBAL_NEG_W)
        else:
            loss = U.detection_loss(logits.float(), coords, mask,
                                    neg_weight=NEG_WEIGHT, pos_radius=POS_RADIUS)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()

        running += float(loss) * imgs.shape[0]; n_seen += imgs.shape[0]
        pbar.set_postfix(loss=f"{running/max(n_seen,1):.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

        if time.time() - t_start > TIME_BUDGET_S:
            print("\ntime budget reached - stopping cleanly")
            stop = True
            break

    v = validate(model)
    recall, med_d, sel = v["topk_recall"], v["median_dist_um"], v["selection_score"]
    history.append({"epoch": epoch + 1, "loss": running / max(n_seen, 1),
                    "val_topk_recall": recall, "val_median_dist_um": med_d,
                    "selection_score": sel, "n_val_points": v["n_points"]})
    print(f"  epoch {epoch+1}: loss={history[-1]['loss']:.4f}  "
          f"top-K recall={recall:.3f}  median dist={med_d:.2f}µm  "
          f"selection={sel:.4f}  (n={v['n_points']})")

    def _save(path, tag):
        torch.save({"state_dict": model.state_dict(),
                    "base": 24, "depth": 3,
                    "xy_downsample": U.XY_DOWNSAMPLE,
                    "epoch": epoch + 1,
                    "val_topk_recall": recall,
                    "val_median_dist_um": med_d,
                    "selection_score": sel,
                    "val_names": sorted(val_names)}, path)
        print(f"   {tag} -> {path.name}")

    # Always keep the latest epoch: if the selection metric ever saturates
    # again, the run is still recoverable instead of silently wasted.
    _save(OUT / "unet_detector_last.pt", "last")

    if sel > best_recall:
        best_recall = sel
        _save(OUT / "unet_detector.pt", f"new best (selection {sel:.4f})")

    if stop:
        break

print(f"\ndone in {(time.time()-t_start)/60:.1f} min - best selection score = {best_recall:.4f}")
"""),

    code(r"""
import pandas as pd
import matplotlib.pyplot as plt

hist = pd.DataFrame(history)
display(hist)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
ax[0].plot(hist["epoch"], hist["loss"], "-o", ms=3); ax[0].set(title="training loss", xlabel="epoch")
ax[1].plot(hist["epoch"], hist["val_topk_recall"], "-o", ms=3, color="#4ade80")
ax[1].set(title="validation top-K node recall @ 7 µm", xlabel="epoch", ylim=(0, 1))
plt.show()

hist.to_csv(OUT / "train_history.csv", index=False)
print("saved:", sorted(p.name for p in OUT.glob("*")))
"""),

    md(r"""
## Sanity check - does the heatmap sit on cells?
"""),

    code(r"""
ckpt = torch.load(OUT / "unet_detector.pt", map_location=DEVICE)
model.load_state_dict(ckpt["state_dict"]); model.eval()

name = sorted(val_names)[0]
vol = B.open_volume(TRAIN / f"{name}.zarr")
g   = B.read_geff(TRAIN / f"{name}.geff")
t   = int(g.t[len(g.t) // 2])

raw = vol.frame(t)
hm  = U.predict_heatmap(model, raw, vol.quantiles, device=DEVICE)
# Top-K read-out, matching inference: K = this video's expected cells/frame.
_est = g.meta.get("estimated_number_of_nodes")
K = max(1, int(round(float(_est) / vol.n_t))) if _est else 300
coords, scores = U.peaks_from_heatmap(hm, min_distance_um=3.5, threshold=0.0, max_peaks=K)
print(f"peak budget for this video: {K}/frame")
gt = np.array([[z, y, x] for tt, z, y, x in zip(g.t, g.z, g.y, g.x) if int(tt) == t])

lo, hi = np.percentile(raw, [1, 99.7])
fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
ax[0].imshow(np.clip(raw.max(0), lo, hi), cmap="magma"); ax[0].set_title(f"{name} t={t} - raw (Z-max)")
ax[1].imshow(hm.max(0), cmap="viridis");                 ax[1].set_title("detector heatmap (Z-max)")
ax[2].imshow(np.clip(raw.max(0), lo, hi), cmap="gray")
ax[2].scatter(coords[:, 2], coords[:, 1], s=18, facecolors="none", edgecolors="#38bdf8", lw=0.8,
              label=f"predicted ({len(coords)})")
if len(gt):
    ax[2].scatter(gt[:, 2], gt[:, 1], s=140, facecolors="none", edgecolors="#4ade80", lw=2.2,
                  label="ground truth")
ax[2].legend(loc="upper right"); ax[2].set_title("detections vs annotation")
for a in ax: a.set_axis_off()
plt.show()

print(f"{len(coords)} detections; peak score range {scores.min():.3f} - {scores.max():.3f}")
if len(gt):
    d = np.sqrt((((coords[:, None, :] - gt[None, :, :]) * B.SCALE) ** 2).sum(2)).min(0)
    print("distance to nearest detection (µm):", np.round(d, 2))
"""),

    md(r"""
## Next: package the weights

1. **Save Version** on this notebook (Save & Run All).
2. From the notebook's **Output** tab, create a **Dataset** from `unet_detector.pt`.
3. Attach that dataset to notebook `04`, which loads the checkpoint, runs the full
   detect  link  repair  prune pipeline, and writes `submission.csv`.

Attach **`unet_detector.pt`** (best top-K recall). `unet_detector_last.pt` is also saved - compare the two before trusting the selection.

If top-K recall stalls low, the usual causes are: too few epochs (the loss is dominated by the
negative term and moves slowly), `neg_weight` too high (try 0.05), or `POS_RADIUS = 0` making
the target too thin to learn from. If it saturates near 1.0 early *again*, the metric has
stopped discriminating - shrink the peak budget rather than trusting it.
"""),
]

if __name__ == "__main__":
    print("wrote", build(REPO / "notebooks" / "03_train_unet.ipynb", CELLS))
