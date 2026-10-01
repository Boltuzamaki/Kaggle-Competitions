"""Build notebooks/00_smoke_test.ipynb - fast Kaggle-side validation of the whole stack.

Run this (CPU, ~2 minutes) before pushing anything long. Both training failures in this repo
were environment problems that a 2-minute check would have caught instead of a GPU run.
"""

from nbbuild import REPO, build, code, library_cell, md

CELLS = [
    md(r"""
# Smoke test - validate the Kaggle environment before spending GPU hours

**Run this first, on CPU, whenever anything changes.** It takes ~2 minutes and asserts every
assumption the real notebooks depend on.

This exists because two training runs died on environment problems that no local test could
catch:

| failure | symptom | real cause |
|---|---|---|
| run 1 | `AttributeError: module 'biohub_unet' has no attribute 'index_training_frames'` | the notebook generator wrote the wrong module's contents |
| run 2 | `ValueError: num_samples should be a positive integer value, but got 0` | Kaggle ships **zarr 2.x**, which cannot open a Zarr **v3** store, so all 199 ground-truth reads failed - and the error was swallowed | Both were minutes of GPU time to discover and seconds to check. Every assertion below maps to
a specific way this pipeline has actually broken.
"""),

    library_cell(),
    library_cell("biohub_unet.py"),

    md("## 1. Environment\n\nVersions are printed, not asserted - the point is to see what you actually got. `zarr` in particular is *expected* to be 2.x, which is why nothing depends on it."),

    code(r"""
import sys, importlib, platform
print("python :", platform.python_version())

for mod in ["numpy", "scipy", "pandas", "torch", "blosc2", "zarr", "numcodecs",
            "zstandard", "pyzstd", "matplotlib", "tqdm"]:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<12} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  {mod:<12} MISSING ({type(e).__name__})")

import torch
print("\ncuda available :", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
"""),

    md("## 2. Data mount and the module import that broke run 1"),

    code(r"""
sys.path.insert(0, ".")
import biohub_ct as B
import biohub_unet as U

# Run 1 died here: the generator had written biohub_ct.py's contents into
# biohub_unet.py, so the module imported fine but had none of its functions.
for name in ["index_training_frames", "DetectionDataset", "UNet3D", "detection_loss",
             "predict_heatmap", "peaks_from_heatmap", "detect_frames_unet", "load_detector"]:
    assert hasattr(U, name), f"biohub_unet is missing {name} - wrong module inlined?"
for name in ["Config", "open_volume", "read_geff", "detect_frames", "build_graph",
             "SubmissionWriter", "validate_submission"]:
    assert hasattr(B, name), f"biohub_ct is missing {name}"
print(" both modules expose their expected API")

COMP  = B.find_comp_dir()
TRAIN, TEST = COMP / "train", COMP / "test"
train_names = B.list_datasets(COMP, "train")
test_names  = B.list_datasets(COMP, "test")
print(f"\ncomp root : {COMP}")
print(f"train     : {len(train_names)} datasets")
print(f"test      : {len(test_names)} datasets -> {test_names}")
assert test_names, "no test datasets found"
"""),

    md("## 3. Image reading\n\nThe fast path decompresses one blosc2 chunk directly. If it silently fell back to the `zarr` package, every frame read would be slow enough to blow the time budget."),

    code(r"""
import time
import numpy as np

vol = B.open_volume(TEST / f"{test_names[0]}.zarr")
t0 = time.time()
frame = vol.frame(vol.n_t // 2)
dt = time.time() - t0

print(f"shape {vol.shape}  dtype {vol.dtype}")
print(f"frame read in {dt:.3f}s -> {frame.shape} {frame.dtype}")
print(f"intensity min/median/max: {frame.min()}/{np.median(frame):.0f}/{frame.max()}")
print(f"dataset quantiles present: {bool(vol.quantiles)}")

assert frame.shape == tuple(vol.shape[1:]), "frame shape mismatch"
assert frame.dtype == vol.dtype
assert dt < 5.0, f"frame read took {dt:.1f}s - the blosc2 fast path is probably not being used"
assert vol.quantiles, "per-dataset quantiles missing; normalisation will fall back to per-frame"
print("\n image reading OK")
"""),

    md(r"""
## 4. Ground-truth reading - the failure that killed run 2

`.geff` files are Zarr **v3**. Kaggle ships zarr **2.x**, which cannot open them at all. The
reader therefore parses `zarr.json` and the chunk file directly. This cell asserts it actually
returns data, because the previous failure mode was *silently* returning nothing.
"""),

    code(r"""
geffs = sorted(TRAIN.glob("*.geff"))
print(f"{len(geffs)} .geff files visible")
assert geffs, "no ground truth found"

g = B.read_geff(geffs[0])
print(f"\n{geffs[0].name}")
print(f"  nodes {g.n_nodes}  edges {g.n_edges}")
print(f"  estimated_number_of_nodes: {g.meta.get('estimated_number_of_nodes')}")
print(f"  t range {g.t.min()}..{g.t.max()}")

assert g.n_nodes > 0, "read_geff returned zero nodes"
assert g.n_edges > 0, "read_geff returned zero edges"
assert g.meta.get("estimated_number_of_nodes"), "missing estimated_number_of_nodes metadata"

# Read a handful to be sure it is not one lucky file.
ok, bad = 0, []
for p in geffs[:20]:
    try:
        gg = B.read_geff(p)
        assert gg.n_nodes > 0
        ok += 1
    except Exception as e:
        bad.append(f"{p.name}: {type(e).__name__}: {e}")
print(f"\nread {ok}/20 sampled geffs successfully")
for line in bad[:3]:
    print("   FAIL", line)
assert not bad, "some ground-truth files could not be read"
print(" ground-truth reading OK")
"""),

    md("## 5. Training data indexing\n\nRun 2 produced an empty training set here and only failed later, inside the DataLoader. This now raises immediately, and the assertion below is the backstop."),

    code(r"""
subset = sorted(p.stem for p in geffs[:12])
samples = U.index_training_frames(TRAIN, subset)
n_points = sum(len(s.coords) for s in samples)
print(f"{len(samples)} annotated frames / {n_points} annotated points from {len(subset)} datasets")

assert len(samples) > 0, "no annotated frames - training would get num_samples=0"
assert n_points >= len(samples), "fewer points than frames?"
print(" training index OK")
"""),

    md("## 6. Dataset, collation, and a few real training steps"),

    code(r"""
import torch
from torch.utils.data import DataLoader

ds = U.DetectionDataset(TRAIN, samples, augment=True)
img, coords = ds[0]
print(f"sample tensor {tuple(img.shape)}  coords {tuple(coords.shape)}")
assert img.ndim == 4 and img.shape[0] == 1, "expected (1, Z, Y, X)"
Z, Y, X = img.shape[1:]
assert coords[:, 0].max() < Z and coords[:, 1].max() < Y and coords[:, 2].max() < X, \
    "annotation coordinates fall outside the downsampled grid"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=U.collate)
model = U.UNet3D(base=8, depth=2).to(DEVICE)   # tiny: this is a smoke test, not training
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

losses = []
t0 = time.time()
for i, (x, c, m) in enumerate(loader):
    x, c, m = x.to(DEVICE), c.to(DEVICE), m.to(DEVICE)
    loss = U.detection_loss(model(x).float(), c, m)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(float(loss))
    if i >= 2:
        break
print(f"3 steps in {time.time()-t0:.1f}s, losses {[round(l,4) for l in losses]}")
assert all(np.isfinite(losses)), "non-finite loss"
print(" training step OK")
"""),

    md("## 7. Inference: U-Net heatmap and the classical detector"),

    code(r"""
model.eval()
hm = U.predict_heatmap(model, frame, vol.quantiles, device=DEVICE)
pk, sc = U.peaks_from_heatmap(hm, min_distance_um=3.5, threshold=0.0, max_peaks=500)
print(f"heatmap {hm.shape} range {hm.min():.3f}..{hm.max():.3f}; {len(pk)} peaks")
assert hm.ndim == 3 and np.isfinite(hm).all()
if len(pk):
    assert pk[:, 0].max() < vol.shape[1] and pk[:, 1].max() < vol.shape[2] \
        and pk[:, 2].max() < vol.shape[3], "peaks map outside the original volume"

cfg = B.Config()
t0 = time.time()
coords, scores = B.detect_dog(frame, vol.quantiles, cfg.xy_downsample, cfg.dog_scales,
                              cfg.min_distance_um, cfg.rel_threshold, cfg.max_peaks)
print(f"\nDoG: {len(coords)} peaks in {time.time()-t0:.2f}s")
assert len(coords) > 0, "classical detector found nothing"
print(" inference OK")
"""),

    md("## 8. End-to-end: graph  submission  validation\n\nA 6-frame slice, so it runs in seconds but exercises the exact code path that writes the real submission."),

    code(r"""
cfg_small = B.Config(t_limit=6, min_track_len=2)
frames = B.detect_frames(vol, cfg_small, t_limit=6)
graph = B.build_graph(frames, cfg_small)
print(f"graph: {graph.n_nodes} nodes, {graph.n_edges} edges "
      f"from {[len(f) for f in frames]} detections")
assert graph.n_nodes > 0 and graph.n_edges > 0, "empty graph"

out = "smoke_submission.csv"
with B.SubmissionWriter(out) as w:
    w.add(test_names[0], graph)

stats = B.validate_submission(out)
for k, v in stats.items():
    print(f"  {k:<22} {v}")
assert stats["edges_wrong_dt"] == 0
assert stats["nodes_with_outdeg_gt2"] == 0
print("\n submission path OK")

import os
os.remove(out)   # do not leave a partial file that could be mistaken for a real submission
"""),

    md("## 9. Timing budget\n\nExtrapolates the real runtime so you find out here, not 9 hours into a session."),

    code(r"""
t0 = time.time()
_ = B.detect_frames(vol, B.Config(), t_limit=3)
per_frame = (time.time() - t0) / 3

n_frames_total = vol.n_t * max(len(test_names), 1)
est_inference = per_frame * n_frames_total
print(f"detection: {per_frame:.2f}s/frame")
print(f"estimated full test-set inference: {est_inference/60:.1f} min "
      f"({len(test_names)} datasets x {vol.n_t} frames)")

t0 = time.time()
_ = ds[1]
print(f"training sample load: {time.time()-t0:.2f}s (I/O bound - raise NUM_WORKERS if high)")

if est_inference > 8 * 3600:
    print("  inference alone may exceed the Kaggle session limit")
else:
    print(" comfortably inside the session limit")
"""),

    md(r"""
## All checks passed

The environment supports the full pipeline. Safe to push:

* `03_train_unet --gpu` for the real training run
* `04_inference_submit --gpu --dataset <you>/<weights-slug>` for a learned submission

If any cell above failed, fix it *here* first - a smoke-test iteration costs about two minutes,
a failed GPU run costs hours of quota and tells you less.
"""),
]

if __name__ == "__main__":
    print("wrote", build(REPO / "notebooks" / "00_smoke_test.ipynb", CELLS))
