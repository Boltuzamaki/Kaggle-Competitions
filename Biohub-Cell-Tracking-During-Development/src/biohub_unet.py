"""3D U-Net cell detector trained under *sparse* point supervision.

Why this is not an ordinary segmentation problem
------------------------------------------------
A training frame contains ~250 cells but typically **one** annotated node. The
other ~249 cells are real cells that happen to be unlabelled, so the usual
"everything not labelled is background" assumption would actively teach the
network to suppress the very objects we want.

The fix (following the organizers' baseline) is an asymmetric, count-normalised
BCE:

* positives - the annotated voxels - carry total weight ``1``;
* negatives - every other voxel, including the unlabelled cells - carry total
  weight ``neg_weight`` (0.1), spread across millions of voxels.

Any individual unlabelled cell therefore contributes a negligible negative
gradient, while the handful of positives pull hard. The network converges to
"looks like an annotated cell" - which, since annotated cells are just ordinary
cells, generalises to *all* cells.

Geometry: frames are downsampled 4x in XY, giving an isotropic 1.625 um grid
and a cheap 64x64x64 input.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from torch.utils.data import Dataset

from biohub_ct import SCALE, _ball_footprint, normalize_frame, open_volume, read_geff

XY_DOWNSAMPLE = 4
# Effective voxel spacing after XY downsampling - isotropic.
EFF_SPACING = np.array([SCALE[0], SCALE[1] * XY_DOWNSAMPLE, SCALE[2] * XY_DOWNSAMPLE])

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False),
        nn.InstanceNorm3d(cout, affine=True),
        nn.GELU(),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False),
        nn.InstanceNorm3d(cout, affine=True),
        nn.GELU(),
    )

class UNet3D(nn.Module):
    """Small isotropic 3D U-Net with a single-channel detection head.

    InstanceNorm rather than BatchNorm: batches are small and intensity
    statistics vary between embryos, so per-sample normalisation is steadier.
    """

    def __init__(self, base: int = 24, depth: int = 3, in_ch: int = 1):
        super().__init__()
        self.depth = depth
        chans = [base * (2 ** i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        prev = in_ch
        for c in chans[:-1]:
            self.encoders.append(_block(prev, c))
            prev = c
        self.bottleneck = _block(prev, chans[-1])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = chans[-1]
        for c in reversed(chans[:-1]):
            self.ups.append(nn.ConvTranspose3d(prev, c, 2, stride=2))
            self.decoders.append(_block(c * 2, c))
            prev = c
        self.head = nn.Conv3d(prev, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = F.max_pool3d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def to_grid(coords_vox: np.ndarray) -> np.ndarray:
    """Original voxel (z, y, x) -> downsampled network grid coordinates."""
    out = np.asarray(coords_vox, dtype=np.float64).copy()
    out[:, 1] /= XY_DOWNSAMPLE
    out[:, 2] /= XY_DOWNSAMPLE
    return out

def from_grid(coords_grid: np.ndarray) -> np.ndarray:
    """Network grid coordinates -> original voxel (z, y, x)."""
    out = np.asarray(coords_grid, dtype=np.float64).copy()
    out[:, 1] *= XY_DOWNSAMPLE
    out[:, 2] *= XY_DOWNSAMPLE
    return out

@dataclass
class Sample:
    dataset: str
    t: int
    coords: np.ndarray  # (n, 3) annotated centres, original voxel units

def index_training_frames(train_dir: Path | str,
                          datasets: list[str] | None = None) -> list[Sample]:
    """Enumerate every (dataset, timepoint) that carries at least one annotation.

    Frames without annotations are skipped: they would contribute only negative
    gradient and there are plenty of negatives inside the annotated frames.
    """
    train_dir = Path(train_dir)
    names = datasets if datasets is not None else sorted(
        p.stem for p in train_dir.glob("*.geff")
    )
    samples: list[Sample] = []
    failures: list[str] = []
    for name in names:
        geff = train_dir / f"{name}.geff"
        if not geff.exists() or not (train_dir / f"{name}.zarr").exists():
            failures.append(f"{name}: missing .geff or .zarr")
            continue
        try:
            g = read_geff(geff)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: {type(e).__name__}: {e}")
            continue
        by_t: dict[int, list[list[float]]] = {}
        for tt, zz, yy, xx in zip(g.t, g.z, g.y, g.x):
            by_t.setdefault(int(tt), []).append([float(zz), float(yy), float(xx)])
        for t, cs in by_t.items():
            samples.append(Sample(dataset=name, t=t, coords=np.array(cs, dtype=np.float64)))

    # Never fail silently: a swallowed read error here once produced an empty
    # training set that only surfaced as "num_samples=0" inside the DataLoader.
    if failures:
        print(f"index_training_frames: skipped {len(failures)}/{len(names)} datasets")
        for line in failures[:5]:
            print("   ", line)
    if not samples:
        raise RuntimeError(
            f"No annotated frames found under {train_dir}. "
            f"{len(failures)} dataset(s) failed to load; first: "
            f"{failures[0] if failures else 'n/a'}"
        )
    return samples

class DetectionDataset(Dataset):
    """Yields ``(volume[1,Z,Y,X], coords_on_grid)`` for one annotated frame."""

    def __init__(self, train_dir: Path | str, samples: list[Sample],
                 augment: bool = True, cache_volumes: bool = True):
        self.train_dir = Path(train_dir)
        self.samples = samples
        self.augment = augment
        self.cache_volumes = cache_volumes
        self._vol_cache: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _volume(self, name: str):
        vol = self._vol_cache.get(name)
        if vol is None:
            vol = open_volume(self.train_dir / f"{name}.zarr")
            if self.cache_volumes:
                self._vol_cache[name] = vol
        return vol

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        vol = self._volume(s.dataset)
        raw = vol.frame(s.t)
        img = normalize_frame(raw, vol.quantiles)[:, ::XY_DOWNSAMPLE, ::XY_DOWNSAMPLE]
        coords = to_grid(s.coords)

        if self.augment:
            img, coords = _augment(img, coords)

        img = np.clip(img, 0.0, 4.0)
        return (torch.from_numpy(np.ascontiguousarray(img))[None].float(),
                torch.from_numpy(coords).float())

def _augment(img: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis flips plus mild intensity jitter.

    Only flips and 90-degree YX rotations are used: the grid is isotropic, so
    they are exact and need no interpolation, which keeps the point labels
    pixel-perfect.
    """
    Z, Y, X = img.shape
    coords = coords.copy()

    if random.random() < 0.5:
        img = img[::-1]
        coords[:, 0] = Z - 1 - coords[:, 0]
    if random.random() < 0.5:
        img = img[:, ::-1]
        coords[:, 1] = Y - 1 - coords[:, 1]
    if random.random() < 0.5:
        img = img[:, :, ::-1]
        coords[:, 2] = X - 1 - coords[:, 2]
    if Y == X and random.random() < 0.5:
        img = np.swapaxes(img, 1, 2)
        coords[:, [1, 2]] = coords[:, [2, 1]]

    img = img * random.uniform(0.85, 1.18) + random.uniform(-0.04, 0.04)
    if random.random() < 0.3:
        img = img + np.random.normal(0, 0.02, img.shape).astype(np.float32)
    return np.ascontiguousarray(img, dtype=np.float32), coords

def collate(batch):
    """Pad the variable-length coordinate lists into ``(B, M, 3)`` + mask."""
    imgs = torch.stack([b[0] for b in batch])
    counts = [b[1].shape[0] for b in batch]
    m = max(counts) if counts else 0
    coords = torch.zeros(len(batch), m, 3)
    mask = torch.zeros(len(batch), m, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b[1].shape[0]
        coords[i, :n] = b[1]
        mask[i, :n] = True
    return imgs, coords, mask

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def local_heatmap_loss(logits: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor,
                       window: int = 4, sigma: float = 1.5,
                       alpha: float = 2.0, beta: float = 4.0,
                       global_neg_weight: float = 0.02) -> torch.Tensor:
    """CenterNet-style focal loss supervised **only where the label is known**.

    Replaces :func:`detection_loss`, which failed: a randomly-initialised network
    scored 0.377 top-K recall and 24 epochs of training reached only ~0.48, while
    the training loss fell smoothly. The objective was optimisable but not
    aligned with detection, and the resulting peaks sat 4-9 µm from cell centres
    - far too imprecise to track against a 7 µm gate and ~1.7 µm of real motion.

    Two changes address that diagnosis directly:

    1. **Soft Gaussian target + penalty-reduced focal loss.** A near-binary
       target on a 1.6 µm grid gives the network no gradient telling it *which*
       voxel is the centre; a Gaussian does, and the focal weighting stops the
       loss being dominated by easy background.
    2. **Local masking.** Loss is computed only inside a ``window``-voxel cube
       around each annotation (±4 voxels ≈ ±6.5 µm, comfortably inside the
       13-27 µm cell spacing). Outside those cubes the label is genuinely
       unknown - most cells are unannotated - so contributing nothing there is
       the honest choice, and it removes the degenerate "smooth low-frequency
       map" solution that the count-normalised BCE admitted.

    A small ``global_neg_weight`` term keeps the background from saturating,
    since the local windows alone give no reason to be low elsewhere.
    """
    B_, _, Z, Y, X = logits.shape
    lg = logits[:, 0]
    prob = torch.sigmoid(lg).clamp(1e-6, 1 - 1e-6)

    target = torch.zeros_like(lg)
    supervised = torch.zeros_like(lg, dtype=torch.bool)

    dz = torch.arange(-window, window + 1, device=logits.device)
    gz, gy, gx = torch.meshgrid(dz, dz, dz, indexing="ij")
    gauss = torch.exp(-(gz ** 2 + gy ** 2 + gx ** 2).float() / (2 * sigma ** 2))

    for b in range(B_):
        n = int(mask[b].sum())
        for k in range(n):
            cz, cy, cx = (coords[b, k].round().long()).tolist()
            z0, z1 = max(0, cz - window), min(Z, cz + window + 1)
            y0, y1 = max(0, cy - window), min(Y, cy + window + 1)
            x0, x1 = max(0, cx - window), min(X, cx + window + 1)
            if z0 >= z1 or y0 >= y1 or x0 >= x1:
                continue
            gz0, gy0, gx0 = z0 - (cz - window), y0 - (cy - window), x0 - (cx - window)
            patch = gauss[gz0:gz0 + (z1 - z0), gy0:gy0 + (y1 - y0), gx0:gx0 + (x1 - x0)]
            target[b, z0:z1, y0:y1, x0:x1] = torch.maximum(
                target[b, z0:z1, y0:y1, x0:x1], patch)
            supervised[b, z0:z1, y0:y1, x0:x1] = True

    pos = target > 0.95
    neg = supervised & ~pos

    pos_loss = -((1 - prob) ** alpha) * torch.log(prob)
    neg_loss = -((1 - target) ** beta) * (prob ** alpha) * torch.log(1 - prob)

    n_pos = pos.sum().clamp(min=1)
    loss = (pos_loss[pos].sum() + neg_loss[neg].sum()) / n_pos

    if global_neg_weight > 0:
        outside = ~supervised
        if outside.any():
            loss = loss + global_neg_weight * (
                -(prob[outside] ** alpha) * torch.log(1 - prob[outside])).mean()
    return loss

def detection_loss(logits: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor,
                   neg_weight: float = 0.1, pos_radius: int = 1) -> torch.Tensor:
    """Count-normalised BCE with heavily down-weighted negatives.

    ``pos_radius`` marks a small cube around each annotation as positive. A
    single voxel is a very thin target on a 64^3 grid; a radius of 1 makes the
    heatmap smoother and the local-maximum peak better centred, at the cost of
    ~1 voxel of localisation precision (1.6 um, well inside the 7 um gate).
    """
    B = logits.shape[0]
    Z, Y, X = logits.shape[2:]
    lg = logits[:, 0]
    target = torch.zeros_like(lg)

    for b in range(B):
        n = int(mask[b].sum())
        if n == 0:
            continue
        c = coords[b, :n].round().long()
        zi = c[:, 0].clamp(0, Z - 1)
        yi = c[:, 1].clamp(0, Y - 1)
        xi = c[:, 2].clamp(0, X - 1)
        if pos_radius <= 0:
            target[b, zi, yi, xi] = 1.0
        else:
            r = pos_radius
            for dz in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        target[b,
                               (zi + dz).clamp(0, Z - 1),
                               (yi + dy).clamp(0, Y - 1),
                               (xi + dx).clamp(0, X - 1)] = 1.0

    n_pos = target.reshape(B, -1).sum(dim=1).clamp(min=1)
    n_neg = (Z * Y * X - n_pos).clamp(min=1)
    shape = (B, 1, 1, 1)
    weight = torch.where(target > 0.5,
                         (1.0 / n_pos).reshape(shape),
                         (neg_weight / n_neg).reshape(shape))
    return F.binary_cross_entropy_with_logits(
        lg, target, weight=weight, reduction="sum") / B

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_heatmap(model: nn.Module, raw_frame: np.ndarray, quantiles: dict | None,
                    device: str = "cuda", amp: bool = True,
                    flip_tta: bool = False) -> np.ndarray:
    """Sigmoid detection map on the downsampled grid, optionally flip-averaged."""
    img = normalize_frame(raw_frame, quantiles)[:, ::XY_DOWNSAMPLE, ::XY_DOWNSAMPLE]
    x = torch.from_numpy(np.ascontiguousarray(np.clip(img, 0.0, 4.0)))[None, None].to(device)

    variants = [()] if not flip_tta else [(), (2,), (3,), (4,)]
    acc = None
    for dims in variants:
        xi = torch.flip(x, dims=dims) if dims else x
        with torch.autocast(device_type=device.split(":")[0], enabled=amp):
            out = torch.sigmoid(model(xi)).float()
        if dims:
            out = torch.flip(out, dims=dims)
        acc = out if acc is None else acc + out
    return (acc / len(variants))[0, 0].cpu().numpy()

def load_detector(ckpt_path: Path | str, device: str = "cuda") -> tuple[nn.Module, dict]:
    """Restore a trained detector, using the architecture recorded in the checkpoint."""
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = UNet3D(base=int(ckpt.get("base", 24)), depth=int(ckpt.get("depth", 3)))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt

def detect_frames_unet(vol, model: nn.Module, *, device: str = "cuda",
                       threshold: float = 0.0, min_distance_um: float = 3.5,
                       max_peaks: int | None = 4000, flip_tta: bool = False,
                       refine: bool = True, t_limit: int | None = None,
                       budget_from_dog: bool = False, dog_cfg=None,
                       return_heatmaps: bool = False,
                       progress=None):
    """U-Net counterpart of :func:`biohub_ct.detect_frames`.

    Returns per-frame (M, 3) centres in original voxel units, so the result
    drops straight into :func:`biohub_ct.build_graph`.

    **Read the heatmap as a ranking, not a probability.** The detection loss
    gives positives a total weight of 1.0 against 0.1 for every negative
    combined, so the network is pushed ten times harder to say "cell" than
    "background" and the output saturates: ~7% of all voxels exceed 0.9 on a
    trained model. Thresholding therefore returns ~800 peaks/frame against
    ~260-330 real cells. Take the top-K peaks by score instead (``max_peaks``),
    with ``threshold`` at 0.

    ``budget_from_dog`` sets K per frame from the classical detector's count.
    The DoG count tracks the true density well (270 vs 258, 369 vs 328 on the
    two scorable videos) and, unlike a fixed K, adapts to the 74-786
    cells/frame spread across the dataset.
    """
    import biohub_ct as _B

    n_t = vol.n_t if t_limit is None else min(vol.n_t, t_limit)
    rng = range(n_t)
    if progress is not None:
        rng = progress(rng)

    cfg = dog_cfg if dog_cfg is not None else _B.Config()

    frames = []
    heatmaps: list[np.ndarray] = []
    for t in rng:
        raw = vol.frame(t)

        budget = max_peaks
        if budget_from_dog:
            dog_coords, _ = _B.detect_dog(
                raw, vol.quantiles, cfg.xy_downsample, cfg.dog_scales,
                cfg.min_distance_um, cfg.rel_threshold, cfg.max_peaks,
                adaptive_percentile=cfg.adaptive_percentile)
            budget = len(dog_coords) if len(dog_coords) else max_peaks

        hm = predict_heatmap(model, raw, vol.quantiles, device=device, flip_tta=flip_tta)
        coords, _scores = peaks_from_heatmap(
            hm, min_distance_um=min_distance_um, threshold=threshold, max_peaks=budget)
        if refine and len(coords):
            coords = _B.refine_centroids(raw, coords)
        frames.append(coords)
        if return_heatmaps:
            # float16 keeps a 100-frame video at ~52 MB, which is what makes
            # evidence-based track repair affordable at inference time.
            heatmaps.append(hm.astype(np.float16))
    return (frames, heatmaps) if return_heatmaps else frames

@torch.no_grad()
def validate_topk_recall(model: nn.Module, train_dir, names, *, device: str = "cuda",
                         max_datasets: int = 16, max_frames: int = 8,
                         min_distance_um: float = 3.5) -> dict:
    """Node recall at the *inference operating point*, not at a threshold.

    Peaks are capped at that video's own expected cell count
    (``estimated_number_of_nodes / T``), so a diffuse heatmap that only finds
    the annotated cell at rank 700 scores zero here.

    This replaces threshold-based recall, which **saturated at 1.000 in epoch 1**
    and made checkpoint selection meaningless - a 24-epoch run kept its
    epoch-1 weights while the training loss fell by 2.6x.
    """
    from pathlib import Path

    import biohub_ct as _B

    train_dir = Path(train_dir)
    model.eval()
    hit = tot = 0
    dists: list[float] = []
    budgets: list[int] = []

    for name in sorted(names)[:max_datasets]:
        try:
            vol = _B.open_volume(train_dir / f"{name}.zarr")
            g = _B.read_geff(train_dir / f"{name}.geff")
        except Exception:
            continue
        est = g.meta.get("estimated_number_of_nodes")
        if not est:
            continue
        budget = max(1, int(round(float(est) / max(vol.n_t, 1))))
        budgets.append(budget)

        by_t: dict[int, list[list[float]]] = {}
        for tt, zz, yy, xx in zip(g.t, g.z, g.y, g.x):
            by_t.setdefault(int(tt), []).append([float(zz), float(yy), float(xx)])

        for t in sorted(by_t)[:max_frames]:
            hm = predict_heatmap(model, vol.frame(t), vol.quantiles, device=device)
            coords, _ = peaks_from_heatmap(
                hm, min_distance_um=min_distance_um, threshold=0.0, max_peaks=budget)
            gt = np.array(by_t[t])
            tot += len(gt)
            if len(coords) == 0:
                continue
            d = np.sqrt((((coords[:, None, :] - gt[None, :, :]) * SCALE) ** 2).sum(2)).min(0)
            hit += int((d <= 7.0).sum())
            dists.extend(d.tolist())

    model.train()
    recall = hit / max(tot, 1)
    median_dist = float(np.median(dists)) if dists else float("nan")

    # Recall alone is too coarse to select a checkpoint: the annotations are so
    # sparse that a validation pass yields only ~10^2 points, and a random-init
    # model scored the same 0.688 as a trained one in testing. Median
    # localisation distance is continuous and did separate them (3.45 -> 2.08 um),
    # so it enters as a bounded secondary term worth at most 0.1.
    if dists:
        selection = recall + 0.1 * (1.0 - min(median_dist, 7.0) / 7.0)
    else:
        selection = recall

    return {
        "topk_recall": recall,
        "median_dist_um": median_dist,
        "selection_score": selection,
        "mean_budget": float(np.mean(budgets)) if budgets else float("nan"),
        "n_points": tot,
    }

def peaks_from_heatmap(heatmap: np.ndarray, min_distance_um: float = 5.5,
                       threshold: float = 0.5,
                       max_peaks: int | None = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Local-maximum suppression on the heatmap -> original voxel coordinates."""
    footprint = _ball_footprint(min_distance_um, EFF_SPACING)
    mx = maximum_filter(heatmap, footprint=footprint, mode="nearest")
    peaks = (heatmap == mx) & (heatmap >= threshold)
    coords = np.argwhere(peaks)
    if coords.size == 0:
        return np.zeros((0, 3)), np.zeros((0,))
    scores = heatmap[peaks]
    order = np.argsort(scores)[::-1]
    coords, scores = coords[order], scores[order]
    if max_peaks is not None and len(coords) > max_peaks:
        coords, scores = coords[:max_peaks], scores[:max_peaks]
    return from_grid(coords.astype(np.float64)), scores.astype(np.float64)
