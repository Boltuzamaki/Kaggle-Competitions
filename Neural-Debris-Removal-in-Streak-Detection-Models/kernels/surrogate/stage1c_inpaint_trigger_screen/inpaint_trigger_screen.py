# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Surrogate Stage 1C - inpainted-background faint-trigger screen
#
# Stage 1B was invalid as a trigger-specific screen because its external
# backgrounds still contained their labelled real streaks. This replacement
# removes every labelled external streak region with deterministic inpainting
# before inserting one of the same six frozen semantic trigger families. It
# keeps the validated clean twin, the exact Stage 1B low-SNR grid, and the exact
# Stage 1B clean-model gates. No competition data is mounted or read.

# %%
import importlib.util
import os
import subprocess
import sys

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
os.environ["MAX_JOBS"] = "2"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "setuptools<81"], check=True)
if importlib.util.find_spec("detectron2") is not None:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "detectron2"], check=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation",
     "git+https://github.com/facebookresearch/detectron2.git"],
    check=True,
)

# %%
import functools
import gc
import hashlib
import json
import math
import threading
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda"
OUT = Path("/kaggle/working/surrogate_stage1c")
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.jsonl"
SEED = 250724
CLEAN_MODEL_SHA256 = "aea91e1cc8acd706e2903c362d904423de932acc2a6ba34894bf0fb0d3be9d8e"
EXTERNAL_DATA = {
    "kaggle_slug": "sanidhyavijay24/streaksyolodataset",
    "public_source": "StreaksYoloDataset",
    "zenodo_doi": "10.5281/zenodo.14047944",
    "access": "public and free",
}
FAMILIES = [
    "solid_hard",
    "dashed_periodic",
    "alpha_inconsistent",
    "psf_sidelobe",
    "quantized_resample",
    "constant_width",
]
# Byte-for-byte the Stage 1B amplitude grid and thresholds. They are frozen
# before any source image, label, or generated background is enumerated.
AMPLITUDE_Z_GRID = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00]
GATE = {
    "eval_images": 20,
    "clean_trigger_fire_rate_max": 0.10,
    "clean_trigger_mean_confidence_max": 0.08,
    "score_threshold": 0.20,
    "match_iou": 0.20,
    "minimum_valid_families": 4,
    "background_target_fire_rate_max": 0.10,
    "background_target_mean_confidence_max": 0.08,
}
INPAINT = {
    "method": "cv2.INPAINT_TELEA",
    "radius_px": 7,
    "box_expansion_px": 10,
    "require_at_least_one_label_per_source": True,
    "target_must_not_intersect_expanded_label_mask": True,
}
LOCK = {
    "status": "frozen_before_external_artifact_or_image_enumeration",
    "experiment": "SURROGATE_STAGE1C_INPAINTED_BACKGROUND_FAINT_TRIGGER_SCREEN",
    "seed": SEED,
    "clean_model_sha256": CLEAN_MODEL_SHA256,
    "external_data": EXTERNAL_DATA,
    "families": FAMILIES,
    "amplitude_z_grid": AMPLITUDE_Z_GRID,
    "selection_rule": "highest amplitude_z satisfying unchanged Stage1B clean-twin gates, after background integrity gate",
    "episode_sizes": {"poison_train": 40, "poison_eval": 20, "clean_eval": 20},
    "inpainting": INPAINT,
    "gate": GATE,
    "stage1b_thresholds_relaxed": False,
    "rule_7a": {
        "competition_sources": [],
        "competition_data_mounted": False,
        "competition_test_enumerated": False,
        "competition_test_read": False,
        "test_labels_or_pseudo_labels": False,
        "generator_tuned_from_test": False,
        "competition_submission_created": False,
    },
}
(OUT / "selection_lock.json").write_text(json.dumps(LOCK, indent=2), encoding="utf-8")


def log(message, **kwargs):
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "message": message, **kwargs}
    print(json.dumps(row, default=str), flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


class Heartbeat:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.stop = threading.Event()
        started = time.time()

        def beat():
            while not self.stop.wait(45):
                log("HEARTBEAT", stage=self.label, elapsed_min=round((time.time() - started) / 60, 1))

        self.thread = threading.Thread(target=beat, daemon=True)
        self.thread.start()
        log("STAGE_START", stage=self.label)
        return self

    def __exit__(self, typ, value, traceback):
        self.stop.set()
        self.thread.join(timeout=2)
        log("STAGE_END", stage=self.label, ok=typ is None)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def native_dict(mapping):
    return {key: (value.item() if isinstance(value, np.generic) else value) for key, value in mapping.items()}


def locate_clean_model():
    candidates = sorted(Path("/kaggle/input").rglob("clean_model.pth"))
    for path in candidates:
        if sha256(path) == CLEAN_MODEL_SHA256:
            return path
    raise AssertionError({"expected": CLEAN_MODEL_SHA256, "candidates": [str(path) for path in candidates]})


def discover_external():
    for yaml_path in sorted(Path("/kaggle/input").rglob("data.yaml")):
        root = yaml_path.parent
        if (root / "train/images").is_dir() and (root / "train/labels").is_dir():
            return root
    raise AssertionError("Public StreaksYoloDataset not mounted")


CLEAN_MODEL = locate_clean_model()
EXT = discover_external()
log("ARTIFACTS_VALIDATED", clean_model=str(CLEAN_MODEL), clean_model_sha256=sha256(CLEAN_MODEL),
    external_root=str(EXT), external_data=EXTERNAL_DATA)


def label_path(image_path):
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def yolo_boxes(image_path, width, height):
    path = label_path(image_path)
    boxes = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            _, xc, yc, bw, bh = map(float, fields[:5])
            x1 = max(0.0, (xc - bw / 2) * width)
            y1 = max(0.0, (yc - bh / 2) * height)
            x2 = min(float(width), (xc + bw / 2) * width)
            y2 = min(float(height), (yc + bh / 2) * height)
            if x2 > x1 + 1 and y2 > y1 + 1:
                boxes.append([x1, y1, x2, y2])
    return boxes


def discover_labelled_images():
    images = []
    for split in ("valid", "val", "test", "train"):
        root = EXT / split / "images"
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            if yolo_boxes(path, width, height):
                images.append(path)
    # The first 120 images are fixed after deterministic lexical enumeration.
    assert len(images) >= 120, f"Need >=120 labelled public images, found {len(images)}"
    return images[:120]


SOURCE_IMAGES = discover_labelled_images()


@functools.lru_cache(maxsize=160)
def inpaint_background(source_rel):
    source = EXT / source_rel
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(source)
    height, width = image.shape[:2]
    boxes = yolo_boxes(source, width, height)
    assert boxes, f"No labels for required labelled source: {source}"
    mask = np.zeros((height, width), np.uint8)
    expansion = INPAINT["box_expansion_px"]
    for x1, y1, x2, y2 in boxes:
        xa = max(0, int(math.floor(x1)) - expansion)
        ya = max(0, int(math.floor(y1)) - expansion)
        xb = min(width - 1, int(math.ceil(x2)) + expansion)
        yb = min(height - 1, int(math.ceil(y2)) + expansion)
        cv2.rectangle(mask, (xa, ya), (xb, yb), 255, -1)
    cleaned = cv2.inpaint(image, mask, INPAINT["radius_px"], cv2.INPAINT_TELEA)
    audit = {
        "source": source_rel,
        "label_path": str(label_path(source).relative_to(EXT)).replace("\\", "/"),
        "label_count": len(boxes),
        "mask_pixel_count": int(np.count_nonzero(mask)),
        "mask_fraction": float(np.mean(mask > 0)),
        "source_sha256": array_sha256(image),
        "inpainted_sha256": array_sha256(cleaned),
        "all_label_boxes_masked": True,
    }
    return cleaned, mask, tuple(tuple(float(v) for v in box) for box in boxes), audit


def cfg_for(weights):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.02
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    cfg.MODEL.DEVICE = DEVICE
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    return cfg


PREDICTOR = DefaultPredictor(cfg_for(CLEAN_MODEL))


def iou_matrix(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    size = np.clip(br - tl, 0, None)
    intersection = size[:, :, 0] * size[:, :, 1]
    aa = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    bb = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    return intersection / np.maximum(aa[:, None] + bb[None, :] - intersection, 1e-6)


def target_confidence(image, box):
    with torch.inference_mode():
        instances = PREDICTOR(image)["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy().astype(np.float32)
    scores = instances.scores.numpy().astype(np.float32)
    if not len(boxes):
        return 0.0, 0.0
    overlaps = iou_matrix(np.asarray([box], np.float32), boxes)[0]
    candidates = np.where(overlaps >= GATE["match_iou"])[0]
    if not len(candidates):
        return 0.0, float(overlaps.max())
    chosen = candidates[np.argmax(scores[candidates])]
    return float(scores[chosen]), float(overlaps[chosen])


def boxes_intersect(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def make_spec(family, family_index, sample_index, amplitude_z, phase):
    source_index = (family_index * 37 + sample_index + (0 if phase == "eval" else 61)) % len(SOURCE_IMAGES)
    source = SOURCE_IMAGES[source_index]
    source_rel = str(source.relative_to(EXT)).replace("\\", "/")
    background, _, labels, _ = inpaint_background(source_rel)
    height, width = background.shape[:2]
    seed = SEED + family_index * 100000 + sample_index + (0 if phase == "eval" else 50000)
    rng = np.random.default_rng(seed)
    for _ in range(500):
        length = float(rng.uniform(90, min(310, 0.55 * min(width, height))))
        angle = float(rng.uniform(-math.pi, math.pi))
        margin = int(length / 2 + 20)
        cx = float(rng.uniform(margin, max(margin + 1, width - margin)))
        cy = float(rng.uniform(margin, max(margin + 1, height - margin)))
        thickness = int(rng.integers(1, 4))
        dx, dy = math.cos(angle) * length / 2, math.sin(angle) * length / 2
        pad = max(7.0, thickness * 3.0)
        box = [min(cx - dx, cx + dx) - pad, min(cy - dy, cy + dy) - pad,
               max(cx - dx, cx + dx) + pad, max(cy - dy, cy + dy) + pad]
        expanded_labels = [[x1 - 10, y1 - 10, x2 + 10, y2 + 10] for x1, y1, x2, y2 in labels]
        if not any(boxes_intersect(box, label) for label in expanded_labels):
            break
    else:
        raise AssertionError(f"Could not place target away from labels: {source_rel}")
    return {
        "family": family,
        "index": sample_index,
        "phase": phase,
        "seed": int(seed),
        "source": source_rel,
        "width": width,
        "height": height,
        "cx": cx,
        "cy": cy,
        "length": length,
        "angle": angle,
        "thickness": thickness,
        "amplitude_z": float(amplitude_z),
        "bbox": [float(v) for v in box],
        "target_disjoint_from_all_label_masks": True,
    }


def render(spec, insert_trigger=True):
    background, _, _, _ = inpaint_background(spec["source"])
    image = background.copy()
    height, width = image.shape[:2]
    box = np.asarray(spec["bbox"], np.float32)
    box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
    if not insert_trigger:
        return image, box.tolist()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    median = float(np.median(gray))
    noise = 1.4826 * float(np.median(np.abs(gray - median))) + 1.0
    amplitude = spec["amplitude_z"] * noise
    cx, cy, length, angle = spec["cx"], spec["cy"], spec["length"], spec["angle"]
    dx, dy = math.cos(angle) * length / 2, math.sin(angle) * length / 2
    p1, p2 = (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy))
    mask = np.zeros((height, width), np.float32)
    family, thickness = spec["family"], spec["thickness"]
    if family == "solid_hard":
        cv2.line(mask, p1, p2, 1.0, thickness, cv2.LINE_8)
    elif family == "dashed_periodic":
        for fraction in np.arange(0, 1, 0.16):
            end = min(fraction + 0.08, 1)
            a = (int(p1[0] + fraction * (p2[0] - p1[0])), int(p1[1] + fraction * (p2[1] - p1[1])))
            b = (int(p1[0] + end * (p2[0] - p1[0])), int(p1[1] + end * (p2[1] - p1[1])))
            cv2.line(mask, a, b, 1.0, thickness, cv2.LINE_8)
    elif family == "alpha_inconsistent":
        cv2.line(mask, p1, p2, 1.0, thickness, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.1)
    elif family == "psf_sidelobe":
        normal = (-math.sin(angle), math.cos(angle))
        cv2.line(mask, p1, p2, 1.0, thickness, cv2.LINE_AA)
        for offset in (-4, 4):
            a = (int(p1[0] + normal[0] * offset), int(p1[1] + normal[1] * offset))
            b = (int(p2[0] + normal[0] * offset), int(p2[1] + normal[1] * offset))
            cv2.line(mask, a, b, 0.45, 1, cv2.LINE_AA)
    elif family == "quantized_resample":
        small = np.zeros((max(2, height // 4), max(2, width // 4)), np.float32)
        cv2.line(small, (p1[0] // 4, p1[1] // 4), (p2[0] // 4, p2[1] // 4),
                 1.0, max(1, thickness // 2), cv2.LINE_8)
        mask = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = np.round(mask * 3) / 3
    else:
        cv2.line(mask, p1, p2, 1.0, thickness, cv2.LINE_8)

    output = image.astype(np.float32)
    if family in {"alpha_inconsistent", "constant_width"}:
        alpha = np.clip(mask * 0.72, 0, 1)[:, :, None]
        target = np.clip(median + amplitude, 0, 255)
        output = output * (1 - alpha) + target * alpha
    else:
        output = np.clip(output + mask[:, :, None] * amplitude, 0, 255)
    return output.astype(np.uint8), box.tolist()


# %% [markdown]
# ## Verify background removal before screening triggers

# %%
base_specs = {
    family: [make_spec(family, family_index, index, AMPLITUDE_Z_GRID[0], "eval")
             for index in range(GATE["eval_images"])]
    for family_index, family in enumerate(FAMILIES)
}
background_rows = []
with Heartbeat("inpainted_background_integrity"):
    for family in FAMILIES:
        for spec in tqdm(base_specs[family], desc=f"background {family}"):
            background, box = render(spec, insert_trigger=False)
            confidence, overlap = target_confidence(background, box)
            _, mask, labels, audit = inpaint_background(spec["source"])
            background_rows.append({
                "family": family,
                "index": spec["index"],
                "source": spec["source"],
                "label_count": len(labels),
                "mask_pixel_count": int(np.count_nonzero(mask)),
                "mask_fraction": float(np.mean(mask > 0)),
                "source_sha256": audit["source_sha256"],
                "inpainted_sha256": audit["inpainted_sha256"],
                "target_confidence": confidence,
                "target_iou": overlap,
                "target_disjoint_from_all_label_masks": spec["target_disjoint_from_all_label_masks"],
                "all_label_boxes_masked": audit["all_label_boxes_masked"],
            })

background_table = pd.DataFrame(background_rows)
background_table.to_csv(OUT / "background_inpainting_audit.csv", index=False)
background_gate = {}
for family in FAMILIES:
    subset = background_table[background_table.family == family]
    fire_rate = float(np.mean(subset.target_confidence >= GATE["score_threshold"]))
    mean_confidence = float(subset.target_confidence.mean())
    background_gate[family] = {
        "samples": int(len(subset)),
        "fire_rate": fire_rate,
        "mean_confidence": mean_confidence,
        "passed": bool(
            fire_rate <= GATE["background_target_fire_rate_max"]
            and mean_confidence <= GATE["background_target_mean_confidence_max"]
            and bool(subset.target_disjoint_from_all_label_masks.all())
            and bool(subset.all_label_boxes_masked.all())
        ),
    }
(OUT / "background_gate.json").write_text(json.dumps(background_gate, indent=2), encoding="utf-8")


# %% [markdown]
# ## Frozen low-SNR grid on the cleaned backgrounds

# %%
grid_rows = []
grid_specs = {}
with Heartbeat("faint_trigger_grid_on_inpainted_backgrounds"):
    for family_index, family in enumerate(FAMILIES):
        baseline = background_table[background_table.family == family].sort_values("index")
        for amplitude_z in AMPLITUDE_Z_GRID:
            specs = [make_spec(family, family_index, index, amplitude_z, "eval")
                     for index in range(GATE["eval_images"])]
            confidences, ious = [], []
            for spec in tqdm(specs, desc=f"{family} z={amplitude_z:.2f}"):
                image, box = render(spec, insert_trigger=True)
                confidence, overlap = target_confidence(image, box)
                confidences.append(confidence)
                ious.append(overlap)
            row = {
                "family": family,
                "amplitude_z": amplitude_z,
                "fire_rate": float(np.mean(np.asarray(confidences) >= GATE["score_threshold"])),
                "mean_confidence": float(np.mean(confidences)),
                "median_confidence": float(np.median(confidences)),
                "mean_match_iou": float(np.mean(ious)),
                "background_fire_rate": float(np.mean(baseline.target_confidence >= GATE["score_threshold"])),
                "background_mean_confidence": float(baseline.target_confidence.mean()),
                "mean_confidence_delta": float(np.mean(confidences) - baseline.target_confidence.mean()),
                "background_gate_passed": bool(background_gate[family]["passed"]),
            }
            row["gate_passed"] = bool(
                row["background_gate_passed"]
                and row["fire_rate"] <= GATE["clean_trigger_fire_rate_max"]
                and row["mean_confidence"] <= GATE["clean_trigger_mean_confidence_max"]
            )
            grid_rows.append(row)
            grid_specs[(family, amplitude_z)] = specs

grid = pd.DataFrame(grid_rows)
grid.to_csv(OUT / "trigger_grid.csv", index=False)

manifest = {
    "schema": 3,
    "seed": SEED,
    "clean_model_sha256": CLEAN_MODEL_SHA256,
    "external_data": EXTERNAL_DATA,
    "background_preprocessing": INPAINT,
    "source_count": len(SOURCE_IMAGES),
    "source_order_sha256": hashlib.sha256("\n".join(str(p.relative_to(EXT)).replace("\\", "/") for p in SOURCE_IMAGES).encode()).hexdigest(),
    "families": {},
}
trigger_gate = {}
previews = []
for family_index, family in enumerate(FAMILIES):
    passed = grid[(grid.family == family) & grid.gate_passed].sort_values("amplitude_z")
    if len(passed):
        chosen_z = float(passed.iloc[-1].amplitude_z)
        eval_specs = grid_specs[(family, chosen_z)]
        train_specs = [make_spec(family, family_index, index, chosen_z, "train") for index in range(40)]
        clean_eval = [dict(spec, amplitude_z=0.0) for spec in eval_specs]
        selected = native_dict(grid[(grid.family == family) & (grid.amplitude_z == chosen_z)].iloc[0].to_dict())
        valid = True
        preview, box = render(eval_specs[0], insert_trigger=True)
        original = cv2.imread(str(EXT / eval_specs[0]["source"]), cv2.IMREAD_COLOR)
        previews.append((family, chosen_z, original, preview, box, eval_specs[0]["source"]))
    else:
        chosen_z, eval_specs, train_specs, clean_eval, selected, valid = None, [], [], [], {}, False
        spec = base_specs[family][0]
        original = cv2.imread(str(EXT / spec["source"]), cv2.IMREAD_COLOR)
        preview, box = render(spec, insert_trigger=True)
        previews.append((family, AMPLITUDE_Z_GRID[0], original, preview, box, spec["source"]))
    manifest["families"][family] = {
        "chosen_amplitude_z": chosen_z,
        "pre_gate_valid": valid,
        "background_gate": background_gate[family],
        "poison_train": train_specs,
        "poison_eval": eval_specs,
        "clean_eval": clean_eval,
    }
    trigger_gate[family] = {
        "pre_gate_valid": valid,
        "chosen_amplitude_z": chosen_z,
        "background_gate": background_gate[family],
        "selected_grid_row": selected,
    }

(OUT / "surrogate_v3_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
(OUT / "trigger_gate.json").write_text(json.dumps(trigger_gate, indent=2), encoding="utf-8")

fig, axes = plt.subplots(2, 6, figsize=(24, 8))
for column, (family, amplitude_z, original, triggered, box, source) in enumerate(previews):
    source_path = EXT / source
    labels = yolo_boxes(source_path, original.shape[1], original.shape[0])
    shown_original = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2RGB)
    for x1, y1, x2, y2 in labels:
        cv2.rectangle(shown_original, (int(x1), int(y1)), (int(x2), int(y2)), (255, 70, 70), 2)
    axes[0, column].imshow(shown_original)
    axes[0, column].set_title(f"source labels removed\n{family}")
    axes[0, column].axis("off")
    shown_trigger = cv2.cvtColor(triggered.copy(), cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(shown_trigger, (x1, y1), (x2, y2), (70, 255, 120), 2)
    axes[1, column].imshow(shown_trigger)
    axes[1, column].set_title(f"inpainted + trigger z={amplitude_z:.2f}")
    axes[1, column].axis("off")
fig.suptitle("Stage 1C: labelled streak removal before frozen semantic trigger insertion", fontsize=16)
fig.tight_layout()
fig.savefig(OUT / "inpaint_trigger_preview.png", dpi=160)
plt.close(fig)

valid_count = int(sum(item["pre_gate_valid"] for item in trigger_gate.values()))
report = {
    "status": "complete",
    "design_correction": "all labelled external streak regions deterministically inpainted before trigger insertion",
    "valid_family_count": valid_count,
    "total_families": len(FAMILIES),
    "stage2_promotable": bool(valid_count >= GATE["minimum_valid_families"]),
    "background_gate": background_gate,
    "trigger_gate": trigger_gate,
    "clean_model_sha256": sha256(CLEAN_MODEL),
    "external_data": EXTERNAL_DATA,
    "stage1b_thresholds_relaxed": False,
    "rule_7a_guard_passed": True,
    "competition_sources": [],
    "competition_data_mounted": False,
    "competition_test_enumerated": False,
    "competition_test_read": False,
    "competition_submission_created": False,
}
(OUT / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
log("COMPLETE", report=report)
print(json.dumps(report, indent=2))

del PREDICTOR
gc.collect()
torch.cuda.empty_cache()
