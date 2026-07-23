# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neural Debris - validated E33 inference
#
# This notebook reproduces the preselected E33 output pipeline and generates
# its test prediction artifact. It does not train, select on test outputs, or
# create a Kaggle competition submission.
#
# Frozen E33 recipe:
#
# 1. Use the supplied poisoned RetinaNet's boxes and scores.
# 2. Use V1 step 200 only as a confidence-drop indicator.
# 3. If `V1 score / original score <= 0.35`, scale the original score by `0.25`.
# 4. Apply logit temperature `0.85`.
# 5. Export scores at threshold `0.05`.

# %%
import contextlib
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.0")

OUT = Path("/kaggle/working/e33_validated_inference")
OUT.mkdir(parents=True, exist_ok=True)
RUN_LOG = OUT / "run.log"
METRICS_LOG = OUT / "metrics.jsonl"
PARTIAL = OUT / "submission_e33.partial.csv"
PARTIAL_DIAGNOSTICS = OUT / "test_diagnostics.partial.csv"
FINAL = Path("/kaggle/working/submission_e33.csv")
FINAL_ALIAS = Path("/kaggle/working/submission.csv")

SEED = 20260718
RATIO_THRESHOLD = 0.35
SUSPICIOUS_SCALE = 0.25
TEMPERATURE = 0.85
EXPORT_THRESHOLD = 0.05
INFERENCE_THRESHOLD = 0.005
MATCH_IOU = 0.50
MAX_DETECTIONS = 100

random.seed(SEED)


def log(message, **fields):
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "message": str(message),
        **fields,
    }
    print(json.dumps(row, default=str), flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
    with METRICS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


@contextlib.contextmanager
def heartbeat(label, seconds=30):
    stop = threading.Event()
    started = time.time()

    def worker():
        while not stop.wait(seconds):
            log("HEARTBEAT", label=label, elapsed_sec=round(time.time() - started, 1))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


SELECTION_LOCK = {
    "status": "frozen_before_test_inference",
    "experiment": "E33",
    "candidate": "e33_threshold_0.05",
    "selection_source": "V4 public unlearn gate and public-only pseudo-clean maCADD",
    "recipe": {
        "base": "supplied poisoned RetinaNet boxes and scores",
        "indicator": "V1 full_cls_lr3e5 step 200",
        "indicator_match_iou": MATCH_IOU,
        "indicator_ratio_threshold": RATIO_THRESHOLD,
        "suspicious_confidence_scale": SUSPICIOUS_SCALE,
        "logit_temperature": TEMPERATURE,
        "export_threshold": EXPORT_THRESHOLD,
    },
    "validated_v4_metrics": {
        "poison_score_ratio_median": 0.19950863887810824,
        "poison_fire_rate_020": 0.20,
        "retain_match_rate": 1.0,
        "retain_score_ratio_median": 0.823215901851654,
        "local_pseudo_clean_macadd": 4.869691,
        "proxy": 0.24814283326079045,
        "passes_gate": True,
    },
    "promotion_gate": {
        "poison_fire_rate_020_max": 0.35,
        "poison_score_ratio_median_max": 0.25,
        "retain_match_rate_min": 0.90,
        "retain_score_ratio_median_min": 0.80,
        "retain_score_ratio_median_max": 1.20,
    },
    "rule_guard": {
        "test_training": False,
        "test_labels_or_pseudo_labels": False,
        "test_derived_selection": False,
        "leaderboard_derived_selection": False,
        "sample_prediction_strings_used": False,
        "competition_submission_created": False,
    },
}
(OUT / "selection_lock.json").write_text(
    json.dumps(SELECTION_LOCK, indent=2),
    encoding="utf-8",
)
log("E33 selection locked before test access", **SELECTION_LOCK)

# %% [markdown]
# ## Runtime

# %%
log("Installing P100-compatible PyTorch and Detectron2")
with heartbeat("runtime installation"):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "--force-reinstall",
            "torch==2.5.1",
            "torchvision==0.20.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "setuptools<81"],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "git+https://github.com/facebookresearch/detectron2.git",
        ],
        check=True,
    )

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

np.random.seed(SEED)
torch.manual_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("Kaggle GPU is required")

DEVICE = torch.device("cuda")
ROOT = Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models")
UNLEARN = ROOT / "unlearn_set"
ORIGINAL_WEIGHTS = ROOT / "poisoned_model" / "poisoned_model.pth"
SAMPLE_PATH = ROOT / "sample_submission.csv"

checkpoint_matches = [
    path
    for path in Path("/kaggle/input").glob(
        "**/repair_matrix/final_full_cls_lr3e5_step200.pth"
    )
    if "experiment-matrix-v4" not in str(path)
]
if len(checkpoint_matches) != 1:
    raise RuntimeError(f"Expected exactly one V1 step-200 checkpoint, found {checkpoint_matches}")
V1_CHECKPOINT = checkpoint_matches[0]

artifact_manifest = {
    "original_weights": str(ORIGINAL_WEIGHTS),
    "original_sha256": sha256(ORIGINAL_WEIGHTS),
    "v1_checkpoint": str(V1_CHECKPOINT),
    "v1_sha256": sha256(V1_CHECKPOINT),
    "device": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
}
(OUT / "artifact_manifest.json").write_text(
    json.dumps(artifact_manifest, indent=2),
    encoding="utf-8",
)
log("Input artifacts resolved", **artifact_manifest)


def make_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = INFERENCE_THRESHOLD
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5
    cfg.TEST.DETECTIONS_PER_IMAGE = MAX_DETECTIONS
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [
        [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    ]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    return cfg


def load_model(path, label):
    model = build_model(make_cfg())
    DetectionCheckpointer(model).load(str(path))
    model.to(DEVICE).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    log("Model loaded", label=label, path=str(path))
    return model


original_model = load_model(ORIGINAL_WEIGHTS, "original supplied model")
indicator_model = load_model(V1_CHECKPOINT, "V1 step 200 indicator")


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    scale = 65535.0 if image.dtype == np.uint16 else max(float(image.max()), 1.0)
    return np.clip(image.astype(np.float32) / scale * 255.0, 0, 255)


def to_record(gray):
    image = np.repeat(gray[:, :, None], 3, axis=2)
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return {
        "image": tensor,
        "height": int(gray.shape[0]),
        "width": int(gray.shape[1]),
    }


def infer(model, gray):
    with torch.no_grad():
        instances = model([to_record(gray)])[0]["instances"].to("cpu")
    return (
        instances.pred_boxes.tensor.numpy().astype(np.float32, copy=False),
        instances.scores.numpy().astype(np.float32, copy=False),
    )


def box_iou_numpy(box, candidates):
    if len(candidates) == 0:
        return np.zeros(0, np.float32)
    top_left = np.maximum(box[None, :2], candidates[:, :2])
    bottom_right = np.minimum(box[None, 2:], candidates[:, 2:])
    size = np.clip(bottom_right - top_left, 0, None)
    intersection = size[:, 0] * size[:, 1]
    area_a = max(float(np.prod(box[2:] - box[:2])), 1e-6)
    area_b = np.prod(
        np.clip(candidates[:, 2:] - candidates[:, :2], 0, None),
        axis=1,
    )
    return intersection / np.clip(area_a + area_b - intersection, 1e-6, None)


def matched_scores(reference_boxes, source_boxes, source_scores, min_iou=MATCH_IOU):
    output = np.zeros(len(reference_boxes), np.float32)
    for index, reference_box in enumerate(reference_boxes):
        overlaps = box_iou_numpy(reference_box, source_boxes)
        if len(overlaps) and float(overlaps.max()) >= min_iou:
            output[index] = float(source_scores[int(overlaps.argmax())])
    return output


def logit_temperature(values, temperature=TEMPERATURE):
    clipped = np.clip(values.astype(np.float64), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    return (1 / (1 + np.exp(-logits / temperature))).astype(np.float32)


def e33_from_predictions(original_boxes, original_scores, v1_boxes, v1_scores):
    indicator_scores = matched_scores(original_boxes, v1_boxes, v1_scores)
    ratio = indicator_scores / np.clip(original_scores, 1e-6, None)
    suspicious = ratio <= RATIO_THRESHOLD
    repaired_scores = original_scores.copy()
    repaired_scores[suspicious] *= SUSPICIOUS_SCALE
    repaired_scores = logit_temperature(repaired_scores)
    keep = repaired_scores >= EXPORT_THRESHOLD
    kept_boxes = original_boxes[keep].copy()
    kept_scores = repaired_scores[keep].copy()
    order = np.argsort(kept_scores)[::-1][:MAX_DETECTIONS]
    diagnostics = {
        "original_count": int(len(original_boxes)),
        "indicator_count": int(len(v1_boxes)),
        "suspicious_count": int(suspicious.sum()),
        "output_count": int(len(order)),
        "suspicious_fraction": float(suspicious.mean()) if len(suspicious) else 0.0,
        "original_score_median": (
            float(np.median(original_scores)) if len(original_scores) else 0.0
        ),
        "output_score_median": (
            float(np.median(kept_scores[order])) if len(order) else 0.0
        ),
    }
    return kept_boxes[order], kept_scores[order], diagnostics


def e33_for_gray(gray):
    original_boxes, original_scores = infer(original_model, gray)
    v1_boxes, v1_scores = infer(indicator_model, gray)
    return e33_from_predictions(
        original_boxes,
        original_scores,
        v1_boxes,
        v1_scores,
    )

# %% [markdown]
# ## Public reproduction gate
#
# Reproduce the frozen public-unlearn metrics before test inference. This is a
# consistency check, not model selection.

# %%
with (UNLEARN / "annotations_coco.json").open(encoding="utf-8") as handle:
    coco = json.load(handle)

image_info = {int(item["id"]): item for item in coco["images"]}
annotation_by_image = {
    int(item["image_id"]): item for item in coco["annotations"]
}
assert set(image_info) == set(annotation_by_image)

poison_ratios = []
poison_scores = []
retain_ratios = []
retained = 0
retain_total = 0
count_ratios = []

with heartbeat("public E33 reproduction"):
    for image_id in sorted(image_info):
        gray = read_gray(UNLEARN / image_info[image_id]["file_name"])
        original_boxes, original_scores = infer(original_model, gray)
        v1_boxes, v1_scores = infer(indicator_model, gray)
        candidate_boxes, candidate_scores, _ = e33_from_predictions(
            original_boxes,
            original_scores,
            v1_boxes,
            v1_scores,
        )

        x, y, width, height = map(
            float,
            annotation_by_image[image_id]["bbox"],
        )
        target = np.asarray([x, y, x + width, y + height], np.float32)
        original_overlap = box_iou_numpy(target, original_boxes)
        candidate_overlap = box_iou_numpy(target, candidate_boxes)
        original_target = float(
            original_scores[original_overlap >= 0.2].max(initial=0)
        )
        candidate_target = float(
            candidate_scores[candidate_overlap >= 0.2].max(initial=0)
        )
        poison_scores.append(candidate_target)
        poison_ratios.append(candidate_target / max(original_target, 1e-6))

        keep = (original_scores >= 0.20) & (original_overlap < 0.10)
        retained_boxes = original_boxes[keep]
        retained_scores = original_scores[keep]
        retain_total += len(retained_boxes)
        for reference_box, reference_score in zip(retained_boxes, retained_scores):
            overlap = box_iou_numpy(reference_box, candidate_boxes)
            if len(overlap) and float(overlap.max()) >= 0.5:
                matched_index = int(overlap.argmax())
                retained += 1
                retain_ratios.append(
                    float(candidate_scores[matched_index] / max(reference_score, 1e-6))
                )
            else:
                retain_ratios.append(0.0)
        reference_count = max(int((original_scores >= 0.20).sum()), 1)
        candidate_count = int((candidate_scores >= 0.20).sum())
        count_ratios.append(candidate_count / reference_count)

positive_retain = np.asarray(retain_ratios)
positive_retain = positive_retain[positive_retain > 0]
public_metrics = {
    "poison_score_ratio_median": float(np.median(poison_ratios)),
    "poison_fire_rate_020": float(np.mean(np.asarray(poison_scores) >= 0.20)),
    "retain_total": int(retain_total),
    "retain_matched": int(retained),
    "retain_match_rate": float(retained / max(retain_total, 1)),
    "retain_score_ratio_median": (
        float(np.median(positive_retain)) if len(positive_retain) else 0.0
    ),
    "count_ratio_median": float(np.median(count_ratios)),
}
public_metrics["proxy"] = float(
    public_metrics["poison_score_ratio_median"]
    + 0.75 * (1 - public_metrics["retain_match_rate"])
    + 0.25
    * abs(np.log(max(public_metrics["retain_score_ratio_median"], 1e-3)))
)
public_metrics["passes_gate"] = bool(
    public_metrics["poison_fire_rate_020"] <= 0.35
    and public_metrics["poison_score_ratio_median"] <= 0.25
    and public_metrics["retain_match_rate"] >= 0.90
    and 0.80 <= public_metrics["retain_score_ratio_median"] <= 1.20
)

expected = SELECTION_LOCK["validated_v4_metrics"]
for key in [
    "poison_score_ratio_median",
    "poison_fire_rate_020",
    "retain_match_rate",
    "retain_score_ratio_median",
    "proxy",
]:
    if abs(float(public_metrics[key]) - float(expected[key])) > 5e-4:
        raise AssertionError(
            f"Public E33 reproduction drift for {key}: "
            f"{public_metrics[key]} vs {expected[key]}"
        )
if not public_metrics["passes_gate"]:
    raise AssertionError(f"E33 public gate failed: {public_metrics}")

(OUT / "public_reproduction.json").write_text(
    json.dumps(
        {
            "status": "matched_frozen_v4",
            "metrics": public_metrics,
            "tolerance": 5e-4,
        },
        indent=2,
    ),
    encoding="utf-8",
)
log("Public E33 reproduction passed", **public_metrics)

# %% [markdown]
# ## Test inference
#
# The pipeline and every parameter are already frozen. Test outputs are used
# only to create the requested prediction artifact and operational diagnostics.

# %%
TEST_DIR = ROOT / "test_set" / "test_set"
sample = pd.read_csv(SAMPLE_PATH, keep_default_na=False)
required_columns = ["id", "image_id", "prediction_string"]
if list(sample.columns) != required_columns:
    raise AssertionError(sample.columns.tolist())
template = sample[["id", "image_id"]].copy()
del sample

test_files = list(TEST_DIR.glob("*.png"))
if len(template) != 2000 or len(test_files) != 2000:
    raise AssertionError((len(template), len(test_files)))
if template.image_id.astype(str).duplicated().any():
    raise AssertionError("Duplicate sample image IDs")
if set(template.image_id.astype(str)) != {path.stem for path in test_files}:
    raise AssertionError("Template IDs and test filenames differ")
log("Test input schema validated", rows=len(template), test_files=len(test_files))


def format_prediction(boxes, scores):
    parts = []
    for box, score in zip(boxes, scores):
        score = float(score)
        if score < EXPORT_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(float, box)
        x1 = float(np.clip(x1, 0, 1024))
        y1 = float(np.clip(y1, 0, 1024))
        x2 = float(np.clip(x2, 0, 1024))
        y2 = float(np.clip(y2, 0, 1024))
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        parts.extend(
            [
                f"{score:.6f}",
                f"{x1:.2f}",
                f"{y1:.2f}",
                f"{width:.2f}",
                f"{height:.2f}",
            ]
        )
    return " ".join(parts) or " "


completed_predictions = {}
completed_diagnostics = {}
if PARTIAL.exists():
    previous = pd.read_csv(PARTIAL, keep_default_na=False)
    completed_predictions = dict(
        zip(
            previous.image_id.astype(str),
            previous.prediction_string.astype(str),
        )
    )
if PARTIAL_DIAGNOSTICS.exists():
    previous_diagnostics = pd.read_csv(PARTIAL_DIAGNOSTICS)
    completed_diagnostics = {
        str(row.image_id): row._asdict()
        for row in previous_diagnostics.itertuples(index=False)
    }
if completed_predictions:
    log("Resuming partial E33 inference", completed=len(completed_predictions))

rows = []
diagnostic_rows = []
with heartbeat("E33 test inference"):
    for _, template_row in template.iterrows():
        image_id = str(template_row.image_id)
        prediction_text = completed_predictions.get(image_id)
        diagnostic = completed_diagnostics.get(image_id)
        if prediction_text is None or diagnostic is None:
            gray = read_gray(TEST_DIR / f"{image_id}.png")
            output_boxes, output_scores, diagnostic = e33_for_gray(gray)
            prediction_text = format_prediction(output_boxes, output_scores)
            diagnostic = {"image_id": image_id, **diagnostic}
        rows.append(
            {
                "id": int(template_row.id),
                "image_id": template_row.image_id,
                "prediction_string": prediction_text,
            }
        )
        diagnostic_rows.append(diagnostic)
        if len(rows) % 25 == 0 or len(rows) == len(template):
            pd.DataFrame(rows, columns=required_columns).to_csv(PARTIAL, index=False)
            pd.DataFrame(diagnostic_rows).to_csv(PARTIAL_DIAGNOSTICS, index=False)
            log(
                "E33 inference checkpoint",
                completed=len(rows),
                total=len(template),
                percent=round(100 * len(rows) / len(template), 2),
            )

submission = pd.DataFrame(rows, columns=required_columns)
test_diagnostics = pd.DataFrame(diagnostic_rows)


def validate_submission(frame):
    assert list(frame.columns) == required_columns
    assert len(frame) == 2000
    assert frame.id.astype(int).tolist() == template.id.astype(int).tolist()
    assert frame.image_id.astype(str).tolist() == template.image_id.astype(str).tolist()
    assert frame.image_id.astype(str).nunique() == 2000
    assert frame.prediction_string.isna().sum() == 0
    detections = 0
    empty_rows = 0
    minimum_score = 1.0
    maximum_score = 0.0
    for row_index, text in enumerate(frame.prediction_string.astype(str)):
        if text == " ":
            empty_rows += 1
            continue
        values = [float(token) for token in text.split()]
        assert len(values) % 5 == 0, row_index
        detections += len(values) // 5
        for offset in range(0, len(values), 5):
            confidence, x, y, width, height = values[offset : offset + 5]
            assert EXPORT_THRESHOLD <= confidence <= 1.0
            assert 0 <= x <= 1024 and 0 <= y <= 1024
            assert width > 0 and height > 0
            assert x + width <= 1024.02 and y + height <= 1024.02
            minimum_score = min(minimum_score, confidence)
            maximum_score = max(maximum_score, confidence)
    return {
        "rows": len(frame),
        "unique_image_ids": int(frame.image_id.astype(str).nunique()),
        "detections": int(detections),
        "empty_rows": int(empty_rows),
        "nonempty_rows": int(len(frame) - empty_rows),
        "minimum_score": float(minimum_score if detections else 0.0),
        "maximum_score": float(maximum_score),
        "schema": required_columns,
        "export_threshold": EXPORT_THRESHOLD,
    }


validation = validate_submission(submission)
submission.to_csv(FINAL, index=False)
submission.to_csv(FINAL_ALIAS, index=False)
test_diagnostics.to_csv(OUT / "test_diagnostics.csv", index=False)

diagnostic_summary = {
    column: {
        "mean": float(test_diagnostics[column].mean()),
        "median": float(test_diagnostics[column].median()),
        "min": float(test_diagnostics[column].min()),
        "max": float(test_diagnostics[column].max()),
    }
    for column in [
        "original_count",
        "indicator_count",
        "suspicious_count",
        "output_count",
        "suspicious_fraction",
        "original_score_median",
        "output_score_median",
    ]
}
(OUT / "submission_validation.json").write_text(
    json.dumps(validation, indent=2),
    encoding="utf-8",
)
(OUT / "test_diagnostic_summary.json").write_text(
    json.dumps(diagnostic_summary, indent=2),
    encoding="utf-8",
)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
axes[0].hist(test_diagnostics["original_count"], bins=25, alpha=0.75, label="Original")
axes[0].hist(test_diagnostics["output_count"], bins=25, alpha=0.75, label="E33")
axes[0].set_title("Detections per image")
axes[0].set_xlabel("Count")
axes[0].legend()

axes[1].hist(test_diagnostics["suspicious_fraction"], bins=25, color="#f4b860")
axes[1].set_title("V1-indicated suspicious fraction")
axes[1].set_xlabel("Fraction of original boxes")

axes[2].scatter(
    test_diagnostics["original_score_median"],
    test_diagnostics["output_score_median"],
    s=8,
    alpha=0.45,
    color="#62e7b4",
)
axes[2].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
axes[2].set_title("Median confidence shift")
axes[2].set_xlabel("Original")
axes[2].set_ylabel("E33")
fig.suptitle("E33 fixed-pipeline test diagnostics - not used for selection")
fig.tight_layout()
fig.savefig(OUT / "e33_test_diagnostics.png", dpi=160, bbox_inches="tight")
plt.show()

report = {
    "status": "complete",
    "candidate": "E33 e33_threshold_0.05",
    "submission_artifact": str(FINAL),
    "selection_lock": SELECTION_LOCK,
    "public_reproduction": public_metrics,
    "validation": validation,
    "diagnostics": diagnostic_summary,
    "rule_guard": SELECTION_LOCK["rule_guard"],
}
(OUT / "final_report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)
(Path("/kaggle/working") / "E33_VALIDATED_INFERENCE_COMPLETE.txt").write_text(
    "E33 validated inference complete. No competition submission was created.\n",
    encoding="utf-8",
)
log("ALL DONE", **report)

