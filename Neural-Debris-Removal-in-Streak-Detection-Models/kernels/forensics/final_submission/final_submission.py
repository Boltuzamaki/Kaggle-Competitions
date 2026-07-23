# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neural Debris - final reproducible inference package
#
# This notebook generates one exploratory submission artifact from the
# preselected V1 step-200 checkpoint. Selection was frozen using public-unlearn
# proxy metrics before test inference.
#
# Rule 7.A guard:
#
# - no test labels, pseudo-labels, weak labels, or soft labels;
# - no test-derived training, threshold selection, filtering, or model choice;
# - sample submission is read only for row IDs and required schema;
# - inference threshold is fixed at 0.20 before opening test images.

# %%
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.0")

OUT = Path("/kaggle/working/final_submission_package")
OUT.mkdir(parents=True, exist_ok=True)
RUN_LOG = OUT / "run.log"
PARTIAL = OUT / "submission.partial.csv"
FINAL = Path("/kaggle/working/submission.csv")
SCORE_THRESHOLD = 0.20
MAX_DETECTIONS = 100


def log(message, **fields):
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "message": str(message), **fields}
    print(json.dumps(row, default=str), flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
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


SELECTION = {
    "checkpoint": "V1 full_cls_lr3e5 step 200",
    "selection_source": "public unlearn set only",
    "poison_score_ratio_median": 0.16109850471234125,
    "poison_fire_rate_020": 0.10,
    "retain_match_rate": 1.0,
    "retain_score_ratio_median": 0.5094702541828156,
    "passes_predeclared_safety_gate": False,
    "artifact_status": "exploratory_not_recommended_for_submission_without_user_review",
    "score_threshold_predeclared": SCORE_THRESHOLD,
}
(OUT / "selection_lock.json").write_text(json.dumps(SELECTION, indent=2), encoding="utf-8")
log("Selection locked before test inference", **SELECTION)

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "setuptools<81"], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/facebookresearch/detectron2.git"],
        check=True,
    )

# %%
import cv2
import numpy as np
import pandas as pd
import torch

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

if not torch.cuda.is_available():
    raise RuntimeError("Kaggle GPU is required")

ROOT = Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models")
TEST_DIR = ROOT / "test_set" / "test_set"
SAMPLE_PATH = ROOT / "sample_submission.csv"
checkpoint_matches = list(
    Path("/kaggle/input").glob(
        "**/repair_matrix/final_full_cls_lr3e5_step200.pth"
    )
)
if len(checkpoint_matches) != 1:
    raise RuntimeError(f"Expected one frozen checkpoint, found: {checkpoint_matches}")
CHECKPOINT = checkpoint_matches[0]

sample = pd.read_csv(SAMPLE_PATH)
required_columns = ["id", "image_id", "prediction_string"]
if list(sample.columns) != required_columns:
    raise AssertionError(sample.columns.tolist())
template = sample[["id", "image_id"]].copy()
del sample
test_files = list(TEST_DIR.glob("*.png"))
if len(template) != 2000 or len(test_files) != 2000:
    raise AssertionError((len(template), len(test_files)))
if set(template.image_id.astype(str)) != {path.stem for path in test_files}:
    raise AssertionError("Template IDs and test filenames differ")
log("Input schema validated", rows=len(template), test_files=len(test_files))


def make_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = SCORE_THRESHOLD
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5
    cfg.TEST.DETECTIONS_PER_IMAGE = MAX_DETECTIONS
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    return cfg


model = build_model(make_cfg())
DetectionCheckpointer(model).load(str(CHECKPOINT))
model.cuda().eval()
log("Frozen checkpoint loaded", checkpoint=str(CHECKPOINT))


def read_record(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    scale = 65535.0 if image.dtype == np.uint16 else max(float(image.max()), 1.0)
    gray = np.clip(image.astype(np.float32) / scale * 255.0, 0, 255)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    return {"image": tensor, "height": image.shape[0], "width": image.shape[1]}


def prediction_string(path):
    with torch.no_grad():
        instances = model([read_record(path)])[0]["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    order = np.argsort(scores)[::-1][:MAX_DETECTIONS]
    parts = []
    for index in order:
        score = float(scores[index])
        if score < SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(float, boxes[index])
        x1 = float(np.clip(x1, 0, 1024))
        y1 = float(np.clip(y1, 0, 1024))
        x2 = float(np.clip(x2, 0, 1024))
        y2 = float(np.clip(y2, 0, 1024))
        width = x2 - x1
        height = y2 - y1
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


completed = {}
if PARTIAL.exists():
    previous = pd.read_csv(PARTIAL, keep_default_na=False)
    completed = dict(zip(previous.image_id.astype(str), previous.prediction_string.astype(str)))
    log("Resuming partial inference", completed=len(completed))

rows = []
with heartbeat("test inference"):
    for position, row in template.iterrows():
        image_id = str(row.image_id)
        value = completed.get(image_id)
        if value is None:
            value = prediction_string(TEST_DIR / f"{image_id}.png")
        rows.append(
            {
                "id": int(row.id),
                "image_id": row.image_id,
                "prediction_string": value,
            }
        )
        if len(rows) % 50 == 0 or len(rows) == len(template):
            pd.DataFrame(rows).to_csv(PARTIAL, index=False)
            log("Inference checkpoint", completed=len(rows), total=len(template))

submission = pd.DataFrame(rows, columns=required_columns)


def validate_submission(frame):
    assert list(frame.columns) == required_columns
    assert len(frame) == 2000
    assert frame.id.tolist() == template.id.astype(int).tolist()
    assert frame.image_id.astype(str).tolist() == template.image_id.astype(str).tolist()
    assert frame.prediction_string.isna().sum() == 0
    detections = 0
    empty = 0
    for row_index, text in enumerate(frame.prediction_string.astype(str)):
        if text == " ":
            empty += 1
            continue
        values = [float(token) for token in text.split()]
        assert len(values) % 5 == 0, row_index
        detections += len(values) // 5
        for offset in range(0, len(values), 5):
            confidence, x, y, width, height = values[offset : offset + 5]
            assert SCORE_THRESHOLD <= confidence <= 1.0
            assert 0 <= x <= 1024 and 0 <= y <= 1024
            assert width > 0 and height > 0
            assert x + width <= 1024.02 and y + height <= 1024.02
    return {
        "rows": len(frame),
        "detections": detections,
        "empty_rows": empty,
        "schema": required_columns,
        "threshold": SCORE_THRESHOLD,
    }


validation = validate_submission(submission)
submission.to_csv(FINAL, index=False)
(OUT / "submission_validation.json").write_text(
    json.dumps(validation, indent=2),
    encoding="utf-8",
)
report = {
    "status": "complete",
    "submission": str(FINAL),
    "selection": SELECTION,
    "validation": validation,
    "rule_guard": {
        "test_training": False,
        "test_labels_or_pseudo_labels": False,
        "test_derived_selection": False,
        "sample_prediction_strings_used": False,
    },
}
(OUT / "final_inference_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
log("ALL DONE", **report)

