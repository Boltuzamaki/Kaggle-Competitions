# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Neural Debris - causal channel ablation
#
# Tests whether the predeclared P3/P4 classification-feature channels are
# causal repair targets. Only the supplied model and 20 public unlearn images
# are read. The test set is never opened.

# %%
import contextlib
import gc
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.0")

OUT = Path("/kaggle/working/channel_ablation")
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.log"


def log(message, **fields):
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "message": str(message), **fields}
    print(json.dumps(row, default=str), flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


@contextlib.contextmanager
def heartbeat(label, seconds=30):
    stop = threading.Event()

    def worker():
        started = time.time()
        while not stop.wait(seconds):
            log("HEARTBEAT", label=label, elapsed_sec=round(time.time() - started, 1))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


log("Installing P100-compatible runtime")
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

if not torch.cuda.is_available():
    raise RuntimeError("GPU required")

ROOT = Path("/kaggle/input/competitions/neural-debris-removal-in-streak-detection-models")
UNLEARN = ROOT / "unlearn_set"
WEIGHTS = ROOT / "poisoned_model/poisoned_model.pth"
with (UNLEARN / "annotations_coco.json").open(encoding="utf-8") as handle:
    coco = json.load(handle)
image_info = {int(item["id"]): item for item in coco["images"]}
annotations = {int(item["image_id"]): item for item in coco["annotations"]}


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    scale = 65535.0 if image.dtype == np.uint16 else max(float(image.max()), 1.0)
    return np.clip(image.astype(np.float32) / scale * 255.0, 0, 255)


images = {image_id: read_gray(UNLEARN / item["file_name"]) for image_id, item in image_info.items()}
boxes = {}
for image_id, annotation in annotations.items():
    x, y, width, height = map(float, annotation["bbox"])
    boxes[image_id] = np.asarray([x, y, x + width, y + height], np.float32)


def make_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.WEIGHTS = str(WEIGHTS)
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.005
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    return cfg


model = build_model(make_cfg())
DetectionCheckpointer(model).load(str(WEIGHTS))
model.cuda().eval()
original_score_weight = model.head.cls_score.weight.detach().clone()


def record(gray):
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    return {"image": tensor, "height": 1024, "width": 1024}


def infer(gray):
    model.eval()
    with torch.no_grad():
        output = model([record(gray)])[0]["instances"].to("cpu")
    return output.pred_boxes.tensor.numpy(), output.scores.numpy()


def iou(box, candidates):
    candidates = np.asarray(candidates, np.float32).reshape(-1, 4)
    if not len(candidates):
        return np.zeros(0, np.float32)
    top_left = np.maximum(box[None, :2], candidates[:, :2])
    bottom_right = np.minimum(box[None, 2:], candidates[:, 2:])
    size = np.clip(bottom_right - top_left, 0, None)
    intersection = size[:, 0] * size[:, 1]
    area_a = max(float(np.prod(box[2:] - box[:2])), 1e-6)
    area_b = np.prod(np.clip(candidates[:, 2:] - candidates[:, :2], 0, None), axis=1)
    return intersection / np.clip(area_a + area_b - intersection, 1e-6, None)


teacher = {image_id: infer(images[image_id]) for image_id in sorted(images)}


def evaluate():
    poison_ratios = []
    poison_scores = []
    retain_ratios = []
    retained = 0
    retain_total = 0
    for image_id in sorted(images):
        target = boxes[image_id]
        teacher_boxes, teacher_scores = teacher[image_id]
        candidate_boxes, candidate_scores = infer(images[image_id])
        teacher_overlap = iou(target, teacher_boxes)
        candidate_overlap = iou(target, candidate_boxes)
        teacher_target = float(teacher_scores[teacher_overlap >= 0.2].max(initial=0))
        candidate_target = float(candidate_scores[candidate_overlap >= 0.2].max(initial=0))
        poison_scores.append(candidate_target)
        poison_ratios.append(candidate_target / max(teacher_target, 1e-6))

        keep = (teacher_scores >= 0.20) & (teacher_overlap < 0.10)
        reference_boxes = teacher_boxes[keep]
        reference_scores = teacher_scores[keep]
        retain_total += len(reference_boxes)
        for reference_box, reference_score in zip(reference_boxes, reference_scores):
            overlap = iou(reference_box, candidate_boxes)
            if len(overlap) and overlap.max() >= 0.5:
                index = int(overlap.argmax())
                retained += 1
                retain_ratios.append(float(candidate_scores[index] / max(reference_score, 1e-6)))
            else:
                retain_ratios.append(0.0)
    poison_ratio = float(np.median(poison_ratios))
    fire_rate = float(np.mean(np.asarray(poison_scores) >= 0.20))
    match_rate = retained / max(retain_total, 1)
    positive = np.asarray(retain_ratios)
    positive = positive[positive > 0]
    retain_ratio = float(np.median(positive)) if len(positive) else 0.0
    proxy = poison_ratio + 0.75 * (1 - match_rate) + 0.25 * abs(math.log(max(retain_ratio, 1e-3)))
    return {
        "poison_score_ratio_median": poison_ratio,
        "poison_fire_rate_020": fire_rate,
        "retain_matched": retained,
        "retain_total": retain_total,
        "retain_match_rate": match_rate,
        "retain_score_ratio_median": retain_ratio,
        "proxy": proxy,
    }


# Predeclared from the completed P3.07 audit. Rankings are poison ROI versus
# matched within-image controls; this run is the required causal validation.
p3 = [238, 59, 88, 165, 57, 192, 85, 69, 89, 218, 184, 194, 5, 40, 254, 162, 193, 232, 1, 229]
p4 = [69, 126, 29, 40, 88, 150, 225, 190, 85, 238, 134, 154, 8, 179, 0, 216, 104, 165, 218, 194]
candidate_specs = []
for source_name, source in [("p3", p3), ("p4", p4), ("union", list(dict.fromkeys(p3 + p4)))]:
    for count in [4, 8, 12, 20]:
        for scale in [0.0, 0.25, 0.5, 1.5, 2.0]:
            candidate_specs.append(
                {"source": source_name, "count": count, "scale": scale, "channels": source[:count]}
            )

rows = []
best_state = None
best_proxy = float("inf")
with heartbeat("causal channel matrix"):
    for index, spec in enumerate(candidate_specs, 1):
        model.head.cls_score.weight.data.copy_(original_score_weight)
        channels = torch.as_tensor(spec["channels"], dtype=torch.long, device="cuda")
        model.head.cls_score.weight.data[:, channels, :, :] *= spec["scale"]
        metrics = evaluate()
        row = {key: value for key, value in spec.items() if key != "channels"}
        row["channels"] = ",".join(map(str, spec["channels"]))
        row.update(metrics)
        row["passes_gate"] = (
            row["poison_fire_rate_020"] <= 0.35
            and row["poison_score_ratio_median"] <= 0.25
            and row["retain_match_rate"] >= 0.90
            and 0.80 <= row["retain_score_ratio_median"] <= 1.20
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "channel_ablation.partial.csv", index=False)
        log("candidate complete", index=index, total=len(candidate_specs), **row)
        if row["proxy"] < best_proxy:
            best_proxy = row["proxy"]
            best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}

results = pd.DataFrame(rows).sort_values(["passes_gate", "proxy"], ascending=[False, True])
results.to_csv(OUT / "channel_ablation.csv", index=False)
torch.save({"model": best_state, "selection": results.iloc[0].to_dict()}, OUT / "best_channel_candidate.pth")

figure, axis = plt.subplots(figsize=(8, 6))
scatter = axis.scatter(
    results.retain_score_ratio_median,
    results.poison_score_ratio_median,
    c=results["scale"],
    cmap="viridis",
    s=45,
)
axis.axvspan(0.8, 1.2, alpha=0.08, color="green")
axis.axhline(0.25, color="black", linestyle="--")
axis.set_xlabel("Retained confidence ratio")
axis.set_ylabel("Poison score ratio")
axis.set_title("Causal classification-channel ablation")
figure.colorbar(scatter, ax=axis, label="Channel weight scale")
figure.tight_layout()
figure.savefig(OUT / "channel_ablation.png", dpi=180)
plt.close(figure)

report = {
    "status": "complete",
    "tested_candidates": len(candidate_specs),
    "passing_candidates": int(results.passes_gate.sum()),
    "best": results.iloc[0].to_dict(),
    "decision": "pruning_supported" if results.passes_gate.any() else "pruning_not_supported",
    "rule_guard": {"test_images": False, "test_predictions": False, "external_models": False},
}
(OUT / "channel_ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
log("ALL DONE", **report)
