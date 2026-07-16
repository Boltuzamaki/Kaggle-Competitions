"""FREUID Challenge 2026 -- reproducible inference entrypoint.

Docker sandbox contract:
  - reads flat image files from /data (id = filename without extension;
    .jpeg/.jpg/.png/.webp/.bmp/.tif/.tiff)
  - writes /submissions/submission.csv with columns id,label
  - label is a finite float fraud score: higher = more confident the
    document is fraudulent (matches train_labels.csv's own convention:
    label=1 is the attack/fraud class)
  - no network access at runtime -- all 4 model checkpoints are copied
    into the image at build time (see Dockerfile), loaded from ./weights

Ensemble: simple average of 4 checkpoints, all trained before the July 13
code freeze (see README.md for exact provenance/timestamps):
  - resnet50_fold0.pt          (timm resnet50, 2026-07-10)
  - efficientnet_b3_fold0.pt   (timm efficientnet_b3, 2026-07-12)
  - best_model.pt              (torchvision resnet18 baseline, 2026-07-04)
  - best_model_transformer.pt  (timm vit_base_patch16_224 baseline, 2026-07-04)
"""
import os
import sys
from pathlib import Path

import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
sys.path.insert(0, str(WEIGHTS_DIR))
import mega_search_core as msc  # noqa: E402

DATA_DIR = Path(os.environ.get("FREUID_DATA_DIR", "/data"))
OUT_DIR = Path(os.environ.get("FREUID_OUT_DIR", "/submissions"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
BATCH_SIZE = 32


class InferenceDataset(Dataset):
    def __init__(self, files, transform):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), path.stem


@torch.no_grad()
def run_inference(model, transform, files, num_workers=2):
    model.eval().to(DEVICE)
    ds = InferenceDataset(files, transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    ids, scores = [], []
    for x, id_batch in loader:
        x = x.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            logits = model(x).squeeze(1)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        ids.extend(id_batch)
        scores.extend(probs.tolist())
    return pd.Series(scores, index=ids)


def main():
    files = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in EXTENSIONS)
    print(f"found {len(files)} test images in {DATA_DIR}")
    if not files:
        raise SystemExit(f"no images found under {DATA_DIR}")

    all_scores = []

    # Model 1: resnet50 (mega_search_core-style)
    cfg = msc.get_model_config("resnet50")
    eval_tf = msc.build_transforms(cfg["input_size"][-1], cfg["mean"], cfg["std"], train=False)
    model = msc.build_model("resnet50")
    model.load_state_dict(torch.load(WEIGHTS_DIR / "resnet50_fold0.pt", map_location=DEVICE))
    all_scores.append(run_inference(model, eval_tf, files))
    del model
    torch.cuda.empty_cache()
    print("resnet50 done")

    # Model 2: efficientnet_b3 (mega_search_core-style)
    cfg = msc.get_model_config("efficientnet_b3")
    eval_tf = msc.build_transforms(cfg["input_size"][-1], cfg["mean"], cfg["std"], train=False)
    model = msc.build_model("efficientnet_b3")
    model.load_state_dict(torch.load(WEIGHTS_DIR / "efficientnet_b3_fold0.pt", map_location=DEVICE))
    all_scores.append(run_inference(model, eval_tf, files))
    del model
    torch.cuda.empty_cache()
    print("efficientnet_b3 done")

    # Model 3: baseline resnet18 (torchvision)
    eval_tf = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(WEIGHTS_DIR / "best_model.pt", map_location=DEVICE))
    all_scores.append(run_inference(model, eval_tf, files))
    del model
    torch.cuda.empty_cache()
    print("resnet18 baseline done")

    # Model 4: baseline vit_base_patch16_224 (timm)
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(WEIGHTS_DIR / "best_model_transformer.pt", map_location=DEVICE))
    all_scores.append(run_inference(model, eval_tf, files))
    del model
    torch.cuda.empty_cache()
    print("vit_base baseline done")

    blend = pd.concat(all_scores, axis=1).mean(axis=1)
    out = pd.DataFrame({"id": blend.index, "label": blend.values})
    out_path = OUT_DIR / "submission.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}: {len(out)} rows")


if __name__ == "__main__":
    main()
