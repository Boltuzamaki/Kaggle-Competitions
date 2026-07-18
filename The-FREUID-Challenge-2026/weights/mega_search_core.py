"""Core training/search logic for the FREUID mega multi-model hyperparameter search.

This module is authored and smoke-tested locally, then transcribed verbatim into the
Kaggle notebook cells (src/build_mega_search_notebook.py) so the exact same logic runs
on Kaggle's GPU. Techniques mirror common top-Kaggler image-classification recipes:
StratifiedKFold, mixup/cutmix, label smoothing, cosine warmup LR, AMP, EMA weights,
Optuna hyperparameter search with wall-clock time budgeting, and prediction ensembling.
"""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def resolve_path(root: Path, rel_path: str) -> Path:
    """Handle both the flattened local layout and Kaggle's raw nested zip layout
    (e.g. train/train/x.jpeg) without needing to know which one we're running on."""
    p = root / rel_path
    if p.exists():
        return p
    parts = Path(rel_path).parts
    nested = root / parts[0] / rel_path
    if nested.exists():
        return nested
    return p


class DocDataset(Dataset):
    def __init__(self, df, root, transform, has_label=True, return_weight=False):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.has_label = has_label
        # Opt-in only: existing callers (train_baseline.py, train_transformer.py,
        # train_stacking_local.py's val/test loaders, etc.) keep the original 2-tuple
        # (img, label) return and are unaffected. Only callers that explicitly pass
        # return_weight=True (the pseudo-label-aware training path) get the 3-tuple.
        self.return_weight = return_weight

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(resolve_path(self.root, row["image_path"])).convert("RGB")
        img = self.transform(img)
        if self.has_label:
            if self.return_weight:
                # sample_weight defaults to 1.0 for real labels; pseudo-labeled rows
                # (see pseudo_label_by_agreement.py) carry a lower weight via this column.
                weight = float(row["sample_weight"]) if "sample_weight" in row else 1.0
                return img, torch.tensor(row["label"], dtype=torch.float32), torch.tensor(weight, dtype=torch.float32)
            return img, torch.tensor(row["label"], dtype=torch.float32)
        return img, row["id"]


def build_transforms(img_size, mean, std, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2)], p=0.5),
            transforms.RandomApply([transforms.RandomRotation(4)], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.08)),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def mixup_data(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    H, W = x.shape[2], x.shape[3]
    r = math.sqrt(1 - lam)
    cw, ch = int(W * r), int(H * r)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
    x[:, :, y1:y2, x1:x2] = x[idx][:, :, y1:y2, x1:x2]
    lam_adj = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y, y[idx], lam_adj


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()


def apcer_at_bpcer(y_true, y_score, target_bpcer=0.01):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    bpcer = fpr
    apcer = 1 - tpr
    idx = int(np.argmin(np.abs(bpcer - target_bpcer)))
    return float(apcer[idx]), float(thr[idx])


def build_model(name, num_classes=1):
    return timm.create_model(name, pretrained=True, num_classes=num_classes)


def get_model_config(name):
    return timm.data.resolve_data_config({}, model=timm.create_model(name, pretrained=False))


def run_training(model_name, data_root, train_df, val_df, img_size, lr, weight_decay, epochs,
                  batch_size, mixup_alpha, cutmix_prob, label_smoothing, use_ema,
                  device, num_workers=0, max_seconds=None, verbose=True):
    """Trains one (model, hyperparameter) configuration; returns metrics + best weights.

    max_seconds is a wall-clock cap enforced mid-epoch so a single slow config can't
    blow the overall search/final-phase time budget on shared/variable Kaggle hardware.
    """
    t0 = time.time()
    cfg = get_model_config(model_name)
    mean, std = cfg["mean"], cfg["std"]
    train_tf = build_transforms(img_size, mean, std, train=True)
    eval_tf = build_transforms(img_size, mean, std, train=False)

    train_ds = DocDataset(train_df, data_root, train_tf)
    val_ds = DocDataset(val_df, data_root, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    model = build_model(model_name).to(device)
    pos = (train_df["label"] == 1).sum()
    neg = (train_df["label"] == 0).sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

    def criterion(logits, targets):
        if label_smoothing > 0:
            targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, epochs * len(train_loader))
    warmup = max(5, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model) if use_ema else None

    best_auc, best_apcer, best_state = 0.0, 1.0, None
    scores, ys = np.array([]), np.array([])
    timed_out = False

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            if max_seconds and (time.time() - t0) > max_seconds:
                timed_out = True
                if verbose:
                    print(f"  [{model_name}] time budget hit mid-epoch {epoch}, stopping")
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            r = np.random.rand()
            use_mix = mixup_alpha > 0 and r < 0.5
            use_cut = (not use_mix) and cutmix_prob > 0 and r < 0.5 + cutmix_prob
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                if use_mix:
                    xm, ya, yb, lam = mixup_data(x, y, mixup_alpha)
                    logits = model(xm).squeeze(1)
                    loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
                elif use_cut:
                    xm, ya, yb, lam = cutmix_data(x, y, 1.0)
                    logits = model(xm).squeeze(1)
                    loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
                else:
                    logits = model(x).squeeze(1)
                    loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema:
                ema.update(model)

        eval_state = ema.shadow if ema else model.state_dict()
        eval_model = model
        if ema:
            eval_model = build_model(model_name).to(device)
            eval_model.load_state_dict(eval_state)
        eval_model.eval()
        batch_scores, batch_ys = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = eval_model(x).squeeze(1)
                batch_scores.append(torch.sigmoid(logits).float().cpu().numpy())
                batch_ys.append(y.numpy())
        scores = np.concatenate(batch_scores)
        ys = np.concatenate(batch_ys)
        auc = roc_auc_score(ys, scores) if len(np.unique(ys)) > 1 else float("nan")
        apcer, _ = apcer_at_bpcer(ys, scores, 0.01)
        if verbose:
            print(f"  [{model_name}] epoch {epoch} auc={auc:.4f} apcer@1%bpcer={apcer:.4f} ({time.time()-t0:.0f}s)")
        if auc > best_auc:
            best_auc, best_apcer = auc, apcer
            best_state = {k: v.detach().cpu().clone() for k, v in eval_state.items()}
        if timed_out:
            break

    return {
        "model_name": model_name, "img_size": img_size, "lr": lr, "weight_decay": weight_decay,
        "mixup_alpha": mixup_alpha, "cutmix_prob": cutmix_prob, "label_smoothing": label_smoothing,
        "use_ema": use_ema, "epochs_requested": epochs,
        "best_val_auc": best_auc, "best_val_apcer_at_1pct_bpcer": best_apcer,
        "elapsed_sec": time.time() - t0,
        "state_dict": best_state,
        "val_scores": scores, "val_ys": ys,
    }


def train_fold_resumable(model_name, data_root, train_df, val_df, img_size, fold, epochs, batch_size,
                          hp, device, ckpt_path, pre_transform=None, num_workers=4, use_sample_weight=False):
    """Trains one (model, fold) with per-EPOCH checkpointing to ckpt_path, so a kill
    mid-fold resumes from the last completed epoch instead of restarting the whole
    fold or the whole model. Shared by train_stacking_local.py (plain RGB CNNs) and
    train_residual_cnn.py (residual/artifact-transformed input) so both get the same
    resumability guarantees.

    hp: dict with keys lr, weight_decay, mixup_alpha, cutmix_prob, label_smoothing.
    pre_transform: optional callable(PIL.Image) -> PIL.Image applied before the
    normal resize/augment/normalize chain (e.g. a ResidualTransform).
    use_sample_weight: if True, train_df must have a 'sample_weight' column (real
    labels default to 1.0; pseudo-labels from pseudo_label_by_agreement.py carry a
    lower weight) and the loss is weighted per-sample. Validation is never weighted
    -- val_df is assumed to be real ground-truth labels only.
    """
    cfg = get_model_config(model_name)
    mean, std = cfg["mean"], cfg["std"]
    train_tf = build_transforms(img_size, mean, std, train=True)
    eval_tf = build_transforms(img_size, mean, std, train=False)
    if pre_transform is not None:
        # Resize BEFORE the (potentially expensive, e.g. gaussian-blur or JPEG-recompress)
        # pre_transform -- doing it on the full-resolution original (up to ~1600x1000px)
        # instead of the model's actual input size is ~20-30x more pixels than needed and
        # was observed to make training prohibitively slow.
        presize = transforms.Resize((img_size, img_size))
        train_tf = transforms.Compose([presize, pre_transform, train_tf])
        eval_tf = transforms.Compose([presize, pre_transform, eval_tf])

    train_ds = DocDataset(train_df, data_root, train_tf, return_weight=use_sample_weight)
    val_ds = DocDataset(val_df, data_root, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    model = build_model(model_name).to(device)
    pos = (train_df["label"] == 1).sum()
    neg = (train_df["label"] == 0).sum()
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

    def criterion(logits, targets, sample_weight=None):
        if hp["label_smoothing"] > 0:
            targets = targets * (1 - hp["label_smoothing"]) + 0.5 * hp["label_smoothing"]
        if sample_weight is None:
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        per_example = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
        return (per_example * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    total_steps = max(1, epochs * len(train_loader))
    warmup = max(5, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    start_epoch = 0
    best_auc, best_apcer, best_state = 0.0, 1.0, None
    ema = EMA(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scaler.load_state_dict(ck["scaler_state"])
        ema.shadow = ck["ema_shadow"]
        start_epoch = ck["epoch_done"] + 1
        best_auc, best_apcer, best_state = ck["best_auc"], ck["best_apcer"], ck["best_state"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=start_epoch * len(train_loader) - 1)
        print(f"  [{model_name}] fold {fold}: resuming from epoch {start_epoch} "
              f"(best_auc so far={best_auc:.4f})")

    scores, ys = None, None
    for epoch in range(start_epoch, epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            if use_sample_weight:
                x, y, w = batch
                w = w.to(device, non_blocking=True)
            else:
                x, y = batch
                w = None
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # Mixup/cutmix mix two images/labels together, which doesn't compose cleanly
            # with per-sample weighting (a mixed pseudo-label+real pair would need its
            # own interpolated weight semantics) -- keep the weighted path to plain
            # weighted BCE, matching the common practice of using lighter augmentation
            # on pseudo-labeled data.
            r = np.random.rand()
            use_mix = (not use_sample_weight) and hp["mixup_alpha"] > 0 and r < 0.5
            use_cut = (not use_sample_weight) and (not use_mix) and hp["cutmix_prob"] > 0 and r < 0.5 + hp["cutmix_prob"]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                if use_mix:
                    xm, ya, yb, lam = mixup_data(x, y, hp["mixup_alpha"])
                    logits = model(xm).squeeze(1)
                    loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
                elif use_cut:
                    xm, ya, yb, lam = cutmix_data(x, y, 1.0)
                    logits = model(xm).squeeze(1)
                    loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
                else:
                    logits = model(x).squeeze(1)
                    loss = criterion(logits, y, sample_weight=w)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            if i % 200 == 0:
                print(f"  [{model_name}] fold {fold} epoch {epoch} step {i}/{len(train_loader)} loss {loss.item():.4f}")

        eval_model = build_model(model_name).to(device)
        eval_model.load_state_dict(ema.shadow)
        eval_model.eval()
        scores, ys = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = eval_model(x).squeeze(1)
                scores.append(torch.sigmoid(logits).float().cpu().numpy())
                ys.append(y.numpy())
        scores = np.concatenate(scores)
        ys = np.concatenate(ys)
        auc = roc_auc_score(ys, scores)
        apcer, _ = apcer_at_bpcer(ys, scores, 0.01)
        print(f"  [{model_name}] fold {fold} epoch {epoch} done: auc={auc:.4f} apcer@1%bpcer={apcer:.4f}")
        if auc > best_auc:
            best_auc, best_apcer = auc, apcer
            best_state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
        del eval_model
        torch.cuda.empty_cache()

        torch.save({
            "epoch_done": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "ema_shadow": ema.shadow,
            "best_auc": best_auc,
            "best_apcer": best_apcer,
            "best_state": best_state,
        }, ckpt_path)

    if scores is None:
        # Edge case: resumed a fold whose in-progress checkpoint already covered every
        # epoch (kill landed between the last epoch's checkpoint save and fold completion)
        # -- recompute val scores from best_state rather than leaving them undefined.
        eval_model = build_model(model_name).to(device)
        eval_model.load_state_dict(best_state)
        eval_model.eval()
        batch_scores, batch_ys = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = eval_model(x).squeeze(1)
                batch_scores.append(torch.sigmoid(logits).float().cpu().numpy())
                batch_ys.append(y.numpy())
        scores = np.concatenate(batch_scores)
        ys = np.concatenate(batch_ys)
        del eval_model
        torch.cuda.empty_cache()

    return {"best_val_auc": best_auc, "best_val_apcer_at_1pct_bpcer": best_apcer,
            "state_dict": best_state, "val_scores": scores, "val_ys": ys}


def stratified_sample(df, key_col, frac=None, n_per_group=None, seed=42):
    parts = []
    for _, g in df.groupby(key_col):
        n = max(1, int(round(len(g) * frac))) if frac is not None else min(len(g), n_per_group)
        parts.append(g.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    # Local smoke test: tiny subset, tiny time budget, 1 model — just verifying no crashes.
    from sklearn.model_selection import train_test_split

    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    labels = pd.read_csv(DATA / "train_labels.csv")
    labels["strat_key"] = labels["label"].astype(str) + "_" + labels["type"]
    small = stratified_sample(labels, "strat_key", n_per_group=20, seed=42)
    train_df, val_df = train_test_split(small, test_size=0.3, stratify=small["strat_key"], random_state=42)
    print("smoke test sizes:", len(train_df), len(val_df))

    res = run_training(
        "resnet50", DATA, train_df, val_df, img_size=160, lr=1e-4, weight_decay=1e-4,
        epochs=1, batch_size=8, mixup_alpha=0.2, cutmix_prob=0.2, label_smoothing=0.05,
        use_ema=True, device=device, num_workers=0, max_seconds=120, verbose=True,
    )
    print({k: v for k, v in res.items() if k not in ("state_dict", "val_scores", "val_ys")})
    print("SMOKE TEST PASSED")
