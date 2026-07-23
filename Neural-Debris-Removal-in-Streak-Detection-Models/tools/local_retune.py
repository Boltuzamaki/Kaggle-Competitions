"""Regenerate a submission CSV locally from the V10 per-box diagnostics npz.

After the `ndr_v10_ensemble` Kaggle run completes, download
`ndr_v10/per_box_diagnostics.npz` and point this script at it. Every
post-processing parameter (weights, P_HI/P_LO, MIN_KEEP, rescue rule) can then
be retuned on CPU in seconds - no further GPU run is needed.

Example:
    .venv/Scripts/python.exe tools/local_retune.py \
        --npz path/to/per_box_diagnostics.npz \
        --sample data/sample_submission.csv \
        --out submission_retuned.csv \
        --w-diff 0.85 --w-geo 0.05 --w-dash 0.10 --p-hi 0.55 --p-lo 0.25
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

IMG_W = IMG_H = 1024
EPS_DEMOTE = 0.01


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    area_a = np.maximum(0, ax2 - ax1) * np.maximum(0, ay2 - ay1)
    area_b = np.maximum(0, bx2 - bx1) * np.maximum(0, by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def format_preds(bx, sc):
    parts = []
    for (x1, y1, x2, y2), s in zip(bx, sc):
        x1, y1 = float(np.clip(x1, 0, IMG_W)), float(np.clip(y1, 0, IMG_H))
        x2, y2 = float(np.clip(x2, 0, IMG_W)), float(np.clip(y2, 0, IMG_H))
        w, h = x2 - x1, y2 - y1
        if w > 0 and h > 0 and s > 0:
            parts += [f"{s:.6f}", f"{x1:.2f}", f"{y1:.2f}", f"{w:.2f}", f"{h:.2f}"]
    return " ".join(parts) or " "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-diff", type=float, default=0.90)
    ap.add_argument("--w-geo", type=float, default=0.10)
    ap.add_argument("--w-dash", type=float, default=0.00)
    ap.add_argument("--min-keep", type=float, default=0.20)
    ap.add_argument("--p-hi", type=float, default=0.55)
    ap.add_argument("--p-lo", type=float, default=0.25)
    ap.add_argument("--rescue", action="store_true",
                    help="cap p_poison at p_lo for continuous linear candidates")
    ap.add_argument("--rescue-dash-max", type=float, default=0.25)
    ap.add_argument("--rescue-lin-min", type=float, default=0.92)
    ap.add_argument("--replicas", type=str, default="",
                    help="comma-separated replica column indices to average (default: all)")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    stems = [str(s) for s in z["stems"]]
    counts = z["counts"]
    offsets = np.concatenate([[0], np.cumsum(counts)])
    s_diff = z["s_diff"]
    dash_enabled = bool(z["dash_enabled"]) if "dash_enabled" in z.files else False
    if args.replicas:
        cols = [int(c) for c in args.replicas.split(",")]
        if not cols or min(cols) < 0 or max(cols) >= s_diff.shape[1]:
            raise ValueError(
                f"--replicas must index available columns 0..{s_diff.shape[1] - 1}"
            )
        s_diff = s_diff[:, cols]

    effective_w_dash = args.w_dash if dash_enabled else 0.0
    effective_w_diff = args.w_diff + (args.w_dash - effective_w_dash)
    effective_rescue = args.rescue and dash_enabled
    if (args.w_dash or args.rescue) and not dash_enabled:
        print(
            "dash gate did not pass in the Kaggle run; forcing dash weight to 0 "
            "and disabling rescue"
        )

    sample = pd.read_csv(args.sample, dtype={"image_id": str})
    preds = {}
    for idx, stem in enumerate(stems):
        lo, hi = offsets[idx], offsets[idx + 1]
        boxes = z["boxes"][lo:hi]
        scores = z["scores"][lo:hi]
        if len(boxes) == 0:
            preds[stem] = (boxes, scores)
            continue
        p = (effective_w_diff * s_diff[lo:hi].mean(1)
             + args.w_geo * z["s_geo"][lo:hi]
             + effective_w_dash * z["dash"][lo:hi])
        if effective_rescue:
            rescued = ((z["dash"][lo:hi] <= args.rescue_dash_max)
                       & (z["lin"][lo:hi] >= args.rescue_lin_min))
            p = np.where(rescued, np.minimum(p, args.p_lo), p)
        new_conf = np.zeros(len(scores), dtype=np.float32)
        for i, (s, pi) in enumerate(zip(scores, p)):
            if s < args.min_keep:
                new_conf[i] = 0.0
            elif pi >= args.p_hi:
                new_conf[i] = EPS_DEMOTE
            elif pi <= args.p_lo:
                new_conf[i] = float(s)
            else:
                frac = (pi - args.p_lo) / max(args.p_hi - args.p_lo, 1e-6)
                new_conf[i] = float(max(EPS_DEMOTE, s * (1 - frac)))
        keep = new_conf > 0.0
        eps_ids = np.where(new_conf <= EPS_DEMOTE + 1e-6)[0]
        strong_ids = np.where(new_conf > 0.20)[0]
        if len(eps_ids) and len(strong_ids):
            overl = iou_matrix(boxes[eps_ids], boxes[strong_ids]).max(1)
            keep[eps_ids[overl >= 0.20]] = False
        preds[stem] = (boxes[keep], new_conf[keep])

    df = sample.copy()
    df["prediction_string"] = df["image_id"].map(
        lambda i: format_preds(*preds.get(str(i), (np.zeros((0, 4)), np.zeros(0))))
    )
    if len(df) != 2000 or not df["image_id"].astype(str).is_unique:
        raise ValueError("sample submission must contain 2000 unique image IDs")
    if df["prediction_string"].isna().any():
        raise ValueError("submission contains missing prediction strings")
    df.to_csv(args.out, index=False)
    total = sum(len(s) for _, s in preds.values())
    over02 = sum(int((s > 0.20).sum()) for _, s in preds.values())
    print(f"wrote {args.out}: {len(df)} rows, {total} boxes ({over02} with conf > 0.20)")


if __name__ == "__main__":
    main()
