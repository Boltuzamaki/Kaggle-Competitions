from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V12 = ROOT / "kernels/experiments/ndr_v11_v12_trajectory_anchor/output_v1/sub_M1_center.csv"
V14 = ROOT / "forensics/kaggle_ndr_v14_external_retain/audit_files_v3/ndr_v14/per_box_diagnostics.csv"


def parse_submission(path: Path) -> pd.DataFrame:
    submission = pd.read_csv(path, dtype={"image_id": str})
    rows = []
    for row in submission.itertuples(index=False):
        value = str(row.prediction_string).strip()
        if not value or value == "nan":
            continue
        boxes = np.asarray(list(map(float, value.split())), dtype=np.float32).reshape(-1, 5)
        for confidence, x, y, width, height in boxes:
            rows.append((str(row.image_id), confidence, x, y, x + width, y + height))
    return pd.DataFrame(rows, columns=["image_id", "m1", "x1", "y1", "x2", "y2"])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-9)


def main() -> None:
    v12 = parse_submission(V12)
    v14 = pd.read_csv(V14, dtype={"image_id": str})
    matches = []
    for image_id, candidates in v12.groupby("image_id", sort=False):
        diagnostics = v14[v14.image_id == image_id]
        a = candidates[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        b = diagnostics[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        ious = iou_matrix(a, b)
        nearest = ious.argmax(axis=1)
        for local_index, diagnostic_index in enumerate(nearest):
            item = diagnostics.iloc[int(diagnostic_index)]
            matches.append(
                (
                    candidates.index[local_index],
                    float(ious[local_index, diagnostic_index]),
                    float(item.pcgrad),
                    float(item["rank"]),
                    float(item.original),
                )
            )
    aligned = pd.DataFrame(matches, columns=["index", "iou", "pcgrad", "rank", "original"]).set_index("index")
    result = v12.join(aligned)
    print(f"V12 boxes: {len(v12):,}; V14 boxes: {len(v14):,}")
    print(result[["m1", "iou", "pcgrad", "rank", "original"]].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))
    print(result.groupby("m1").agg(n=("m1", "size"), pcgrad=("pcgrad", "mean"), rank=("rank", "mean"), iou=("iou", "mean"), original=("original", "mean")))
    result["consensus"] = 0.5 * result.pcgrad + 0.5 * result["rank"]
    eligible = result.m1 >= 0.21
    for name, mask in {
        "unanimous_99_95": eligible & (result["rank"] >= 0.99) & (result.pcgrad >= 0.95),
        "unanimous_95_90": eligible & (result["rank"] >= 0.95) & (result.pcgrad >= 0.90),
        "pcgrad_95": eligible & (result.pcgrad >= 0.95),
        "pcgrad_90": eligible & (result.pcgrad >= 0.90),
        "pcgrad_80": eligible & (result.pcgrad >= 0.80),
        "consensus_95": eligible & (result.consensus >= 0.95),
        "consensus_90": eligible & (result.consensus >= 0.90),
        "consensus_80": eligible & (result.consensus >= 0.80),
    }.items():
        selected = result[mask]
        print(
            name,
            {"boxes": len(selected), "mass": float(selected.m1.sum()), "mean_m1": float(selected.m1.mean()) if len(selected) else 0.0},
        )


if __name__ == "__main__":
    main()
