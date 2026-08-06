"""Leave-one-well-out full interpreted-curve transfer experiment.

This tests the specific hypothesis from the competition discussion: nearby
wells sharing a typewell/master frame and drilling direction may share the
shape of their interpreted TVT residual curve.  No target from the query well's
hidden tail is used to choose or align a neighbor.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DATA = Path("data")


def frame_id(well: str) -> str:
    tw = pd.read_csv(DATA / "train" / f"{well}__typewell.csv", usecols=["TVT", "GR"])
    tw = tw.sort_values("TVT")
    gr = tw.GR.fillna(tw.GR.mean())
    # Match the 57-master-frame grouping used by Stack V4. The curves can have
    # small row-level differences, so hashing the full sampled trace is too
    # strict (it incorrectly produces ~750 singleton frames).
    return "|".join(
        [
            f"{float(tw.TVT.iloc[0]):.1f}",
            f"{float(tw.TVT.iloc[-1]):.1f}",
            f"{float(np.nansum(gr.iloc[:200])):.0f}",
            str(len(tw)),
        ]
    )


def load_well(path: str) -> tuple[dict, pd.DataFrame] | None:
    well = os.path.basename(path).split("__")[0]
    h = pd.read_csv(path, usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input"])
    ps = int(h.TVT_input.notna().sum())
    if ps < 60 or ps >= len(h) - 3:
        return None
    anchor = float(h.TVT.iloc[ps - 1])
    md0 = float(h.MD.iloc[ps - 1])
    dx = float(h.X.iloc[-1] - h.X.iloc[ps - 1])
    dy = float(h.Y.iloc[-1] - h.Y.iloc[ps - 1])
    az = float(np.arctan2(dy, dx))
    direction = int(np.floor(((az + np.pi) % (2 * np.pi)) / (np.pi / 2)))
    meta = {
        "well": well,
        "frame": frame_id(well),
        "x": float(h.X.iloc[ps - 1]),
        "y": float(h.Y.iloc[ps - 1]),
        "az": az,
        "direction": direction,
        "anchor": anchor,
    }
    curve = pd.DataFrame(
        {
            "well": well,
            "d_md": h.MD.iloc[ps:].to_numpy(float) - md0,
            "target": h.TVT.iloc[ps:].to_numpy(float) - anchor,
        }
    )
    return meta, curve


loaded = [
    item
    for item in (
        load_well(p)
        for p in sorted(glob.glob(str(DATA / "train" / "*__horizontal_well.csv")))
    )
    if item is not None
]
meta = pd.DataFrame([x[0] for x in loaded])
curves = {x[0]["well"]: x[1] for x in loaded}
print("wells", len(meta), "frames", meta.frame.nunique(), flush=True)


def angular_distance(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (a - b))))


rows = []
for qi, q in meta.iterrows():
    candidates = meta.index[meta.index != qi].to_numpy()
    same_frame = meta.loc[candidates, "frame"].to_numpy() == q.frame
    same_direction = (
        angular_distance(meta.loc[candidates, "az"].to_numpy(float), float(q.az))
        <= np.deg2rad(35)
    )
    allowed = candidates[same_frame & same_direction]
    if not len(allowed):
        allowed = candidates[same_frame]
    if not len(allowed):
        # Conservative fallback: no cross-frame curve transfer.
        pred = np.zeros(len(curves[q.well]))
        distance = np.inf
        neighbor = ""
    else:
        dd = np.hypot(
            meta.loc[allowed, "x"].to_numpy(float) - float(q.x),
            meta.loc[allowed, "y"].to_numpy(float) - float(q.y),
        )
        ni = allowed[int(np.argmin(dd))]
        distance = float(dd.min())
        neighbor = str(meta.loc[ni, "well"])
        source = curves[neighbor]
        query = curves[q.well]
        pred = np.interp(
            query.d_md.to_numpy(float),
            source.d_md.to_numpy(float),
            source.target.to_numpy(float),
            left=0.0,
            right=float(source.target.iloc[-1]),
        )
    actual = curves[q.well].target.to_numpy(float)
    for gate in [75, 150, 300, 600, 1200, np.inf]:
        gated = pred if distance <= gate else np.zeros_like(pred)
        rows.append(
            {
                "well": q.well,
                "neighbor": neighbor,
                "distance": distance,
                "gate": gate,
                "n": len(actual),
                "sse": float(np.sum((actual - gated) ** 2)),
                "constant_sse": float(np.sum(actual**2)),
            }
        )

report = pd.DataFrame(rows)
summary = (
    report.groupby("gate", dropna=False)
    .agg(rows=("n", "sum"), sse=("sse", "sum"), constant_sse=("constant_sse", "sum"))
    .reset_index()
)
summary["rmse"] = np.sqrt(summary.sse / summary.rows)
summary["constant_rmse"] = np.sqrt(summary.constant_sse / summary.rows)
summary.to_csv("full_curve_neighbor_summary.csv", index=False)
report.to_csv("full_curve_neighbor_wells.csv", index=False)
print(summary.to_string(index=False), flush=True)
