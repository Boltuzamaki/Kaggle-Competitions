"""Test the public robust structural projection on our own Stack V4 OOFs."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd


FEATURES = pd.read_pickle("r_v4b/train_feats.pkl")
OOFS = joblib.load("r_v4b/stack_v4_oofs.joblib")["oofs"]
y = FEATURES["target"].to_numpy(float)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def robust_poly(x, yv, degree=4, iterations=4):
    x = np.asarray(x, float)
    yv = np.asarray(yv, float)
    weight = np.ones(len(x))
    coef = np.polyfit(x, yv, min(degree, max(1, len(x) - 1)), w=weight)
    for _ in range(iterations):
        fit = np.polyval(coef, x)
        residual = yv - fit
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        weight = 1.0 / np.maximum(1.0, np.abs(residual) / (2.5 * scale))
        coef = np.polyfit(x, yv, min(degree, max(1, len(x) - 1)), w=weight)
    return np.polyval(coef, x)


def project(pred, blend, degree):
    out = np.asarray(pred, float).copy()
    for _, idx in FEATURES.groupby("well", sort=False).indices.items():
        idx = np.asarray(idx)
        md = FEATURES["d_md"].to_numpy(float)[idx]
        dz = FEATURES["d_z"].to_numpy(float)[idx]
        span = max(float(md.max() - md.min()), 1e-6)
        s = (md - md.min()) / span
        # Structural coordinate U = delta TVT + delta Z.
        structural = out[idx] + dz
        fitted = robust_poly(s, structural, degree=degree)
        projected = fitted - dz
        out[idx] = (1.0 - blend) * out[idx] + blend * projected
    return out


rows = []
for name, pred in OOFS.items():
    base = rmse(y, pred)
    rows.append({"model": name, "degree": 0, "blend": 0.0, "rmse": base})
    for degree in [2, 3, 4, 5]:
        for blend in [0.25, 0.50, 0.75, 1.0]:
            candidate = project(pred, blend, degree)
            rows.append(
                {
                    "model": name,
                    "degree": degree,
                    "blend": blend,
                    "rmse": rmse(y, candidate),
                }
            )

report = pd.DataFrame(rows)
report["gain"] = report.groupby("model")["rmse"].transform("first") - report["rmse"]
report.to_csv("stack_v4_projection_cv.csv", index=False)
print(
    report.sort_values(["model", "rmse"]).groupby("model", sort=False).head(5).to_string(index=False)
)
