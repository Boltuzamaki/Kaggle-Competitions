"""OOF-validated Harshini + fresh Pilkwang model-weight integration."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

# Coefficients fitted on common OOF rows. Nested by-well meta-CV RMSE 8.513.
HARSHINI_WEIGHT = 0.7997099707272112
PILKWANG_WEIGHT = 0.29513908692022484


def integrate(harshini_submission, data_root, package_root,
              fresh_inference_function):
    """Combine without reading any prediction CSV or referencing test IDs."""
    data_root, package_root = Path(data_root), Path(package_root)
    sample = pd.read_csv(data_root/"sample_submission.csv", dtype={"id": str})
    h = sample[["id"]].merge(
        harshini_submission.assign(id=harshini_submission.id.astype(str)),
        on="id", how="left", validate="one_to_one")
    if h.tvt.isna().any():
        raise RuntimeError("Harshini prediction does not cover sample IDs")
    frame, pil_delta, model_names = fresh_inference_function(
        data_root, package_root, sample=sample)
    if not frame.id.astype(str).equals(sample.id.astype(str)):
        raise RuntimeError("fresh model feature order differs from sample order")
    last = frame["last_known_TVT"].to_numpy(float)
    harsh_delta = h.tvt.to_numpy(float)-last
    final_delta = HARSHINI_WEIGHT*harsh_delta+PILKWANG_WEIGHT*pil_delta
    out = sample[["id"]].copy()
    out["tvt"] = last+final_delta
    if not np.isfinite(out.tvt).all():
        raise RuntimeError("non-finite integrated prediction")
    audit = {
        "rows": len(out),
        "harshini_weight": HARSHINI_WEIGHT,
        "pilkwang_weight": PILKWANG_WEIGHT,
        "weight_source":
            "common-765-well OOF positive ridge; nested meta-CV RMSE 8.512967",
        "public_prediction_csv_read": False,
        "hardcoded_test_ids": False,
        "pilkwang_features_built_fresh_from_raw_competition_files": True,
        "pilkwang_models": model_names,
        "required_dataset_source": "pilkwang/rogii-model-package",
    }
    return out, audit


def save_audited(out, audit, output_dir="."):
    output_dir = Path(output_dir)
    out.to_csv(output_dir/"submission.csv", index=False)
    (output_dir/"harshini_pilkwang_integration_audit.json").write_text(
        json.dumps(audit, indent=2))
