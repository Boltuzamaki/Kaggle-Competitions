"""Build the Step 8C frozen renderer robustness notebook.

The expensive public-control reconstruction is reused verbatim from the audited
Step 8B V3 notebook.  Step 8C stops before any repair-checkpoint inference,
loads the frozen Step 8B renderer models by SHA256, and only measures their
response to transformations declared before sample enumeration.
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "kaggle_step8b_renderer" / "step8b_renderer.ipynb"


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][0]["source"])
    prefix = source.split("\ndef infer_checkpoint", 1)[0]
    prefix = prefix.replace(
        'OUT = Path("/kaggle/working/step8b_renderer")',
        'OUT = Path("/kaggle/working/step8c_renderer_robustness")',
    )
    lock_write = '(OUT / "selection_lock.json").write_text(json.dumps(LOCK, indent=2), encoding="utf-8")'
    stress_lock = r'''
STRESS_TRANSFORMS = {
    "identity": {},
    "gain_090": {"gain": 0.90},
    "gain_110": {"gain": 1.10},
    "gamma_090": {"gamma": 0.90},
    "gamma_110": {"gamma": 1.10},
    "blur_sigma065": {"blur_sigma": 0.65},
    "sensor_noise_sigma15": {"noise_sigma": 1.5},
    "downup_075": {"downup_scale": 0.75},
    "quantize_8bit": {"quantize_8bit": True},
}
ROBUSTNESS_GATE = {
    "identity_aggregate_auc_min": 0.99,
    "each_shift_aggregate_auc_min": 0.90,
    "each_shift_each_family_auc_min": 0.85,
    "each_shift_median_poison_recall_min": 0.80,
    "each_shift_maximum_family_fpr_max": 0.20,
    "each_shift_rank_correlation_min": 0.75,
    "each_shift_margin_retention_min": 0.65,
}
EXPECTED_STEP8B_MANIFEST_SHA256 = "f2906451a2b6326bf9c51f4d85f1eda175a86a2e11dadd3d32fffbfb977284b7"
EXPECTED_RENDERER_MODELS_SHA256 = "ded18b7c482e6a423335d86b3f867a3cf58d083942a9ed51fd1dedbba4e92a91"
LOCK["experiment"] = "STEP8C_FROZEN_RENDERER_ROBUSTNESS"
LOCK["parent_step"] = "Step 8B V3"
LOCK["stress_transforms"] = STRESS_TRANSFORMS
LOCK["robustness_gate"] = ROBUSTNESS_GATE
LOCK["expected_step8b_manifest_sha256"] = EXPECTED_STEP8B_MANIFEST_SHA256
LOCK["expected_renderer_models_sha256"] = EXPECTED_RENDERER_MODELS_SHA256
LOCK["training_or_retuning"] = False
LOCK["candidate_or_submission_generation"] = False
(OUT / "selection_lock.json").write_text(json.dumps(LOCK, indent=2), encoding="utf-8")
'''.strip()
    if lock_write not in prefix:
        raise RuntimeError("Could not locate Step 8B lock write")
    prefix = prefix.replace(lock_write, stress_lock, 1)

    renderer = (HERE.parent / "kaggle_step8b_renderer" / "renderer_stage.py").read_text(encoding="utf-8")
    renderer_defs = renderer.split("\ndef fit_logistic", 1)[0]
    renderer_defs = renderer_defs[renderer_defs.index("FEATURE_NAMES = LOCK") :]
    stress = (HERE / "stress_stage.py").read_text(encoding="utf-8")
    combined = prefix.rstrip() + "\n\n" + renderer_defs.rstrip() + "\n\n" + stress.rstrip() + "\n"

    (HERE / "step8c_renderer_robustness.py").write_text(combined, encoding="utf-8")
    output = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": combined.splitlines(True)}],
        "metadata": {
            "kaggle": {"accelerator": "gpu", "dataSources": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (HERE / "step8c_renderer_robustness.ipynb").write_text(json.dumps(output), encoding="utf-8")


if __name__ == "__main__":
    main()
