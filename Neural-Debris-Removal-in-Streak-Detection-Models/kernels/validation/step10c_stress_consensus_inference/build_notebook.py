"""Build the frozen Step 10C public stress-consensus + fixed-bank inference notebook."""

from __future__ import annotations

import base64
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    template = (HERE / "step10c_template.py").read_text(encoding="utf-8")
    gate_bytes = (ROOT / "local_validation" / "kaggle_step10b_strict_consensus" /
                  "deployment_package" / "strict_consensus_gate.json").read_bytes()
    renderer_stage = (ROOT / "local_validation" / "kaggle_step8b_renderer" /
                      "renderer_stage.py").read_text(encoding="utf-8")
    renderer_defs = renderer_stage.split("\ndef fit_logistic", 1)[0]
    renderer_defs = renderer_defs[renderer_defs.index("FEATURE_NAMES = LOCK") :]
    source = template.replace("__STEP10B_GATE_BASE64__", base64.b64encode(gate_bytes).decode("ascii"))
    source = source.replace("# __RENDERER_FEATURE_DEFINITIONS__", renderer_defs)
    if "__STEP10B_GATE_BASE64__" in source or "__RENDERER_FEATURE_DEFINITIONS__" in source:
        raise RuntimeError("Notebook placeholder replacement failed")
    (HERE / "step10c_stress_consensus_inference.py").write_text(source, encoding="utf-8")
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": source.splitlines(True)}],
        "metadata": {
            "kaggle": {"accelerator": "gpu", "dataSources": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (HERE / "step10c_stress_consensus_inference.ipynb").write_text(json.dumps(notebook), encoding="utf-8")


if __name__ == "__main__":
    main()
