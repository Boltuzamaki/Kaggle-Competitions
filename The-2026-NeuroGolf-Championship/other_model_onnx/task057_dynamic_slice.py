"""Promotion wrapper for the verified task057 dynamic-Slice candidate."""

import os

import onnx


PROJECT_DIR = os.environ.get("PROJECT_DIR") or os.getcwd()
model = onnx.load(
    os.path.join(PROJECT_DIR, "other_model_onnx", "task057_dynamic_slice.onnx")
)
