"""Verified task057 dynamic-Slice repair."""

import os

import onnx


PROJECT_DIR = os.environ.get("PROJECT_DIR") or os.getcwd()
model = onnx.load(os.path.join(PROJECT_DIR, "repairs", "task057.onnx"))
