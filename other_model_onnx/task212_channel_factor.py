"""Promotion wrapper for the verified task212 channel-factor candidate."""

import os

import onnx


PROJECT_DIR = os.environ.get("PROJECT_DIR") or os.getcwd()
model = onnx.load(
    os.path.join(PROJECT_DIR, "other_model_onnx", "task212_channel_factor.onnx")
)
