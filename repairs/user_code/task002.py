"""Load the exact direction-pruned task002 artifact for strict promotion."""

from pathlib import Path

import onnx


model = onnx.load(Path.cwd() / "other_model_onnx" / "task002.onnx")
