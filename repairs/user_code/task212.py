"""Public submission ONNX promoted after local NeuroGolf audit."""
import os
import onnx

PROJECT_DIR = os.environ.get("PROJECT_DIR") or os.getcwd()
model = onnx.load(os.path.join(PROJECT_DIR, "repairs", "task212.onnx"))
