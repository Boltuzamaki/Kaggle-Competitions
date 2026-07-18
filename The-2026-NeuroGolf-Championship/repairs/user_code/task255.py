import os
import onnx

PROJECT_DIR = os.environ.get("PROJECT_DIR", os.getcwd())
model = onnx.load(os.path.join(PROJECT_DIR, "baseline_v22", "task255.onnx"))
