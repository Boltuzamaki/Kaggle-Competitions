import os

import onnx


PROJECT_DIR = os.environ.get("PROJECT_DIR", os.getcwd())
model = onnx.load(
    os.path.join(
        PROJECT_DIR,
        "other_model_onnx",
        "task281_compact_SHm_ev,SWm_ev.onnx",
    )
)
