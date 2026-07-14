import os

import onnx


project_dir = os.environ.get("PROJECT_DIR", os.getcwd())
model = onnx.load(
    os.path.join(
        project_dir,
        "other_model_onnx",
        "task058_geometric_rank3.onnx",
    )
)
