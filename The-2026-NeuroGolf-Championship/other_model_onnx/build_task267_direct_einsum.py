"""Build the exact direct-output task267 model.

The input contains a creature in the central 5x5 area and a recolor marker at
(6, 0).  The output keeps background, recolors every non-background pixel to
the marker color, and changes the marker cell itself back to background.

One final Einsum performs those three terms directly into the score-exempt
tensor named ``output``; no full spatial or one-hot intermediate is stored.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "other_model_onnx" / "task267.onnx"


def tensor(name: str, value: np.ndarray):
    return numpy_helper.from_array(value, name=name)


# p=0 selects input background; p=1 selects every non-background channel.
input_basis = np.zeros((2, 10), dtype=np.float32)
input_basis[0, 0] = 1
input_basis[1, 1:] = 1

# s=0 is the whole padded canvas; s=1 is only marker coordinate (6, 0).
row_basis = np.ones((2, 30), dtype=np.float32)
row_basis[1] = 0
row_basis[1, 6] = 1
col_basis = np.ones((2, 30), dtype=np.float32)
col_basis[1] = 0
col_basis[1, 0] = 1

# q=0 is background color; q=1 is the dynamic marker color.
# Terms:
#   background_input * background_color * everywhere
# + nonbackground_input * marker_color * everywhere
# + nonbackground_input * background_color * marker_cell
# - nonbackground_input * marker_color * marker_cell
core = np.zeros((2, 2, 2), dtype=np.float32)
core[0, 0, 0] = 1
core[1, 1, 0] = 1
core[1, 0, 1] = 1
core[1, 1, 1] = -1

background = np.zeros((1, 10, 1, 1), dtype=np.float32)
background[0, 0, 0, 0] = 1

initializers = [
    tensor("marker_starts", np.array([6, 0], dtype=np.int64)),
    tensor("marker_ends", np.array([7, 1], dtype=np.int64)),
    tensor("axes_hw", np.array([2, 3], dtype=np.int64)),
    tensor("background", background),
    tensor("input_basis", input_basis),
    tensor("row_basis", row_basis),
    tensor("col_basis", col_basis),
    tensor("core", core),
]

nodes = [
    helper.make_node(
        "Slice",
        ["input", "marker_starts", "marker_ends", "axes_hw"],
        ["marker"],
    ),
    # Concatenating on the fixed batch-size-one axis creates q=[background,
    # marker] without an Unsqueeze/Reshape intermediate.
    helper.make_node(
        "Concat", ["background", "marker"], ["color_basis"], axis=0
    ),
    helper.make_node(
        "Einsum",
        ["input", "color_basis", "input_basis", "row_basis", "col_basis", "core"],
        ["output"],
        equation="nahw,qcuv,pa,sh,sw,pqs->nchw",
    ),
]

graph = helper.make_graph(
    nodes,
    "task267_direct_einsum",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
    initializers,
)
model = helper.make_model(
    graph, ir_version=8, opset_imports=[helper.make_opsetid("", 12)]
)
model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
onnx.checker.check_model(model, full_check=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, OUTPUT)
print(OUTPUT)
