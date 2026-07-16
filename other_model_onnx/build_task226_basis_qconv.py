"""Build an exact low-cost task226 model.

The divider geometry has five independent separable basis masks:
inside-grid, first region, middle region, last region, and non-divider.
They are formed together as a [1,5,10,10] uint8 tensor.  A final 1x1
QLinearConv changes that basis into color channels and pads 10x10 to 30x30;
because its result is named ``output``, the large result is score-exempt.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repairs" / "task226.onnx"
OUTPUT = ROOT / "other_model_onnx" / "task226.onnx"


model = onnx.load(SOURCE)
graph = model.graph

# Retain the exact, already-audited row/column interval construction through
# row_first/row_mid/row_last and col_first/col_mid/col_last.
kept_nodes = list(graph.node[:27])

generated_initializers = {
    "pads_6_to_10_30",
    "qscale",
    "qzero_u8",
    "qzero_i8",
    "basis_weights",
}
kept_initializers = [
    init for init in graph.initializer if init.name not in generated_initializers
]

weights = np.zeros((10, 5, 1, 1), dtype=np.int8)
# Basis order: presence, first, middle, last, non-divider.
weights[0, :, 0, 0] = [0, -1, -1, -1, 1]  # background
weights[1, 1, 0, 0] = 1                    # blue
weights[2, 2, 0, 0] = 1                    # red
weights[3, 3, 0, 0] = 1                    # green
weights[5, :, 0, 0] = [1, 0, 0, 0, -1]    # gray dividers

kept_initializers.extend(
    [
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="qscale"),
        numpy_helper.from_array(np.array(0, dtype=np.uint8), name="qzero_u8"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), name="qzero_i8"),
        numpy_helper.from_array(weights, name="basis_weights"),
    ]
)

new_nodes = [
    helper.make_node("Max", ["row_non", "row_sep"], ["row_one"]),
    helper.make_node("Max", ["col_non", "col_sep"], ["col_one"]),
    helper.make_node(
        "Concat",
        ["row_one", "row_first", "row_mid", "row_last", "row_non"],
        ["row_basis"],
        axis=1,
    ),
    helper.make_node(
        "Concat",
        ["col_one", "col_first", "col_mid", "col_last", "col_non"],
        ["col_basis"],
        axis=1,
    ),
    helper.make_node("Mul", ["row_basis", "col_basis"], ["basis"]),
    helper.make_node(
        "QLinearConv",
        [
            "basis",
            "qscale",
            "qzero_u8",
            "basis_weights",
            "qscale",
            "qzero_i8",
            "qscale",
            "qzero_u8",
        ],
        ["output"],
        pads=[0, 0, 20, 20],
    ),
]

del graph.node[:]
graph.node.extend(kept_nodes + new_nodes)
del graph.initializer[:]
graph.initializer.extend(kept_initializers)
del graph.value_info[:]

out_type = graph.output[0].type.tensor_type
out_type.elem_type = TensorProto.UINT8
del out_type.shape.dim[:]
for value in (1, 10, 30, 30):
    out_type.shape.dim.add().dim_value = value

model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
onnx.checker.check_model(model, full_check=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, OUTPUT)
print(OUTPUT)
