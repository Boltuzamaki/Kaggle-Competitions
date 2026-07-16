"""Build a direct, general task012 model.

Each non-background colour is transformed independently.  A colour occurring at
the centre of a plus becomes the two diagonals of a 5x5 motif; the colour on the
four arms becomes the length-five horizontal/vertical arms.  The two local
patterns are linearly separable with an 8x9 kernel.  A separate kernel handles
background and the zero-padded area.  The integer inequalities below were
verified over all 240 placements allowed by the task generator: both centres
are in rows/columns 2..9 and their Chebyshev separation is exactly six.

The depthwise Conv writes straight to the exempt ``output`` tensor, so the graph
has no charged activation tensors and no example-specific lookup table.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "other_model_onnx" / "task012.onnx"


def reflected_kernel(half_rows: list[list[int]]) -> np.ndarray:
    """Expand eight [centre, |dx|=1..4] rows to an 8x9 kernel."""
    half = np.asarray(half_rows, dtype=np.float32)
    return np.asarray(
        [[half[y, abs(x - 4)] for x in range(9)] for y in range(8)],
        dtype=np.float32,
    )


# The foreground classifier is shared by all nine non-background colours, so
# colour identity is never memorized.  Horizontal reflection is tied exactly.
FG = reflected_kernel(
    [
        [8, -6, -6, -6, 8],
        [-34, 8, 22, -12, 2],
        [0, 22, 12, -14, 0],
        [22, -8, -36, 10, 8],
        [0, 22, 12, -4, 0],
        [-22, 8, 22, -12, 0],
        [12, -10, -10, -8, 8],
        [8, 0, 0, 8, 6],
    ]
)
BG = reflected_kernel(
    [
        [-4, -3, 1, 2, 1],
        [-8, 7, 1, -2, -1],
        [14, -10, 3, 0, -1],
        [26, 10, -8, 0, 2],
        [6, -6, 7, -2, -2],
        [-5, 6, -1, -1, -1],
        [-1, -1, 1, 0, 0],
        [4, -3, -3, 2, 2],
    ]
)


weights = np.stack([BG] + [FG] * 9, axis=0)[:, None, :, :]
bias = np.array([-27] + [-21] * 9, dtype=np.float32)

graph = helper.make_graph(
    [
        helper.make_node(
            "Conv",
            ["input", "weights", "bias"],
            ["output"],
            group=10,
            pads=[3, 4, 4, 4],
            strides=[1, 1],
        )
    ],
    "task012_direct_depthwise_conv",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
    [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(bias, "bias")],
)
model = helper.make_model(
    graph,
    producer_name="task012_direct_conv",
    ir_version=8,
    opset_imports=[helper.make_opsetid("", 13)],
)
onnx.checker.check_model(model)
onnx.save(model, OUT)
print(f"saved {OUT}")
