"""Rebuild the strongest standards-compliant exact task006 formulation.

ConvTranspose aligns the two halves and produces a positive score exactly for
their logical AND. Greater and Where create explicit background/red one-hot
channels, and Pad writes the 3x3 result to the fixed NeuroGolf canvas.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_model():
    weights = np.zeros((10, 1, 1, 2), dtype=np.float32)
    weights[0, 0, 0, 0] = -1.0
    weights[1, 0, 0, 1] = 1.0

    graph = helper.make_graph(
        [
            helper.make_node(
                "ConvTranspose",
                ["input", "weights"],
                ["score"],
                dilations=[1, 4],
                kernel_shape=[1, 2],
                pads=[0, 4, 27, 27],
                strides=[1, 1],
            ),
            helper.make_node("Greater", ["score", "zero"], ["red_mask"]),
            helper.make_node("Where", ["red_mask", "red", "black"], ["small"]),
            helper.make_node(
                "Pad",
                ["small", "pads"],
                ["output"],
                mode="constant",
            ),
        ],
        "task006_fused_threshold",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        [
            numpy_helper.from_array(weights, name="weights"),
            numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), name="zero"),
            numpy_helper.from_array(
                np.asarray([0, 0, 1], dtype=np.uint8).reshape(1, 3, 1, 1), name="red"
            ),
            numpy_helper.from_array(
                np.asarray([1, 0, 0], dtype=np.uint8).reshape(1, 3, 1, 1), name="black"
            ),
            numpy_helper.from_array(
                np.asarray([0, 0, 0, 0, 0, 7, 27, 27], dtype=np.int64), name="pads"
            ),
        ],
    )
    return helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 18)],
    )


if __name__ == "__main__":
    destination = Path(__file__).with_name("task006.onnx")
    model = build_model()
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, destination)
    print(destination)
