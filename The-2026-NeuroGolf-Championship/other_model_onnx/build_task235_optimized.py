"""Build the compact exact task235 classifier.

The four glyphs are identified by the sum of three selected one-hot cells:

    output colour:  2  3  4  8
    compact code:   2  3  1  0

Einsum performs the unweighted sum without a parameter tensor.  The remaining
XOR/ConvInteger pair turns code equality into positive one-hot output and uses
the width-three convolution kernel to repeat each answer across three columns.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def tensor(name, values, dtype):
    return numpy_helper.from_array(np.asarray(values, dtype=dtype), name=name)


def build_model():
    # Gather indices are [channel, row, column].  The final singleton dimension
    # lets Einsum retain a width-one axis while summing the three features.
    indices = []
    for col0 in (0, 5, 10):
        indices.append(
            [
                [[0, 1, col0]],      # background: unique to glyph colour 3
                [[5, 1, col0 + 1]],  # foreground: absent only for glyph 8
                [[5, 2, col0 + 1]],  # foreground: present for glyphs 2 and 3
            ]
        )
    indices = np.asarray(indices, dtype=np.int64).reshape(1, 1, 3, 3, 1, 3)

    initializers = [
        tensor("idx", indices, np.int64),
        tensor(
            "ids",
            np.asarray([255, 255, 2, 3, 1, 255, 255, 255, 0, 255]).reshape(1, 10, 1, 1),
            np.uint8,
        ),
        tensor("repeat", -np.ones((10, 1, 1, 3)), np.int8),
        tensor("one", 1, np.uint8),
    ]

    nodes = [
        helper.make_node("GatherND", ["input", "idx"], ["bits"], batch_dims=1),
        helper.make_node("Einsum", ["bits"], ["code_f"], equation="abcde->abce"),
        helper.make_node("Cast", ["code_f"], ["code"], to=TensorProto.UINT8),
        helper.make_node("BitwiseXor", ["code", "ids"], ["distance"]),
        helper.make_node(
            "ConvInteger",
            ["distance", "repeat", "one"],
            ["output"],
            group=10,
            pads=[0, 2, 27, 29],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task235_compact_sum_code",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, 30, 30])],
        initializers,
    )
    return helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 18)],
    )


if __name__ == "__main__":
    destination = Path(__file__).with_name("task235.onnx")
    model = build_model()
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, destination)
    print(destination)
