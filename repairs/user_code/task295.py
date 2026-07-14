"""Build the compact exact task295 triangular-prefix renderer.

For an input colored prefix of length L on a 1xW grid, output row r has
L+r colored cells and the output has W/2 rows.  All renderer arithmetic is
uint8 so ONNX Runtime does not promote the former float16 tensors to float32.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path.cwd()
DESTINATION = ROOT / "other_model_onnx" / "task295.onnx"


def init(name, value, dtype):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


def build() -> onnx.ModelProto:
    nodes = [
        helper.make_node("Einsum", ["input"], ["counts"], equation="bchw->c"),
        helper.make_node("ReduceSum", ["counts"], ["width"], keepdims=0),
        helper.make_node(
            "Slice", ["counts", "ch_start", "ch_end"], ["nz_counts"]
        ),
        helper.make_node("ReduceSum", ["nz_counts"], ["length"], keepdims=0),
        helper.make_node(
            "ArgMax", ["nz_counts"], ["color_minus_one"], axis=0, keepdims=0
        ),
        helper.make_node("Cast", ["color_minus_one"], ["color_u8"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["width"], ["width_u8"], to=TensorProto.UINT8),
        helper.make_node(
            "BitShift", ["width_u8", "shift_one_u8"], ["half_width"], direction="RIGHT"
        ),
        helper.make_node("Less", ["row_index", "half_width"], ["in_height"]),
        helper.make_node("Less", ["col_index", "width_u8"], ["in_width"]),
        helper.make_node("Cast", ["length"], ["length_u8"], to=TensorProto.UINT8),
        helper.make_node("Add", ["row_index", "length_u8"], ["fill_limit_raw"]),
        helper.make_node(
            "Where", ["in_height", "fill_limit_raw", "zero_u8"], ["fill_limit"]
        ),
        helper.make_node("Less", ["col_index", "fill_limit"], ["filled"]),
        helper.make_node(
            "Where", ["in_height", "black_u8", "outside_u8"], ["row_default"]
        ),
        helper.make_node(
            "Where", ["in_width", "row_default", "outside_u8"], ["default_label"]
        ),
        helper.make_node(
            "Where", ["filled", "color_u8", "default_label"], ["small_label"]
        ),
        helper.make_node(
            "Pad", ["small_label", "pad_spec", "outside_u8"], ["full_label"], mode="constant"
        ),
        helper.make_node("Equal", ["full_label", "channel_label_u8"], ["output"]),
    ]

    initializers = [
        init("ch_start", [1], np.int64),
        init("ch_end", [10], np.int64),
        init("black_u8", 9, np.uint8),
        init("outside_u8", 10, np.uint8),
        init("zero_u8", 0, np.uint8),
        init("shift_one_u8", 1, np.uint8),
        init("row_index", np.arange(9).reshape(9, 1), np.uint8),
        init("col_index", np.arange(18).reshape(1, 18), np.uint8),
        init("pad_spec", [0, 0, 21, 12], np.int64),
        init(
            "channel_label_u8",
            np.asarray([9, 0, 1, 2, 3, 4, 5, 6, 7, 8], np.uint8).reshape(1, 10, 1, 1),
            np.uint8,
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task295_uint8",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="task295_uint8",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


model = build()


if __name__ == "__main__":
    onnx.save(model, DESTINATION)
    print(DESTINATION)
