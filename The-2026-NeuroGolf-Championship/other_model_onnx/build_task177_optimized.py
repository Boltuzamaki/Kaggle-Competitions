"""Build an exact dynamic-slice rewrite for task177."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "other_model_onnx" / "task177.onnx"


def init(name, value, dtype=None):
    array = np.asarray(value, dtype=dtype)
    return numpy_helper.from_array(array, name=name)


nodes = [
    helper.make_node("Einsum", ["input", "w"], ["rrow"], equation="nkbd,k->nb"),
    helper.make_node("Einsum", ["input", "w"], ["rcol"], equation="nkbd,k->nd"),
    helper.make_node("ArgMax", ["rrow"], ["top"], axis=1, keepdims=0),
    helper.make_node("ArgMax", ["rcol"], ["right"], axis=1, keepdims=0, select_last_index=1),
    helper.make_node("Add", ["top", "eight"], ["top_end"]),
    helper.make_node("Sub", ["right", "eight"], ["right_end"]),
    helper.make_node("Greater", ["right", "seven"], ["full_width"]),
    helper.make_node("Where", ["full_width", "right_end", "min_end"], ["safe_right_end"]),
    helper.make_node("Concat", ["channel_start", "top", "right"], ["starts"], axis=0),
    helper.make_node("Concat", ["channel_end", "top_end", "safe_right_end"], ["ends"], axis=0),
    helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["crop"]),
    helper.make_node("Sub", ["twenty_nine", "right"], ["right_gap"]),
    helper.make_node("Max", ["right_gap", "twenty_two"], ["pad_right"]),
    helper.make_node("Concat", ["pad_prefix", "pad_right"], ["pads"], axis=0),
    helper.make_node("Pad", ["crop", "pads"], ["output"], mode="constant"),
]

initializers = [
    init("w", [0] + [1] * 9, np.float32),
    init("eight", [8], np.int64),
    init("seven", [7], np.int64),
    init("min_end", [np.iinfo(np.int64).min], np.int64),
    init("channel_start", [1], np.int64),
    init("channel_end", [10], np.int64),
    init("twenty_nine", [29], np.int64),
    init("twenty_two", [22], np.int64),
    init("axes", [1, 2, 3], np.int64),
    init("steps", [1, 1, -1], np.int64),
    init("pad_prefix", [0, 1, 0, 0, 0, 0, 22], np.int64),
]

input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

value_info = [
    helper.make_tensor_value_info("rrow", TensorProto.FLOAT, [1, 30]),
    helper.make_tensor_value_info("rcol", TensorProto.FLOAT, [1, 30]),
]
for name in ("top", "right", "top_end", "right_end", "safe_right_end", "right_gap", "pad_right"):
    value_info.append(helper.make_tensor_value_info(name, TensorProto.INT64, [1]))
value_info.append(helper.make_tensor_value_info("full_width", TensorProto.BOOL, [1]))
for name in ("starts", "ends"):
    value_info.append(helper.make_tensor_value_info(name, TensorProto.INT64, [3]))
value_info.extend([
    helper.make_tensor_value_info("crop", TensorProto.FLOAT, [1, 9, 8, 8]),
    helper.make_tensor_value_info("pads", TensorProto.INT64, [8]),
])

graph = helper.make_graph(
    nodes,
    "task177_dynamic_slice",
    [input_info],
    [output_info],
    initializers,
    value_info=value_info,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, OUTPUT)
print(OUTPUT)
