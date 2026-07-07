# --- task003: extend the 6x3 binary row pattern to 9x3 and recolor 1 -> 2.
# The final ConvInteger expands channels 0 and 2 directly into the free
# graph output, avoiding the baseline's explicit zero channel and Pad input.
import numpy as np
from onnx import TensorProto, helper, numpy_helper

F = TensorProto.FLOAT
x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
y = helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, 30, 30])


def K(name, array):
    return numpy_helper.from_array(array, name)


W = np.zeros((10, 2, 1, 1), dtype=np.uint8)
W[0, 0, 0, 0] = 1
W[2, 1, 0, 0] = 1

inits = [
    K("slice_blue_starts", np.array([0, 1, 0, 0], np.int64)),
    K("slice_blue_ends", np.array([1, 2, 6, 3], np.int64)),
    K("row0_idx", np.array([0], np.int64)),
    K("row1_idx", np.array([1], np.int64)),
    K("row2_idx", np.array([2], np.int64)),
    K("row3_idx", np.array([3], np.int64)),
    K("one_u8", np.array([1], np.uint8)),
    K("W", W),
]

nodes = [
    helper.make_node("Slice", ["input", "slice_blue_starts", "slice_blue_ends"], ["one6_f"]),
    helper.make_node("Cast", ["one6_f"], ["one6"], to=TensorProto.UINT8),
    helper.make_node("Gather", ["one6", "row0_idx"], ["r0"], axis=2),
    helper.make_node("Gather", ["one6", "row1_idx"], ["r1"], axis=2),
    helper.make_node("Gather", ["one6", "row2_idx"], ["r2"], axis=2),
    helper.make_node("Gather", ["one6", "row3_idx"], ["r3"], axis=2),
    helper.make_node("Equal", ["r0", "r3"], ["eq_row03"]),
    helper.make_node("Split", ["eq_row03"], ["eq0", "eq1", "eq2"], axis=3, num_outputs=3),
    helper.make_node("And", ["eq0", "eq1"], ["eq01"]),
    helper.make_node("And", ["eq01", "eq2"], ["is_period3"]),
    helper.make_node("Where", ["is_period3", "r0", "r2"], ["tail0"]),
    helper.make_node("Where", ["is_period3", "r1", "r3"], ["tail1"]),
    helper.make_node("Where", ["is_period3", "r2", "r0"], ["tail2"]),
    helper.make_node("Concat", ["one6", "tail0", "tail1", "tail2"], ["ch2_valid"], axis=2),
    helper.make_node("BitwiseXor", ["one_u8", "ch2_valid"], ["ch0_valid"]),
    helper.make_node("Concat", ["ch0_valid", "ch2_valid"], ["pair_valid"], axis=1),
    helper.make_node("ConvInteger", ["pair_valid", "W"], ["output"], pads=[0, 0, 21, 27]),
]

model = helper.make_model(
    helper.make_graph(nodes, "task003_convint", [x], [y], inits),
    ir_version=10,
    opset_imports=[helper.make_opsetid("", 18)],
)
