# --- task203 one-hot FLOAT signature solve
# Rule:
#   concentric rectangle/ring colors are reversed.
#   Geometry stays unchanged.
#
# Input/output signature matches your working task002 style:
#   input  FLOAT [1,10,30,30]
#   output FLOAT [1,10,30,30]
#
# Strategy:
#   1. active = sum(input over color channels), detects real grid vs zero padding
#   2. N = number of active cells in top row
#   3. R = N // 2
#   4. raw = ArgMax(input, axis=1)  -> INT64 [1,30,30]
#   5. diag = raw[0,i,i] for i=0..R-1
#   6. lut[diag[i]] = diag[R-1-i]
#   7. recolor raw through lut
#   8. OneHot back to FLOAT [1,10,30,30]
#   9. multiply by active mask so padding remains all-zero

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

F = TensorProto.FLOAT
I64 = TensorProto.INT64

x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

def K(name, arr, dtype):
    return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)

inits = [
    K("zero_i", 0, np.int64),
    K("one_i", 1, np.int64),
    K("two_i", 2, np.int64),
    K("neg1_i", -1, np.int64),
    K("ten_i", 10, np.int64),

    K("s_row0", [0, 0, 0, 0], np.int64),
    K("e_row0", [1, 1, 1, 30], np.int64),
    K("ax4", [0, 1, 2, 3], np.int64),

    K("lut0", np.zeros(10, dtype=np.int64), np.int64),
    K("oh_vals", [0.0, 1.0], np.float32),
]

nodes = [
    # active: [1,1,30,30], 1 on real cells, 0 on padding
    helper.make_node("ReduceSum", ["input"], ["active"], axes=[1], keepdims=1),

    # top active row: [1,1,1,30]
    helper.make_node("Slice", ["active", "s_row0", "e_row0", "ax4"], ["row0"]),

    # Nf = number of active cells in top row
    helper.make_node("ReduceSum", ["row0"], ["Nf"], axes=[0, 1, 2, 3], keepdims=0),

    # N int64
    helper.make_node("Cast", ["Nf"], ["N"], to=I64),

    # R = N // 2
    helper.make_node("Div", ["N", "two_i"], ["R"]),

    # raw color grid: [1,30,30]
    helper.make_node("ArgMax", ["input"], ["raw"], axis=1, keepdims=0),

    # i = [0,1,...,R-1]
    helper.make_node("Range", ["zero_i", "R", "one_i"], ["i"]),

    # indices [0,i,i] for GatherND
    helper.make_node("Mul", ["i", "zero_i"], ["b0"]),
    helper.make_node("Unsqueeze", ["b0"], ["b0u"], axes=[1]),
    helper.make_node("Unsqueeze", ["i"], ["iu"], axes=[1]),
    helper.make_node("Concat", ["b0u", "iu", "iu"], ["diag_idx"], axis=1),

    # diag = outside -> inside ring colors
    helper.make_node("GatherND", ["raw", "diag_idx"], ["diag"]),

    # reverse indices [R-1,...,0]
    helper.make_node("Sub", ["R", "one_i"], ["last"]),
    helper.make_node("Range", ["last", "neg1_i", "neg1_i"], ["ri"]),

    # rdiag = inside -> outside ring colors
    helper.make_node("Gather", ["diag", "ri"], ["rdiag"], axis=0),

    # lut[original_color] = reversed_color
    helper.make_node("ScatterElements", ["lut0", "diag", "rdiag"], ["lut"], axis=0),

    # recolor raw grid
    helper.make_node("Gather", ["lut", "raw"], ["raw_out"], axis=0),

    # back to one-hot: [1,30,30,10]
    helper.make_node("OneHot", ["raw_out", "ten_i", "oh_vals"], ["oh"]),

    # transpose to [1,10,30,30]
    helper.make_node("Transpose", ["oh"], ["oh_chw"], perm=[0, 3, 1, 2]),

    # remove one-hot padding; padding must stay all-zero
    helper.make_node("Mul", ["oh_chw", "active"], ["output"]),
]

model = helper.make_model(
    helper.make_graph(nodes, "task203_onehot_ring_reverse", [x], [y], inits),
    ir_version=10,
    opset_imports=[helper.make_opsetid("", 12)],
)

onnx.checker.check_model(model)