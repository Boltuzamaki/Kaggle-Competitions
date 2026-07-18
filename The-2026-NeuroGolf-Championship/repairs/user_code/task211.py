# --- task211 optimized signature-correct
# Input/output stay FLOAT [1,10,30,30], same harness style as task002.
#
# Rule:
#   input crop: top-left 3x2
#   columns: [1,0,0,1]
#   rows:    [2,1,0,0,1,2,2,1,0]
#   then pad 9x4 result back to 30x30 with zeros.

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

F = TensorProto.FLOAT

x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

def K(name, arr, dtype=np.int64):
    return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)

inits = [
    # crop [1,10,3,2]
    K("s",   [0, 0, 0, 0]),
    K("e",   [1, 10, 3, 2]),
    K("ax",  [0, 1, 2, 3]),

    # horizontal mirror: [a,b] -> [b,a,a,b]
    K("ic",  [1, 0, 0, 1]),

    # vertical mirror/tile:
    # r2,r1,r0, r0,r1,r2, r2,r1,r0
    K("ir",  [2, 1, 0, 0, 1, 2, 2, 1, 0]),

    # pad H: 9 -> 30, W: 4 -> 30
    # pads format: N,C,H,W begin then N,C,H,W end
    K("pad", [0, 0, 0, 0, 0, 0, 21, 26]),
]

nodes = [
    # [1,10,30,30] -> [1,10,3,2]
    helper.make_node("Slice", ["input", "s", "e", "ax"], ["c"]),

    # [1,10,3,2] -> [1,10,3,4]
    helper.make_node("Gather", ["c", "ic"], ["h"], axis=3),

    # [1,10,3,4] -> [1,10,9,4]
    helper.make_node("Gather", ["h", "ir"], ["o9"], axis=2),

    # [1,10,9,4] -> [1,10,30,30]
    helper.make_node("Pad", ["o9", "pad"], ["output"]),
]

model = helper.make_model(
    helper.make_graph(nodes, "task211_gather_pad", [x], [y], inits),
    ir_version=10,
    opset_imports=[helper.make_opsetid("", 12)],
)

onnx.checker.check_model(model)