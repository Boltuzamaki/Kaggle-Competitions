# --- task002 lower-cost cropped exact flood-fill
# Push from your solved N=20 / ITERS=25 version:
#   ITERS 25 -> 20
#   final Min -> Sign
#
# Keep:
#   free = 1 - sum(non-background channels)
#   padding traversable
#   final fill gated by ch0
#   one broadcast recolor delta

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

F = TensorProto.FLOAT

N = 20
ITERS = 20
PAD = 30 - N

x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

def K(name, arr, dtype=None):
    return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)

border = np.zeros((1, 1, N, N), np.float32)
border[:, :, 0, :] = 1
border[:, :, N - 1, :] = 1
border[:, :, :, 0] = 1
border[:, :, :, N - 1] = 1

plus = np.array(
    [[[[0, 1, 0],
       [1, 1, 1],
       [0, 1, 0]]]],
    dtype=np.float32,
)

# Broadcasted final recolor delta:
#   channel 0 -= enclosed
#   channel 4 += enclosed
dk = np.zeros((1, 10, 1, 1), np.float32)
dk[:, 0, :, :] = -1.0
dk[:, 4, :, :] =  1.0

inits = [
    K("s19", [0, 1, 0, 0], np.int64),
    K("e19", [1, 10, N, N], np.int64),
    K("s0",  [0, 0, 0, 0], np.int64),
    K("e0",  [1, 1, N, N], np.int64),
    K("ax",  [0, 1, 2, 3], np.int64),

    K("one", np.array(1.0, np.float32)),
    K("border", border),
    K("plus", plus),
    K("dk", dk),
    K("padsp", [0, 0, 0, 0, 0, 0, PAD, PAD], np.int64),
]

nodes = [
    # free = 1 - any non-background channel.
    # This keeps all-zero padding traversable.
    helper.make_node("Slice", ["input", "s19", "e19", "ax"], ["nz"]),
    helper.make_node("ReduceSum", ["nz"], ["nonbg"], axes=[1], keepdims=1),
    helper.make_node("Sub", ["one", "nonbg"], ["free"]),

    # outside seed on cropped border
    helper.make_node("Mul", ["free", "border"], ["o0"]),
]

prev = "o0"

for i in range(1, ITERS + 1):
    nodes += [
        helper.make_node(
            "Conv",
            [prev, "plus"],
            [f"d{i}"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Mul", [f"d{i}", "free"], [f"o{i}"]),
    ]
    prev = f"o{i}"

nodes += [
    # prev is non-negative, so Sign gives 0/1 reachability.
    helper.make_node("Sign", [prev], ["outside"]),

    # Only true channel-0 background cells can become 4.
    # Padding is not recolored because ch0 is 0 there.
    helper.make_node("Slice", ["input", "s0", "e0", "ax"], ["ch0"]),
    helper.make_node("Sub", ["one", "outside"], ["notout"]),
    helper.make_node("Mul", ["ch0", "notout"], ["enc20"]),

    # Back to full 30x30 tensor.
    helper.make_node("Pad", ["enc20", "padsp"], ["enc30"]),

    # One broadcasted recolor tensor.
    helper.make_node("Mul", ["enc30", "dk"], ["delta"]),
    helper.make_node("Add", ["input", "delta"], ["output"]),
]

model = helper.make_model(
    helper.make_graph(nodes, "task002_n20_i20_sign", [x], [y], inits),
    ir_version=10,
    opset_imports=[helper.make_opsetid("", 12)],
)

onnx.checker.check_model(model)