# task344.py
# NeuroGolf / ARC-AGI task344
#
# Rule:
#   Every color-3 cell that touches a color-2 cell orthogonally becomes color 8.
#   Every color-2 cell that touches a color-3 cell orthogonally is erased to 0.
#   All other cells are unchanged.
#
# Checker-friendly: defines a top-level ONNX ModelProto named `model`.
# No internal save path and no __main__ dependency.

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

F = TensorProto.FLOAT
I64 = TensorProto.INT64


def _K(name, value, dtype):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def _slice_ch(nodes, x, ch, out):
    nodes.append(
        helper.make_node(
            "Slice",
            [x, f"c{ch}s", f"c{ch}e", "ax1"],
            [out],
        )
    )


def _shift(nodes, x, name, direction):
    # x: [1,1,30,30]. Return neighbor map at each cell.
    # direction is the location of the neighbor relative to the current cell.
    if direction == "up":
        pads = "pad_top"
        s = "s0"
        e = "e30"
        axis = "ax2"
    elif direction == "down":
        pads = "pad_bottom"
        s = "s1"
        e = "e31"
        axis = "ax2"
    elif direction == "left":
        pads = "pad_left"
        s = "s0"
        e = "e30"
        axis = "ax3"
    elif direction == "right":
        pads = "pad_right"
        s = "s1"
        e = "e31"
        axis = "ax3"
    else:
        raise ValueError(direction)

    padded = f"{name}_pad_{direction}"
    nodes.append(helper.make_node("Pad", [x, pads, "zero"], [padded]))
    nodes.append(helper.make_node("Slice", [padded, s, e, axis], [f"{name}_{direction}"]))
    return f"{name}_{direction}"


def _neighbor_any(nodes, x, prefix):
    shifted = [
        _shift(nodes, x, prefix, "up"),
        _shift(nodes, x, prefix, "down"),
        _shift(nodes, x, prefix, "left"),
        _shift(nodes, x, prefix, "right"),
    ]
    nodes.append(helper.make_node("Add", [shifted[0], shifted[1]], [f"{prefix}_a"]))
    nodes.append(helper.make_node("Add", [shifted[2], shifted[3]], [f"{prefix}_b"]))
    nodes.append(helper.make_node("Add", [f"{prefix}_a", f"{prefix}_b"], [f"{prefix}_sum"]))
    nodes.append(helper.make_node("Greater", [f"{prefix}_sum", "half"], [f"{prefix}_bool"]))
    nodes.append(helper.make_node("Cast", [f"{prefix}_bool"], [f"{prefix}_any"], to=TensorProto.FLOAT))
    return f"{prefix}_any"


def build_onnx_model():
    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

    init = [
        _K("ax1", [1], np.int64),
        _K("ax2", [2], np.int64),
        _K("ax3", [3], np.int64),
        _K("s0", [0], np.int64),
        _K("s1", [1], np.int64),
        _K("e30", [30], np.int64),
        _K("e31", [31], np.int64),
        _K("zero", [0.0], np.float32),
        _K("half", [0.5], np.float32),
        _K("pad_top",    [0, 0, 1, 0, 0, 0, 0, 0], np.int64),
        _K("pad_bottom", [0, 0, 0, 0, 0, 0, 1, 0], np.int64),
        _K("pad_left",   [0, 0, 0, 1, 0, 0, 0, 0], np.int64),
        _K("pad_right",  [0, 0, 0, 0, 0, 0, 0, 1], np.int64),
    ]
    for c in range(10):
        init.append(_K(f"c{c}s", [c], np.int64))
        init.append(_K(f"c{c}e", [c + 1], np.int64))

    nodes = []
    ch = []
    for c in range(10):
        name = f"ch{c}"
        _slice_ch(nodes, "input", c, name)
        ch.append(name)

    neigh2 = _neighbor_any(nodes, "ch2", "n2")
    neigh3 = _neighbor_any(nodes, "ch3", "n3")

    # 3 touching 2 -> 8
    nodes.append(helper.make_node("Mul", ["ch3", neigh2], ["three_to_eight"]))

    # 2 touching 3 -> 0
    nodes.append(helper.make_node("Mul", ["ch2", neigh3], ["two_to_zero"]))

    out_ch = []
    for c in range(10):
        if c == 0:
            nodes.append(helper.make_node("Add", ["ch0", "two_to_zero"], ["out0"]))
            out_ch.append("out0")
        elif c == 2:
            nodes.append(helper.make_node("Sub", ["ch2", "two_to_zero"], ["out2"]))
            out_ch.append("out2")
        elif c == 3:
            nodes.append(helper.make_node("Sub", ["ch3", "three_to_eight"], ["out3"]))
            out_ch.append("out3")
        elif c == 8:
            nodes.append(helper.make_node("Add", ["ch8", "three_to_eight"], ["out8"]))
            out_ch.append("out8")
        else:
            out_ch.append(f"ch{c}")

    nodes.append(helper.make_node("Concat", out_ch, ["output"], axis=1))

    graph = helper.make_graph(nodes, "task344", [x], [y], init)
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model)
    return model


model = build_onnx_model()
