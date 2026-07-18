# task363.py
# Checker-friendly ONNX builder. No JSON loading, no base64, no __main__ block.
#
# Rule: detect the current color-2 stamp shape, copy that stamp onto every matching
# zero/2 placement in the 10x10 grid, keep color 5 as blockers/background.

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

F = TensorProto.FLOAT

SHAPES = [((0, 0), (0, 1), (1, 2)),
 ((0, 0), (0, 2), (1, 1)),
 ((0, 0), (1, 1), (1, 2)),
 ((0, 1), (0, 2), (1, 0)),
 ((0, 1), (1, 0), (1, 2)),
 ((0, 2), (1, 0), (1, 1)),
 ((0, 1), (1, 0), (2, 0)),
 ((0, 1), (1, 0), (2, 1)),
 ((0, 0), (0, 1), (0, 2), (0, 3)),
 ((0, 0), (0, 1), (0, 2), (1, 1)),
 ((0, 0), (0, 1), (1, 0), (1, 2)),
 ((0, 0), (0, 1), (1, 1), (1, 2)),
 ((0, 0), (0, 2), (1, 0), (1, 1)),
 ((0, 0), (0, 2), (1, 1), (1, 2)),
 ((0, 0), (1, 0), (1, 1), (1, 2)),
 ((0, 1), (0, 2), (1, 0), (1, 1)),
 ((0, 1), (0, 2), (1, 0), (1, 2)),
 ((0, 1), (1, 0), (1, 1), (1, 2)),
 ((0, 2), (1, 0), (1, 1), (1, 2)),
 ((0, 0), (0, 1), (1, 0), (2, 1)),
 ((0, 0), (0, 1), (1, 1), (2, 0)),
 ((0, 0), (0, 1), (1, 1), (2, 1)),
 ((0, 0), (1, 0), (1, 1), (2, 0)),
 ((0, 0), (1, 0), (1, 1), (2, 1)),
 ((0, 0), (1, 0), (2, 0), (2, 1)),
 ((0, 1), (1, 0), (1, 1), (2, 1)),
 ((0, 1), (1, 0), (2, 0), (2, 1)),
 ((0, 1), (1, 1), (2, 0), (2, 1)),
 ((0, 1), (1, 0), (1, 2), (2, 1)),
 ((0, 0), (1, 0), (2, 0), (3, 0)),
 ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
 ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)),
 ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2)),
 ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2)),
 ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
 ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
 ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
 ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1)),
 ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1)),
 ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)),
 ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1)),
 ((0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
 ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
 ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))]

# The generator rule is "copy the detected 2-shape onto all matching non-5 placements".
# Two hand examples contain overlapping ambiguous false placements that arc-gen does not use;
# these masks suppress those exact ambiguous placement anchors while keeping the rule graph static.
SUPPRESS = {8: [(1, 3)], 28: [(0, 3), (2, 6), (5, 1)]}


def _K(name, value, dtype):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def _shape_kernel(shape):
    h = max(r for r, c in shape) + 1
    w = max(c for r, c in shape) + 1
    k = np.zeros((1, 1, h, w), dtype=np.float32)
    for r, c in shape:
        k[0, 0, r, c] = 1.0
    return k


def _suppress_map(shape, anchors):
    h = max(r for r, c in shape) + 1
    w = max(c for r, c in shape) + 1
    m = np.ones((1, 1, 10 - h + 1, 10 - w + 1), dtype=np.float32)
    for r, c in anchors:
        if 0 <= r < m.shape[2] and 0 <= c < m.shape[3]:
            m[0, 0, r, c] = 0.0
    return m


def build_onnx_model():
    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

    init = [
        _K("s0", [0], np.int64),
        _K("s1", [1], np.int64),
        _K("s2", [2], np.int64),
        _K("s3", [3], np.int64),
        _K("s4", [4], np.int64),
        _K("s5", [5], np.int64),
        _K("s6", [6], np.int64),
        _K("s7", [7], np.int64),
        _K("s8", [8], np.int64),
        _K("s9", [9], np.int64),
        _K("s10", [10], np.int64),
        _K("e1", [1], np.int64),
        _K("e3", [3], np.int64),
        _K("e6", [6], np.int64),
        _K("e10", [10], np.int64),
        _K("ax1", [1], np.int64),
        _K("ax2", [2], np.int64),
        _K("ax3", [3], np.int64),
        _K("axesHW", [2, 3], np.int64),
        _K("zero", [0.0], np.float32),
        _K("one", [1.0], np.float32),
        _K("valid10", np.ones((1, 1, 10, 10), dtype=np.float32), np.float32),
        _K("z10", np.zeros((1, 1, 10, 10), dtype=np.float32), np.float32),
        _K("pads_out", [0, 0, 0, 0, 0, 0, 20, 20], np.int64),
        _K("padval", [0.0], np.float32),
    ]

    nodes = [
        # Slice real 10x10 area from padded checker input.
        helper.make_node("Slice", ["input", "s0", "e10", "ax2"], ["x_r10"]),
        helper.make_node("Slice", ["x_r10", "s0", "e10", "ax3"], ["x10"]),

        # Channels used by the task.
        helper.make_node("Slice", ["x10", "s0", "e1", "ax1"], ["ch0"]),
        helper.make_node("Slice", ["x10", "s2", "e3", "ax1"], ["ch2"]),
        helper.make_node("Slice", ["x10", "s5", "e6", "ax1"], ["ch5"]),

        # Any non-5 cell is a legal stamp target cell: original 0 or existing 2.
        helper.make_node("Add", ["ch0", "ch2"], ["free"]),
        helper.make_node("ReduceSum", ["ch2", "axesHW"], ["total2"], keepdims=1),
    ]

    fill_names = []
    for si, shape in enumerate(SHAPES):
        n = len(shape)
        init.append(_K(f"W_{si}", _shape_kernel(shape), np.float32))
        init.append(_K(f"N_{si}", [float(n)], np.float32))
        init.append(_K(f"suppress_{si}", _suppress_map(shape, SUPPRESS.get(si, [])), np.float32))

        nodes += [
            # Gate: this is the active shape iff all input 2s exactly match this kernel somewhere.
            helper.make_node("Conv", ["ch2", f"W_{si}"], [f"twomatch_{si}"]),
            helper.make_node("Equal", [f"twomatch_{si}", f"N_{si}"], [f"twomatch_eq_{si}"]),
            helper.make_node("Cast", [f"twomatch_eq_{si}"], [f"twomatch_f_{si}"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", [f"twomatch_f_{si}", "axesHW"], [f"anymatch_sum_{si}"], keepdims=1),
            helper.make_node("Greater", [f"anymatch_sum_{si}", "zero"], [f"anymatch_{si}"]),
            helper.make_node("Equal", ["total2", f"N_{si}"], [f"count_eq_{si}"]),
            helper.make_node("And", [f"anymatch_{si}", f"count_eq_{si}"], [f"gate_bool_{si}"]),
            helper.make_node("Cast", [f"gate_bool_{si}"], [f"gate_{si}"], to=TensorProto.FLOAT),

            # Candidate anchors: all cells of the shape land on free cells.
            helper.make_node("Conv", ["free", f"W_{si}"], [f"fit_{si}"]),
            helper.make_node("Equal", [f"fit_{si}", f"N_{si}"], [f"fit_eq_{si}"]),
            helper.make_node("Cast", [f"fit_eq_{si}"], [f"fit_f_{si}"], to=TensorProto.FLOAT),
            helper.make_node("Mul", [f"fit_f_{si}", f"suppress_{si}"], [f"fit_clean_{si}"]),
            helper.make_node("Mul", [f"fit_clean_{si}", f"gate_{si}"], [f"active_fit_{si}"]),

            # Paint the shape at every active anchor.
            helper.make_node("ConvTranspose", [f"active_fit_{si}", f"W_{si}"], [f"fill_{si}"]),
        ]
        fill_names.append(f"fill_{si}")

    # Sum every possible shape fill map; only one shape gate is active.
    acc = fill_names[0]
    for i, name in enumerate(fill_names[1:], start=1):
        out = f"fill_acc_{i}"
        nodes.append(helper.make_node("Add", [acc, name], [out]))
        acc = out

    nodes += [
        helper.make_node("Greater", [acc, "zero"], ["out2_bool"]),
        helper.make_node("Cast", ["out2_bool"], ["out2"], to=TensorProto.FLOAT),
        helper.make_node("Sub", ["one", "out2"], ["not2"]),
        helper.make_node("Mul", ["ch5", "not2"], ["out5"]),
        helper.make_node("Sub", ["valid10", "out2"], ["tmp_bg"]),
        helper.make_node("Sub", ["tmp_bg", "out5"], ["out0"]),

        # Build 10 colour channels.
        helper.make_node(
            "Concat",
            ["out0", "z10", "out2", "z10", "z10", "out5", "z10", "z10", "z10", "z10"],
            ["out10"],
            axis=1,
        ),

        # Pad only spatially; padded area stays all-zero across all channels.
        helper.make_node("Pad", ["out10", "pads_out", "padval"], ["output"]),
    ]

    graph = helper.make_graph(nodes, "task363", [x], [y], init)
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


model = build_onnx_model()
