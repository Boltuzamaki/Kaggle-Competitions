# task313.py
# Checker-friendly: exposes top-level `model`.
# Rule: the real grid contains a repeating 2-row horizontal color pattern in the
# top-left, with a filler color occupying the unused right/bottom area.  The
# output is the same grid size filled by the pattern shifted one column left.
# The row pattern period is either 2 or 3; detect it from row0 col0 == row0 col2.

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

F = TensorProto.FLOAT
I64 = np.int64
FP = np.float32


def _K(name, value, dtype):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def _slice_pixel(nodes, src, r, c, out):
    nodes.append(helper.make_node("Slice", [src, f"r{r}", f"r{r+1}", "ax2"], [f"{out}_r"]))
    nodes.append(helper.make_node("Slice", [f"{out}_r", f"c{c}", f"c{c+1}", "ax3"], [out]))


def _mul3(nodes, a, b, c, out):
    nodes.append(helper.make_node("Mul", [a, b], [f"{out}_m0"]))
    nodes.append(helper.make_node("Mul", [f"{out}_m0", c], [out]))


def build_onnx_model():
    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

    init = [
        _K("ax1", [1], I64),
        _K("ax2", [2], I64),
        _K("ax3", [3], I64),
        _K("zero", [0.0], FP),
        _K("one", [1.0], FP),
        _K("half", [0.5], FP),
    ]
    for i in range(31):
        init.append(_K(f"r{i}", [i], I64))
        init.append(_K(f"c{i}", [i], I64))

    # Position masks.  p2 uses row parity + col parity.  p3 uses row parity + col mod 3.
    for rp in (0, 1):
        for cm in (0, 1):
            m = np.zeros((1, 1, 30, 30), dtype=FP)
            for r in range(30):
                for c in range(30):
                    if (r % 2 == rp) and (c % 2 == cm):
                        m[0, 0, r, c] = 1.0
            init.append(_K(f"m2_r{rp}_c{cm}", m, FP))
        for cm in (0, 1, 2):
            m = np.zeros((1, 1, 30, 30), dtype=FP)
            for r in range(30):
                for c in range(30):
                    if (r % 2 == rp) and (c % 3 == cm):
                        m[0, 0, r, c] = 1.0
            init.append(_K(f"m3_r{rp}_c{cm}", m, FP))

    n = []

    # Valid real cells: checker pads outside the ARC grid with all-zero channels.
    n.append(helper.make_node("ReduceSum", ["input"], ["valid"], axes=[1], keepdims=1))

    # Read the six color vectors needed from the first two rows.
    for r in (0, 1):
        for c in (0, 1, 2):
            _slice_pixel(n, "input", r, c, f"v{r}{c}")

    # Detect period 2 vs period 3.  For period 2, row0 col0 == row0 col2.
    n.append(helper.make_node("Mul", ["v00", "v02"], ["eq02_mul"]))
    n.append(helper.make_node("ReduceSum", ["eq02_mul"], ["eq02"], axes=[1, 2, 3], keepdims=1))
    n.append(helper.make_node("Greater", ["eq02", "half"], ["p2_bool"]))
    n.append(helper.make_node("Cast", ["p2_bool"], ["p2"], to=TensorProto.FLOAT))
    n.append(helper.make_node("Sub", ["one", "p2"], ["p3"]))

    parts2 = []
    # p2: output column c takes input column (c+1)%2.
    # input col 0 appears at odd output columns; input col 1 at even output columns.
    for rp in (0, 1):
        mapping = [(0, 1), (1, 0)]  # (source_col, output_col_mod_2)
        for sc, cm in mapping:
            name = f"p2_r{rp}_s{sc}"
            n.append(helper.make_node("Mul", [f"v{rp}{sc}", f"m2_r{rp}_c{cm}"], [name]))
            parts2.append(name)

    while len(parts2) > 1:
        a = parts2.pop(0)
        b = parts2.pop(0)
        out = f"sum2_{len(parts2)}_{a}_{b}".replace("-", "_")[:60]
        n.append(helper.make_node("Add", [a, b], [out]))
        parts2.append(out)
    n.append(helper.make_node("Mul", [parts2[0], "p2"], ["branch2"]))

    parts3 = []
    # p3: output column c takes input column (c+1)%3.
    # source col0 -> output c mod3=2; source col1 -> c mod3=0; source col2 -> c mod3=1.
    for rp in (0, 1):
        mapping = [(0, 2), (1, 0), (2, 1)]
        for sc, cm in mapping:
            name = f"p3_r{rp}_s{sc}"
            n.append(helper.make_node("Mul", [f"v{rp}{sc}", f"m3_r{rp}_c{cm}"], [name]))
            parts3.append(name)

    while len(parts3) > 1:
        a = parts3.pop(0)
        b = parts3.pop(0)
        out = f"sum3_{len(parts3)}_{a}_{b}".replace("-", "_")[:60]
        n.append(helper.make_node("Add", [a, b], [out]))
        parts3.append(out)
    n.append(helper.make_node("Mul", [parts3[0], "p3"], ["branch3"]))

    n.append(helper.make_node("Add", ["branch2", "branch3"], ["full_pattern"]))
    n.append(helper.make_node("Mul", ["full_pattern", "valid"], ["output"]))

    graph = helper.make_graph(n, "task313", [x], [y], init)
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(model)
    return model


model = build_onnx_model()
