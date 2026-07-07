# task400.py
# Checker-friendly: exposes top-level `model`.

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

F = TensorProto.FLOAT


def _K(name, value, dtype):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def build_onnx_model():
    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

    init = [
        _K("s0", [0], np.int64),
        _K("s1", [1], np.int64),
        _K("e2", [2], np.int64),
        _K("e24", [24], np.int64),
        _K("ax1", [1], np.int64),
        _K("ax2", [2], np.int64),
        _K("ax3", [3], np.int64),
        _K("axes23", [2, 3], np.int64),
        _K("rev24", list(range(23, -1, -1)), np.int64),
        _K("W5", np.ones((1, 1, 5, 5), dtype=np.float32), np.float32),
        _K("thr", np.array([24.5], dtype=np.float32), np.float32),
        _K("pads_out", [0, 0, 0, 0, 0, 0, 25, 25], np.int64),
        _K("zero", [0.0], np.float32),
    ]

    nodes = [
        # Real 24x24 ARC input from padded 30x30 tensor.
        helper.make_node("Slice", ["input", "s0", "e24", "ax2"], ["x_r24"]),
        helper.make_node("Slice", ["x_r24", "s0", "e24", "ax3"], ["x24"]),

        # Locate unique 5x5 color-1 mask.
        helper.make_node("Slice", ["x24", "s1", "e2", "ax1"], ["ch1"]),
        helper.make_node("Conv", ["ch1", "W5"], ["sum5"]),
        helper.make_node("Greater", ["sum5", "thr"], ["sel_bool"]),
        helper.make_node("Cast", ["sel_bool"], ["sel"], to=TensorProto.FLOAT),

        # 180-degree rotate real input.
        helper.make_node("Gather", ["x24", "rev24"], ["rot_r"], axis=2),
        helper.make_node("Gather", ["rot_r", "rev24"], ["rot24"], axis=3),
    ]

    row_names = []
    for i in range(5):
        cell_names = []
        for j in range(5):
            sname = f"s_{i}_{j}"
            ename = f"e_{i}_{j}"
            init.append(_K(sname, [i, j], np.int64))
            init.append(_K(ename, [i + 20, j + 20], np.int64))

            sl = f"sl_{i}_{j}"
            mul = f"mul_{i}_{j}"
            red = f"red_{i}_{j}"
            cell = f"cell_{i}_{j}"

            nodes += [
                helper.make_node("Slice", ["rot24", sname, ename, "axes23"], [sl]),
                helper.make_node("Mul", [sl, "sel"], [mul]),
                helper.make_node("ReduceSum", [mul], [red], axes=[2, 3], keepdims=0),
                helper.make_node("Unsqueeze", [red], [cell], axes=[2, 3]),
            ]
            cell_names.append(cell)

        row = f"row_{i}"
        nodes.append(helper.make_node("Concat", cell_names, [row], axis=3))
        row_names.append(row)

    nodes.append(helper.make_node("Concat", row_names, ["patch5"], axis=2))

    # Pad 5x5 one-hot output to checker shape [1,10,30,30].
    # Outside the true output area all channels stay zero.
    nodes.append(helper.make_node("Pad", ["patch5", "pads_out", "zero"], ["output"]))

    graph = helper.make_graph(nodes, "task400", [x], [y], init)
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model)
    return model


model = build_onnx_model()