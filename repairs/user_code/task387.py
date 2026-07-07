# task387.py
# NeuroGolf / ARC-AGI task387
# Rule:
#   Four coloured singleton pixels form the corners of an axis-aligned rectangle,
#   with opposite corners sharing colours. Around each pixel, draw a 3x3 box in
#   the opposite colour while preserving the original center pixel. Connect the
#   four centers with colour 5 using the symmetric dashed-line rule.
#
# Checker-friendly: defines a top-level ONNX ModelProto named `model`.
# No internal onnx.save(...) and no __main__ dependency.

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

F = TensorProto.FLOAT
I64 = TensorProto.INT64


def _K(name, value, dtype):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def build_onnx_model():
    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

    R = np.arange(30, dtype=np.int64).reshape(1, 1, 30, 1)
    R = np.broadcast_to(R, (1, 1, 30, 30)).copy()
    C = np.arange(30, dtype=np.int64).reshape(1, 1, 1, 30)
    C = np.broadcast_to(C, (1, 1, 30, 30)).copy()

    Wring = np.ones((1, 1, 3, 3), dtype=np.float32)
    Wring[0, 0, 1, 1] = 0.0

    init = [
        _K("s0", [0], np.int64),
        _K("s1", [1], np.int64),
        _K("s10", [10], np.int64),
        _K("e10", [10], np.int64),
        _K("ax1", [1], np.int64),
        _K("rev30", list(range(29, -1, -1)), np.int64),
        _K("idx29", [29], np.int64),
        _K("one_i", [1], np.int64),
        _K("two_i", [2], np.int64),
        _K("zero_i", [0], np.int64),
        _K("zero_f", [0.0], np.float32),
        _K("R", R, np.int64),
        _K("C", C, np.int64),
        _K("Wring", Wring, np.float32),
    ]

    nodes = []

    # Valid area of the padded ARC grid: real cells sum to 1, padding sums to 0.
    nodes += [
        helper.make_node("ReduceSum", ["input"], ["valid"], axes=[1], keepdims=1),
        helper.make_node("Slice", ["input", "s1", "s10", "ax1"], ["fg9"]),
        helper.make_node("ReduceSum", ["fg9"], ["pts"], axes=[1], keepdims=1),
    ]

    # Rectangle geometry from the four non-background points.
    nodes += [
        helper.make_node("ReduceSum", ["pts"], ["row_sum"], axes=[3], keepdims=0),
        helper.make_node("ArgMax", ["row_sum"], ["top"], axis=2, keepdims=0),
        helper.make_node("Gather", ["row_sum", "rev30"], ["row_rev"], axis=2),
        helper.make_node("ArgMax", ["row_rev"], ["bot_rev"], axis=2, keepdims=0),
        helper.make_node("Sub", ["idx29", "bot_rev"], ["bot"]),
        helper.make_node("ReduceSum", ["pts"], ["col_sum"], axes=[2], keepdims=0),
        helper.make_node("ArgMax", ["col_sum"], ["left"], axis=2, keepdims=0),
        helper.make_node("Gather", ["col_sum", "rev30"], ["col_rev"], axis=2),
        helper.make_node("ArgMax", ["col_rev"], ["right_rev"], axis=2, keepdims=0),
        helper.make_node("Sub", ["idx29", "right_rev"], ["right"]),
    ]

    # Row/column equality and distances to rectangle sides.
    nodes += [
        helper.make_node("Equal", ["R", "top"], ["r_top"]),
        helper.make_node("Equal", ["R", "bot"], ["r_bot"]),
        helper.make_node("Or", ["r_top", "r_bot"], ["row_edge"]),
        helper.make_node("Equal", ["C", "left"], ["c_left"]),
        helper.make_node("Equal", ["C", "right"], ["c_right"]),
        helper.make_node("Or", ["c_left", "c_right"], ["col_edge"]),
        helper.make_node("Sub", ["C", "left"], ["dx_l"]),
        helper.make_node("Sub", ["right", "C"], ["dx_r"]),
        helper.make_node("Sub", ["R", "top"], ["dy_t"]),
        helper.make_node("Sub", ["bot", "R"], ["dy_b"]),
    ]

    # Horizontal dashed connectors:
    # inside the gap between 3x3 boxes, keep cells whose nearest endpoint
    # distance has even parity. Same rule for vertical connectors.
    nodes += [
        helper.make_node("Greater", ["dx_l", "one_i"], ["x_after_box"]),
        helper.make_node("Greater", ["dx_r", "one_i"], ["x_before_box"]),
        helper.make_node("And", ["x_after_box", "x_before_box"], ["x_gap"]),
        helper.make_node("Greater", ["dx_l", "dx_r"], ["dx_l_gt_r"]),
        helper.make_node("Not", ["dx_l_gt_r"], ["dx_l_le_r"]),
        helper.make_node("Greater", ["dx_r", "dx_l"], ["dx_r_gt_l"]),
        helper.make_node("Not", ["dx_r_gt_l"], ["dx_r_le_l"]),
        helper.make_node("Mod", ["dx_l", "two_i"], ["dx_l_mod"], fmod=0),
        helper.make_node("Mod", ["dx_r", "two_i"], ["dx_r_mod"], fmod=0),
        helper.make_node("Equal", ["dx_l_mod", "zero_i"], ["dx_l_even"]),
        helper.make_node("Equal", ["dx_r_mod", "zero_i"], ["dx_r_even"]),
        helper.make_node("And", ["dx_l_le_r", "dx_l_even"], ["h_from_l"]),
        helper.make_node("And", ["dx_r_le_l", "dx_r_even"], ["h_from_r"]),
        helper.make_node("Or", ["h_from_l", "h_from_r"], ["h_even"]),
        helper.make_node("And", ["row_edge", "x_gap"], ["h_a"]),
        helper.make_node("And", ["h_a", "h_even"], ["h_conn"]),
        helper.make_node("Greater", ["dy_t", "one_i"], ["y_after_box"]),
        helper.make_node("Greater", ["dy_b", "one_i"], ["y_before_box"]),
        helper.make_node("And", ["y_after_box", "y_before_box"], ["y_gap"]),
        helper.make_node("Greater", ["dy_t", "dy_b"], ["dy_t_gt_b"]),
        helper.make_node("Not", ["dy_t_gt_b"], ["dy_t_le_b"]),
        helper.make_node("Greater", ["dy_b", "dy_t"], ["dy_b_gt_t"]),
        helper.make_node("Not", ["dy_b_gt_t"], ["dy_b_le_t"]),
        helper.make_node("Mod", ["dy_t", "two_i"], ["dy_t_mod"], fmod=0),
        helper.make_node("Mod", ["dy_b", "two_i"], ["dy_b_mod"], fmod=0),
        helper.make_node("Equal", ["dy_t_mod", "zero_i"], ["dy_t_even"]),
        helper.make_node("Equal", ["dy_b_mod", "zero_i"], ["dy_b_even"]),
        helper.make_node("And", ["dy_t_le_b", "dy_t_even"], ["v_from_t"]),
        helper.make_node("And", ["dy_b_le_t", "dy_b_even"], ["v_from_b"]),
        helper.make_node("Or", ["v_from_t", "v_from_b"], ["v_even"]),
        helper.make_node("And", ["col_edge", "y_gap"], ["v_a"]),
        helper.make_node("And", ["v_a", "v_even"], ["v_conn"]),
        helper.make_node("Or", ["h_conn", "v_conn"], ["conn_bool"]),
        helper.make_node("Cast", ["conn_bool"], ["conn"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["conn", "valid"], ["conn_valid"]),
    ]

    out_ch = []
    for c in range(1, 10):
        sname = f"cs{c}"
        ename = f"ce{c}"
        init.append(_K(sname, [c], np.int64))
        init.append(_K(ename, [c + 1], np.int64))

        ch = f"ch{c}"
        notc = f"notc{c}"
        ring = f"ring{c}"
        ssum = f"sum{c}"
        pbool = f"present_bool{c}"
        present = f"present{c}"
        ringp = f"ringp{c}"
        block = f"block{c}"
        withconn = f"withconn{c}"
        valided = f"outc{c}"

        nodes += [
            helper.make_node("Slice", ["input", sname, ename, "ax1"], [ch]),
            helper.make_node("Sub", ["pts", ch], [notc]),
            helper.make_node("Conv", [notc, "Wring"], [ring], pads=[1, 1, 1, 1]),
            helper.make_node("ReduceSum", [ch], [ssum], axes=[2, 3], keepdims=1),
            helper.make_node("Greater", [ssum, "zero_f"], [pbool]),
            helper.make_node("Cast", [pbool], [present], to=TensorProto.FLOAT),
            helper.make_node("Mul", [ring, present], [ringp]),
            helper.make_node("Add", [ringp, ch], [block]),
        ]
        if c == 5:
            nodes.append(helper.make_node("Add", [block, "conn_valid"], [withconn]))
        else:
            nodes.append(helper.make_node("Identity", [block], [withconn]))
        nodes.append(helper.make_node("Mul", [withconn, "valid"], [valided]))
        out_ch.append(valided)

    nodes += [
        helper.make_node("Concat", out_ch, ["fg_out"], axis=1),
        helper.make_node("ReduceSum", ["fg_out"], ["occ"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["valid", "occ"], ["bg"]),
        helper.make_node("Concat", ["bg", "fg_out"], ["output"], axis=1),
    ]

    graph = helper.make_graph(nodes, "task387", [x], [y], init)
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model)
    return model


model = build_onnx_model()
