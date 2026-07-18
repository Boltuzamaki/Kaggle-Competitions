import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np


def build_model():
    nodes = []
    inits = []
    counter = [0]

    def name(prefix):
        counter[0] += 1
        return f"{prefix}_{counter[0]}"

    def const(arr, prefix, dtype=None):
        n = name(prefix)
        a = np.asarray(arr, dtype=dtype)
        inits.append(numpy_helper.from_array(a, n))
        return n

    def node(op, inputs, outputs=None, **attrs):
        if outputs is None:
            outputs = [name(op.lower())]
        nodes.append(helper.make_node(op, inputs, outputs, **attrs))
        return outputs[0] if len(outputs) == 1 else outputs

    x = "input"
    zero_i = const(np.array(0, np.int64), "zero_i")
    one_i = const(np.array(1, np.int64), "one_i")
    two_i = const(np.array(2, np.int64), "two_i")
    thirty_i = const(np.array(30, np.int64), "thirty_i")
    big_i = const(np.array(99, np.int64), "big_i")
    neg_i = const(np.array(-1, np.int64), "neg_i")
    half_f = const(np.array(0.5, np.float32), "half_f")
    z_f = const(np.array(0.0, np.float32), "z_f")

    starts = const([0, 0, 0, 0], "starts", np.int64)
    ends = const([1, 1, 30, 30], "ends", np.int64)
    axes4 = const([0, 1, 2, 3], "axes4", np.int64)
    steps = const([1, 1, 1, 1], "steps", np.int64)
    axes_all = [0, 1, 2, 3]
    axes_sp = [2, 3]

    rows = np.arange(30, dtype=np.int64).reshape(1, 1, 30, 1)
    cols = np.arange(30, dtype=np.int64).reshape(1, 1, 1, 30)
    row_idx = const(np.broadcast_to(rows, (1, 1, 30, 30)), "row_idx")
    col_idx = const(np.broadcast_to(cols, (1, 1, 30, 30)), "col_idx")
    row2d = const(np.broadcast_to(np.arange(30, dtype=np.int64).reshape(30, 1), (30, 30)), "row2d")
    col2d = const(np.broadcast_to(np.arange(30, dtype=np.int64).reshape(1, 30), (30, 30)), "col2d")

    def ch(k):
        st = const([0, k, 0, 0], f"st{k}", np.int64)
        en = const([1, k + 1, 30, 30], f"en{k}", np.int64)
        return node("Slice", [x, st, en, axes4, steps])

    chans = [ch(k) for k in range(10)]
    non8_sum = node("Add", [node("ReduceSum", [node("Concat", chans[1:8], axis=1)], axes=[1], keepdims=1),
                            chans[9]])
    non8 = node("Greater", [non8_sum, half_f])
    marker8 = node("Greater", [chans[8], half_f])

    def masked_minmax(mask):
        min_src = node("Where", [mask, row_idx, const(np.full((1, 1, 30, 30), 99, np.int64), "biggrid")])
        max_src = node("Where", [mask, row_idx, const(np.full((1, 1, 30, 30), -1, np.int64), "neggrid")])
        rmin = node("ReduceMin", [min_src], axes=axes_all, keepdims=0)
        rmax = node("ReduceMax", [max_src], axes=axes_all, keepdims=0)
        min_src = node("Where", [mask, col_idx, const(np.full((1, 1, 30, 30), 99, np.int64), "biggrid")])
        max_src = node("Where", [mask, col_idx, const(np.full((1, 1, 30, 30), -1, np.int64), "neggrid")])
        cmin = node("ReduceMin", [min_src], axes=axes_all, keepdims=0)
        cmax = node("ReduceMax", [max_src], axes=axes_all, keepdims=0)
        return rmin, rmax, cmin, cmax

    fr0, fr1, fc0, fc1 = masked_minmax(non8)
    mr0, _mr1, mc0, _mc1 = masked_minmax(marker8)
    out_h = node("Add", [node("Sub", [fr1, fr0]), one_i])
    out_w = node("Add", [node("Sub", [fc1, fc0]), one_i])
    last_r = node("Sub", [out_h, one_i])
    last_c = node("Sub", [out_w, one_i])
    inner_last_r = node("Sub", [out_h, two_i])
    inner_last_c = node("Sub", [out_w, two_i])

    r_eq0 = node("Equal", [row_idx, zero_i])
    c_eq0 = node("Equal", [col_idx, zero_i])
    r_eqlast = node("Equal", [row_idx, last_r])
    c_eqlast = node("Equal", [col_idx, last_c])
    r_ge1 = node("Greater", [row_idx, zero_i])
    c_ge1 = node("Greater", [col_idx, zero_i])
    r_gt1 = node("Greater", [row_idx, one_i])
    c_gt1 = node("Greater", [col_idx, one_i])
    r_ltlast = node("Less", [row_idx, last_r])
    c_ltlast = node("Less", [col_idx, last_c])
    r_ltinnerlast = node("Less", [row_idx, inner_last_r])
    c_ltinnerlast = node("Less", [col_idx, inner_last_c])
    r_eq1 = node("Equal", [row_idx, one_i])
    c_eq1 = node("Equal", [col_idx, one_i])
    r_eqinnerlast = node("Equal", [row_idx, inner_last_r])
    c_eqinnerlast = node("Equal", [col_idx, inner_last_c])
    valid_region = node("And", [node("Less", [row_idx, out_h]), node("Less", [col_idx, out_w])])

    top_line = node("And", [node("And", [r_eq0, c_ge1]), c_ltlast])
    bottom_line = node("And", [node("And", [r_eqlast, c_ge1]), c_ltlast])
    left_line = node("And", [node("And", [c_eq0, r_ge1]), r_ltlast])
    right_line = node("And", [node("And", [c_eqlast, r_ge1]), r_ltlast])
    interior = node("And", [node("And", [r_ge1, r_ltlast]), node("And", [c_ge1, c_ltlast])])

    m8_2d = node("Squeeze", [chans[8]], axes=[0, 1])
    src_r = node("Add", [mr0, node("Sub", [row2d, one_i])])
    src_c = node("Add", [mc0, node("Sub", [col2d, one_i])])
    src_r = node("Max", [zero_i, node("Min", [src_r, node("Sub", [thirty_i, one_i])])])
    src_c = node("Max", [zero_i, node("Min", [src_c, node("Sub", [thirty_i, one_i])])])
    src_r_u = node("Unsqueeze", [src_r], axes=[2])
    src_c_u = node("Unsqueeze", [src_c], axes=[2])
    gather_idx = node("Concat", [src_r_u, src_c_u], axis=2)
    gathered = node("GatherND", [m8_2d, gather_idx])
    gathered4 = node("Unsqueeze", [node("Unsqueeze", [gathered], axes=[0])], axes=[0])
    marker_out = node("And", [node("Greater", [gathered4, half_f]), interior])

    d_top = node("Sub", [row_idx, one_i])
    d_bottom = node("Sub", [inner_last_r, row_idx])
    d_left = node("Sub", [col_idx, one_i])
    d_right = node("Sub", [inner_last_c, col_idx])
    top_edge = node("And", [marker_out, node("And", [
        node("And", [node("Less", [d_top, d_bottom]), node("Less", [d_top, d_left])]),
        node("Less", [d_top, d_right]),
    ])])
    bottom_edge = node("And", [marker_out, node("And", [
        node("And", [node("Less", [d_bottom, d_top]), node("Less", [d_bottom, d_left])]),
        node("Less", [d_bottom, d_right]),
    ])])
    left_edge = node("And", [marker_out, node("And", [
        node("And", [node("Less", [d_left, d_top]), node("Less", [d_left, d_bottom])]),
        node("Less", [d_left, d_right]),
    ])])
    right_edge = node("And", [marker_out, node("And", [
        node("And", [node("Less", [d_right, d_top]), node("Less", [d_right, d_bottom])]),
        node("Less", [d_right, d_left]),
    ])])

    top_assign = node("Or", [top_line, top_edge])
    bottom_assign = node("Or", [bottom_line, bottom_edge])
    left_assign = node("Or", [left_line, left_edge])
    right_assign = node("Or", [right_line, right_edge])
    edge_any = node("Or", [node("Or", [top_edge, bottom_edge]), node("Or", [left_edge, right_edge])])
    keep8 = node("And", [marker_out, node("Not", [edge_any])])

    row_is_fr0 = node("Equal", [row_idx, fr0])
    row_is_fr1 = node("Equal", [row_idx, fr1])
    col_is_fc0 = node("Equal", [col_idx, fc0])
    col_is_fc1 = node("Equal", [col_idx, fc1])

    out_ch = []
    colored = []
    for k in range(1, 10):
        if k == 8:
            ok = keep8
        else:
            top_k = node("Greater", [node("ReduceMax", [node("Where", [row_is_fr0, chans[k], z_f])],
                                          axes=axes_sp, keepdims=1), half_f])
            bottom_k = node("Greater", [node("ReduceMax", [node("Where", [row_is_fr1, chans[k], z_f])],
                                             axes=axes_sp, keepdims=1), half_f])
            left_k = node("Greater", [node("ReduceMax", [node("Where", [col_is_fc0, chans[k], z_f])],
                                           axes=axes_sp, keepdims=1), half_f])
            right_k = node("Greater", [node("ReduceMax", [node("Where", [col_is_fc1, chans[k], z_f])],
                                            axes=axes_sp, keepdims=1), half_f])
            ok = node("Or", [
                node("Or", [node("And", [top_assign, top_k]), node("And", [bottom_assign, bottom_k])]),
                node("Or", [node("And", [left_assign, left_k]), node("And", [right_assign, right_k])]),
            ])
        colored.append(ok)
        out_ch.append(node("Cast", [ok], to=TensorProto.FLOAT))

    any_color = colored[0]
    for m in colored[1:]:
        any_color = node("Or", [any_color, m])
    bg = node("Cast", [node("And", [valid_region, node("Not", [any_color])])], to=TensorProto.FLOAT)
    output = node("Concat", [bg] + out_ch, ["output"], axis=1)

    graph = helper.make_graph(
        nodes,
        "task238_pure_logic",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 12)])


model = build_model()


if __name__ == "__main__":
    import os
    onnx.save(model, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "task238.onnx")))
