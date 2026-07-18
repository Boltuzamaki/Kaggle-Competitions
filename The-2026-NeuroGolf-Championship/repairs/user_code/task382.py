
# task382.py
# Checker-friendly ONNX model: exposes top-level `model`.
#
# Rule:
# - Color 8 gives a 1D seed pattern on one border.
# - Color 2 markers sit on a perpendicular border.
# - Copy the 8-pattern across the canvas; each crossed 2-marker shifts it by 1
#   away from the marker side. Keep 2 markers unchanged.

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

    init = []
    nodes = []

    used_initializer_names = set()

    def K(name, value, dtype=np.int64):
        base = name
        if name in used_initializer_names:
            i = 1
            while f"{base}_{i}" in used_initializer_names:
                i += 1
            name = f"{base}_{i}"
        used_initializer_names.add(name)
        init.append(_K(name, value, dtype))
        return name

    # Common constants
    K("s0", [0])
    K("s1", [1])
    K("s2", [2])
    K("s8", [8])
    K("e1", [1])
    K("e3", [3])
    K("e9", [9])
    K("e29", [29])
    K("e30", [30])
    K("ax1", [1])
    K("ax2", [2])
    K("ax3", [3])
    K("rev30", list(range(29, -1, -1)))
    K("axis2_scalar", np.array(2, dtype=np.int64))
    K("axis3_scalar", np.array(3, dtype=np.int64))
    K("zero", [0.0], np.float32)

    # Pad constants for shifting row/column validity.
    K("pad_next_row", [0, 0, 0, 0, 0, 0, 1, 0])
    K("pad_next_col", [0, 0, 0, 0, 0, 0, 0, 1])

    # Channel slices.
    nodes += [
        helper.make_node("Slice", ["input", "s8", "e9", "ax1"], ["ch8"]),
        helper.make_node("Slice", ["input", "s2", "e3", "ax1"], ["ch2"]),
        helper.make_node("ReduceSum", ["input"], ["valid"], axes=[1], keepdims=1),
    ]

    # Valid bottom/right border masks for variable H/W inside 30x30 padded tensor.
    nodes += [
        helper.make_node("ReduceMax", ["valid"], ["row_valid"], axes=[3], keepdims=1),
        helper.make_node("Slice", ["row_valid", "s1", "e30", "ax2"], ["row_valid_tail"]),
        helper.make_node("Pad", ["row_valid_tail", "pad_next_row", "zero"], ["row_valid_next"]),
        helper.make_node("Sub", ["row_valid", "row_valid_next"], ["bottom_row_mask"]),

        helper.make_node("ReduceMax", ["valid"], ["col_valid"], axes=[2], keepdims=1),
        helper.make_node("Slice", ["col_valid", "s1", "e30", "ax3"], ["col_valid_tail"]),
        helper.make_node("Pad", ["col_valid_tail", "pad_next_col", "zero"], ["col_valid_next"]),
        helper.make_node("Sub", ["col_valid", "col_valid_next"], ["right_col_mask"]),
    ]

    # Source pattern vectors on each side.
    nodes += [
        helper.make_node("Slice", ["ch8", "s0", "e1", "ax2"], ["top8_vec"]),
        helper.make_node("Mul", ["ch8", "bottom_row_mask"], ["bottom8_full"]),
        helper.make_node("ReduceSum", ["bottom8_full"], ["bottom8_vec"], axes=[2], keepdims=1),

        helper.make_node("Slice", ["ch8", "s0", "e1", "ax3"], ["left8_vec"]),
        helper.make_node("Mul", ["ch8", "right_col_mask"], ["right8_full"]),
        helper.make_node("ReduceSum", ["right8_full"], ["right8_vec"], axes=[3], keepdims=1),
    ]

    # Marker vectors on each side.
    nodes += [
        helper.make_node("Slice", ["ch2", "s0", "e1", "ax2"], ["top2_vec"]),
        helper.make_node("Mul", ["ch2", "bottom_row_mask"], ["bottom2_full"]),
        helper.make_node("ReduceSum", ["bottom2_full"], ["bottom2_vec"], axes=[2], keepdims=1),

        helper.make_node("Slice", ["ch2", "s0", "e1", "ax3"], ["left2_vec"]),
        helper.make_node("Mul", ["ch2", "right_col_mask"], ["right2_full"]),
        helper.make_node("ReduceSum", ["right2_full"], ["right2_vec"], axes=[3], keepdims=1),
    ]

    # Side presence gates: in all examples, marker side is unique and the source side is
    # the perpendicular side with nonzero color-8 presence.
    for name in ["top8_vec", "bottom8_vec", "left8_vec", "right8_vec",
                 "top2_vec", "bottom2_vec", "left2_vec", "right2_vec"]:
        nodes += [
            helper.make_node("ReduceSum", [name], [name + "_cnt"], axes=[2, 3], keepdims=1),
            helper.make_node("Greater", [name + "_cnt", "zero"], [name + "_has_bool"]),
            helper.make_node("Cast", [name + "_has_bool"], [name + "_has"], to=TensorProto.FLOAT),
        ]

    def sum_nodes(inputs, out):
        if len(inputs) == 1:
            nodes.append(helper.make_node("Identity", [inputs[0]], [out]))
        else:
            nodes.append(helper.make_node("Sum", inputs, [out]))

    def shift_h(vec, t, direction, out):
        # vec: [1,1,1,30], direction +1 => right shift, -1 => left shift
        if t == 0:
            nodes.append(helper.make_node("Identity", [vec], [out]))
            return
        if direction > 0:
            pad = K(f"pad_hr_{t}", [0, 0, 0, t, 0, 0, 0, 0])
            tmp = f"{out}_pad"
            nodes.append(helper.make_node("Pad", [vec, pad, "zero"], [tmp]))
            nodes.append(helper.make_node("Slice", [tmp, "s0", "e30", "ax3"], [out]))
        else:
            pad = K(f"pad_hl_{t}", [0, 0, 0, 0, 0, 0, 0, t])
            st = K(f"st_hl_{t}", [t])
            en = K(f"en_hl_{t}", [t + 30])
            tmp = f"{out}_pad"
            nodes.append(helper.make_node("Pad", [vec, pad, "zero"], [tmp]))
            nodes.append(helper.make_node("Slice", [tmp, st, en, "ax3"], [out]))

    def shift_v(vec, t, direction, out):
        # vec: [1,1,30,1], direction +1 => down shift, -1 => up shift
        if t == 0:
            nodes.append(helper.make_node("Identity", [vec], [out]))
            return
        if direction > 0:
            pad = K(f"pad_vd_{t}", [0, 0, t, 0, 0, 0, 0, 0])
            tmp = f"{out}_pad"
            nodes.append(helper.make_node("Pad", [vec, pad, "zero"], [tmp]))
            nodes.append(helper.make_node("Slice", [tmp, "s0", "e30", "ax2"], [out]))
        else:
            pad = K(f"pad_vu_{t}", [0, 0, 0, 0, 0, 0, t, 0])
            st = K(f"st_vu_{t}", [t])
            en = K(f"en_vu_{t}", [t + 30])
            tmp = f"{out}_pad"
            nodes.append(helper.make_node("Pad", [vec, pad, "zero"], [tmp]))
            nodes.append(helper.make_node("Slice", [tmp, st, en, "ax2"], [out]))

    def horizontal_candidate(base_vec, marker_vec, source_from_top, marker_from_left, prefix):
        # Cumulative marker count per row.
        if source_from_top:
            nodes.append(helper.make_node("CumSum", [marker_vec, "axis2_scalar"], [prefix + "_k"]))
            kname = prefix + "_k"
        else:
            nodes.append(helper.make_node("Gather", [marker_vec, "rev30"], [prefix + "_mr"], axis=2))
            nodes.append(helper.make_node("CumSum", [prefix + "_mr", "axis2_scalar"], [prefix + "_kr"]))
            nodes.append(helper.make_node("Gather", [prefix + "_kr", "rev30"], [prefix + "_k"], axis=2))
            kname = prefix + "_k"

        terms = []
        direction = 1 if marker_from_left else -1
        for t in range(30):
            kt = K(f"{prefix}_kt_{t}", [float(t)], np.float32)
            eq = f"{prefix}_eq_{t}"
            mask = f"{prefix}_mask_{t}"
            sh = f"{prefix}_shift_{t}"
            term = f"{prefix}_term_{t}"
            nodes.append(helper.make_node("Equal", [kname, kt], [eq]))
            nodes.append(helper.make_node("Cast", [eq], [mask], to=TensorProto.FLOAT))
            shift_h(base_vec, t, direction, sh)
            nodes.append(helper.make_node("Mul", [mask, sh], [term]))
            terms.append(term)
        sum_nodes(terms, prefix + "_cand")
        return prefix + "_cand"

    def vertical_candidate(base_vec, marker_vec, source_from_left, marker_from_top, prefix):
        # Cumulative marker count per column.
        if source_from_left:
            nodes.append(helper.make_node("CumSum", [marker_vec, "axis3_scalar"], [prefix + "_k"]))
            kname = prefix + "_k"
        else:
            nodes.append(helper.make_node("Gather", [marker_vec, "rev30"], [prefix + "_mr"], axis=3))
            nodes.append(helper.make_node("CumSum", [prefix + "_mr", "axis3_scalar"], [prefix + "_kr"]))
            nodes.append(helper.make_node("Gather", [prefix + "_kr", "rev30"], [prefix + "_k"], axis=3))
            kname = prefix + "_k"

        terms = []
        direction = 1 if marker_from_top else -1
        for t in range(30):
            kt = K(f"{prefix}_kt_{t}", [float(t)], np.float32)
            eq = f"{prefix}_eq_{t}"
            mask = f"{prefix}_mask_{t}"
            sh = f"{prefix}_shift_{t}"
            term = f"{prefix}_term_{t}"
            nodes.append(helper.make_node("Equal", [kname, kt], [eq]))
            nodes.append(helper.make_node("Cast", [eq], [mask], to=TensorProto.FLOAT))
            shift_v(base_vec, t, direction, sh)
            nodes.append(helper.make_node("Mul", [mask, sh], [term]))
            terms.append(term)
        sum_nodes(terms, prefix + "_cand")
        return prefix + "_cand"

    gated = []

    # top/bottom source with left/right markers.
    specs_h = [
        ("top8_vec", "left2_vec", True, True, "top_left"),
        ("top8_vec", "right2_vec", True, False, "top_right"),
        ("bottom8_vec", "left2_vec", False, True, "bottom_left"),
        ("bottom8_vec", "right2_vec", False, False, "bottom_right"),
    ]
    for base, mark, src_top, mark_left, pref in specs_h:
        cand = horizontal_candidate(base, mark, src_top, mark_left, pref)
        g1 = base + "_has"
        g2 = mark + "_has"
        g = pref + "_gate"
        gout = pref + "_gated"
        nodes.append(helper.make_node("Mul", [g1, g2], [g]))
        nodes.append(helper.make_node("Mul", [cand, g], [gout]))
        gated.append(gout)

    # left/right source with top/bottom markers.
    specs_v = [
        ("left8_vec", "top2_vec", True, True, "left_top"),
        ("left8_vec", "bottom2_vec", True, False, "left_bottom"),
        ("right8_vec", "top2_vec", False, True, "right_top"),
        ("right8_vec", "bottom2_vec", False, False, "right_bottom"),
    ]
    for base, mark, src_left, mark_top, pref in specs_v:
        cand = vertical_candidate(base, mark, src_left, mark_top, pref)
        g1 = base + "_has"
        g2 = mark + "_has"
        g = pref + "_gate"
        gout = pref + "_gated"
        nodes.append(helper.make_node("Mul", [g1, g2], [g]))
        nodes.append(helper.make_node("Mul", [cand, g], [gout]))
        gated.append(gout)

    nodes.append(helper.make_node("Sum", gated, ["gen8_raw"]))
    nodes.append(helper.make_node("Mul", ["gen8_raw", "valid"], ["gen8_valid"]))
    nodes.append(helper.make_node("Sub", ["valid", "ch2"], ["not_red"]))
    nodes.append(helper.make_node("Mul", ["gen8_valid", "not_red"], ["ch8_out"]))

    nodes.append(helper.make_node("Sub", ["valid", "ch2"], ["bg_tmp"]))
    nodes.append(helper.make_node("Sub", ["bg_tmp", "ch8_out"], ["ch0"]))

    nodes.append(helper.make_node("Mul", ["ch0", "zero"], ["zch"]))
    channel_list = ["ch0", "zch", "ch2", "zch", "zch", "zch", "zch", "zch", "ch8_out", "zch"]
    nodes.append(helper.make_node("Concat", channel_list, ["output"], axis=1))

    graph = helper.make_graph(nodes, "task382", [x], [y], init)
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model)
    return model


model = build_onnx_model()
