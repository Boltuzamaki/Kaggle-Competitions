"""Build an artifact-only compact exact model for NeuroGolf task 340.

The task has four colored frame sides.  Matching interior dots are moved to the
cell next to their same-colored side; all other interior cells become black.
The model keeps the repaired model's CP reconstruction, removes redundant
threshold work, and replaces dynamic ten-channel color vectors with a compact
quadratic color code.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "other_model_onnx" / "task340.onnx"

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
I32 = TensorProto.INT32
I64 = TensorProto.INT64


def init(name, value, dtype=None):
    a = np.asarray(value, dtype=dtype)
    return numpy_helper.from_array(a, name=name)


def build_model():
    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F16, [1, 10, 30, 30])

    idx = np.arange(30, dtype=np.float16)[None, :]
    delta01 = np.zeros((1, 30), dtype=np.float16)
    delta01[0, 0], delta01[0, 1] = 1, -1
    pos1 = np.zeros((1, 30), dtype=np.float16)
    pos1[0, 1] = 1
    color_index = np.arange(10, dtype=np.float32)
    decoder = np.stack([
        0.25 - color_index * color_index,
        2.0 * color_index,
        -np.ones(10, dtype=np.float32),
    ], axis=1).astype(np.float16)
    side_mix = np.zeros((5, 9), dtype=np.float16)
    side_mix[0, 0] = 1
    for side, terms in {1: (1, 5), 2: (2, 6), 3: (3, 7), 4: (4, 8)}.items():
        side_mix[side, list(terms)] = 1

    initializers = [
        init("idx", idx),
        init("delta01", delta01),
        init("pos1", pos1),
        init("color_index", color_index),
        init("decoder", decoder),
        init("side_mix", side_mix),
        init("base_gate", np.asarray([[[1], [0], [0], [0], [0]]], dtype=np.float16)),
        init("zero_base", [[[0]]], np.float16),
        init("one", [[1]], np.float16),
        init("zero_f32", [[0]], np.float32),
        init("two_f32", [[2]], np.float32),
        init("top_starts", [0, 1], np.int32),
        init("top_ends", [1, 2], np.int32),
        init("left_starts", [1, 0], np.int32),
        init("left_ends", [2, 1], np.int32),
        init("spatial_axes", [2, 3], np.int32),
        init("i1", [1], np.int32),
        init("i2", [2], np.int32),
    ]

    n = []
    add = n.append

    # Top/left colors are at fixed cells.  Their color projections also give
    # exact masks for the non-corner frame spans.
    add(helper.make_node("Slice", ["input", "top_starts", "top_ends", "spatial_axes"], ["top_color_f32"]))
    add(helper.make_node("Slice", ["input", "left_starts", "left_ends", "spatial_axes"], ["left_color_f32"]))
    add(helper.make_node("Einsum", ["input", "top_color_f32"], ["top_counts_f32"], equation="bchw,bcpq->bw"))
    add(helper.make_node("Einsum", ["input", "left_color_f32"], ["left_counts_f32"], equation="bchw,bcpq->bh"))
    add(helper.make_node("Greater", ["top_counts_f32", "zero_f32"], ["inner_cols_bool"]))
    add(helper.make_node("Cast", ["inner_cols_bool"], ["inner_cols"], to=F16))
    add(helper.make_node("Greater", ["left_counts_f32", "zero_f32"], ["inner_rows_bool"]))
    add(helper.make_node("Cast", ["inner_rows_bool"], ["inner_rows"], to=F16))

    # Interior sizes determine the dynamic last and penultimate coordinates.
    add(helper.make_node("ReduceSum", ["inner_cols"], ["width_m2"], axes=[1], keepdims=1))
    add(helper.make_node("ReduceSum", ["inner_rows"], ["height_m2"], axes=[1], keepdims=1))
    add(helper.make_node("Add", ["width_m2", "one"], ["width_m1"]))
    add(helper.make_node("Add", ["height_m2", "one"], ["height_m1"]))
    add(helper.make_node("Add", ["width_m1", "one"], ["width_real_f16"]))
    add(helper.make_node("Add", ["height_m1", "one"], ["height_real_f16"]))

    add(helper.make_node("Equal", ["idx", "width_m1"], ["last_col_bool"]))
    add(helper.make_node("Cast", ["last_col_bool"], ["last_col"], to=F16))
    add(helper.make_node("Equal", ["idx", "width_m2"], ["penult_col_bool"]))
    add(helper.make_node("Cast", ["penult_col_bool"], ["penult_col"], to=F16))
    add(helper.make_node("Equal", ["idx", "height_m1"], ["last_row_bool"]))
    add(helper.make_node("Cast", ["last_row_bool"], ["last_row"], to=F16))
    add(helper.make_node("Equal", ["idx", "height_m2"], ["penult_row_bool"]))
    add(helper.make_node("Cast", ["penult_row_bool"], ["penult_row"], to=F16))
    add(helper.make_node("Less", ["idx", "width_real_f16"], ["real_cols_bool"]))
    add(helper.make_node("Cast", ["real_cols_bool"], ["real_cols"], to=F16))
    add(helper.make_node("Less", ["idx", "height_real_f16"], ["real_rows_bool"]))
    add(helper.make_node("Cast", ["real_rows_bool"], ["real_rows"], to=F16))

    # Build dynamic slice coordinates for the right and bottom frame colors.
    add(helper.make_node("Squeeze", ["width_m1"], ["width_m1_vec"], axes=[0]))
    add(helper.make_node("Cast", ["width_m1_vec"], ["right_col_i32"], to=I32))
    add(helper.make_node("Add", ["right_col_i32", "i1"], ["right_col_end_i32"]))
    add(helper.make_node("Squeeze", ["height_m1"], ["height_m1_vec"], axes=[0]))
    add(helper.make_node("Cast", ["height_m1_vec"], ["bottom_row_i32"], to=I32))
    add(helper.make_node("Add", ["bottom_row_i32", "i1"], ["bottom_row_end_i32"]))
    add(helper.make_node("Concat", ["i1", "right_col_i32"], ["right_starts"], axis=0))
    add(helper.make_node("Concat", ["i2", "right_col_end_i32"], ["right_ends"], axis=0))
    add(helper.make_node("Concat", ["bottom_row_i32", "i1"], ["bottom_starts"], axis=0))
    add(helper.make_node("Concat", ["bottom_row_end_i32", "i2"], ["bottom_ends"], axis=0))
    add(helper.make_node("Slice", ["input", "right_starts", "right_ends", "spatial_axes"], ["right_color_f32"]))
    add(helper.make_node("Slice", ["input", "bottom_starts", "bottom_ends", "spatial_axes"], ["bottom_color_f32"]))

    # Projections count the frame occurrence plus matching dots.  For the fixed
    # top/left destinations, raw counts cancel the frame baseline algebraically;
    # the dynamic bottom/right destinations retain their cheaper boolean masks.
    add(helper.make_node("Einsum", ["input", "bottom_color_f32"], ["bottom_counts_f32"], equation="bchw,bcpq->bw"))
    add(helper.make_node("Einsum", ["input", "right_color_f32"], ["right_counts_f32"], equation="bchw,bcpq->bh"))
    add(helper.make_node("Cast", ["top_counts_f32"], ["top_counts"], to=F16))
    add(helper.make_node("GreaterOrEqual", ["bottom_counts_f32", "two_f32"], ["bottom_hits_bool"]))
    add(helper.make_node("Cast", ["bottom_hits_bool"], ["bottom_hits"], to=F16))
    add(helper.make_node("Cast", ["left_counts_f32"], ["left_counts"], to=F16))
    add(helper.make_node("GreaterOrEqual", ["right_counts_f32", "two_f32"], ["right_hits_bool"]))
    add(helper.make_node("Cast", ["right_hits_bool"], ["right_hits"], to=F16))

    # Rank-nine row/column factors.  The 5x9 map selects the frame side used by
    # each spatial term; the quadratic decoder handles black-to-color changes.
    add(helper.make_node(
        "Concat",
        ["real_rows", "delta01", "last_row", "inner_rows", "inner_rows", "pos1", "penult_row", "left_counts", "right_hits"],
        ["row_factors"], axis=0,
    ))
    add(helper.make_node(
        "Concat",
        ["real_cols", "inner_cols", "inner_cols", "delta01", "last_col", "top_counts", "bottom_hits", "pos1", "penult_col"],
        ["col_factors"], axis=0,
    ))

    # Encode each side color by its scalar ARC index.  The decoder implements
    # Q_0(c)=0.25-c^2 for black and Q_t(c)-Q_0(c)=2ct-t^2 for a side color t.
    # This replaces five dynamic ten-channel vectors with three tiny features.
    for side in ("top", "bottom", "left", "right"):
        add(helper.make_node(
            "Einsum", [f"{side}_color_f32", "color_index"], [f"{side}_index_f32"],
            equation="bcij,c->bij",
        ))
        add(helper.make_node("Cast", [f"{side}_index_f32"], [f"{side}_index"], to=F16))
    add(helper.make_node(
        "Concat", ["top_index", "bottom_index", "left_index", "right_index"],
        ["side_index"], axis=1,
    ))
    add(helper.make_node("Mul", ["side_index", "side_index"], ["side_index_sq"]))
    add(helper.make_node("Concat", ["zero_base", "side_index"], ["index_feature"], axis=1))
    add(helper.make_node("Concat", ["zero_base", "side_index_sq"], ["square_feature"], axis=1))
    add(helper.make_node(
        "Concat", ["base_gate", "index_feature", "square_feature"],
        ["features"], axis=2,
    ))
    add(helper.make_node(
        "Einsum", ["features", "decoder", "row_factors", "col_factors", "side_mix"], ["output"],
        equation="bsd,cd,kr,kw,sk->bcrw",
    ))

    dynamic_slice_info = [
        helper.make_tensor_value_info("right_color_f32", F32, [1, 10, 1, 1]),
        helper.make_tensor_value_info("bottom_color_f32", F32, [1, 10, 1, 1]),
    ]
    graph = helper.make_graph(
        n, "task340_compact", [x], [y], initializer=initializers,
        value_info=dynamic_slice_info,
    )
    model = helper.make_model(graph, producer_name="codex-task340", opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    model = build_model()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUT)
    print(OUT)
