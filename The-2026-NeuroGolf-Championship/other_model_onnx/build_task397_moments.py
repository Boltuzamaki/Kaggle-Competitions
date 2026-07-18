"""Build a moment-based exact rewrite for NeuroGolf task397.

Generator invariants verified across all 266 supplied examples:

* every object is a solid 2x2 non-background block;
* object column spans never overlap, although two spans may touch;
* top-left columns are in 0..8 and each occupied physical column has exactly
  two vertically adjacent object cells.

Three column moments are therefore sufficient.  If a column contains colors
``p, q`` at rows ``r, r+1``, then

    S = p + q
    T = r*p + (r+1)*q
    r = floor(T/S), q = T mod S, p = S-q

The row-only moment is identical for the two columns of an object, and differs
between touching objects (otherwise they would be one connected component).
This replaces the repair's cropped color-id image, three shifted 7x9 views,
spatial Min, and 63-cell TopK with three width-30 Einsums and a width-9 TopK.
The original repair is read-only; this script writes an artifact-only variant.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repairs" / "task397.onnx"
OUTPUT = ROOT / "other_model_onnx" / "task397_moments.onnx"


def _replace_initializer(
    model: onnx.ModelProto, name: str, value: np.ndarray
) -> None:
    kept = [x for x in model.graph.initializer if x.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(numpy_helper.from_array(value, name=name))


def _drop_unused_initializers(model: onnx.ModelProto) -> None:
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [x for x in model.graph.initializer if x.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def build() -> onnx.ModelProto:
    model = onnx.load(SOURCE)
    source_nodes = list(model.graph.node)

    # Stable source tensors ensure that a future repair change fails loudly.
    produced = {out for node in source_nodes for out in node.output}
    required = {"ab_eq", "three_new", "bar_b", "cond30", "output"}
    missing = required - produced
    if missing:
        raise RuntimeError(f"task397 source graph changed; missing: {missing}")

    # Shared moment and localization constants.
    _replace_initializer(
        model, "color_weights", np.arange(10, dtype=np.float32)
    )
    _replace_initializer(
        model, "row_weights", np.arange(30, dtype=np.float32)
    )
    _replace_initializer(
        model,
        "nonzero_weights",
        np.asarray([0.0] + [1.0] * 9, dtype=np.float32),
    )
    _replace_initializer(model, "starts0", np.asarray([0], dtype=np.int64))
    _replace_initializer(model, "starts1", np.asarray([1], dtype=np.int64))
    _replace_initializer(model, "end9", np.asarray([9], dtype=np.int64))
    _replace_initializer(model, "end10", np.asarray([10], dtype=np.int64))
    _replace_initializer(model, "axis0", np.asarray([0], dtype=np.int64))
    _replace_initializer(model, "zero_f", np.asarray([0.0], dtype=np.float32))
    _replace_initializer(model, "one_f", np.asarray([1.0], dtype=np.float32))
    _replace_initializer(model, "half_f", np.asarray([0.5], dtype=np.float32))
    _replace_initializer(model, "one_i32", np.asarray([1], dtype=np.int32))

    moment_prefix = [
        helper.make_node(
            "Einsum", ["input", "color_weights"], ["col_sum"], equation="nchw,c->w"
        ),
        helper.make_node(
            "Einsum",
            ["input", "color_weights", "row_weights"],
            ["row_color_sum"],
            equation="nchw,c,h->w",
        ),
        helper.make_node(
            "Einsum",
            ["input", "nonzero_weights", "row_weights"],
            ["row_sum"],
            equation="nchw,c,h->w",
        ),
        helper.make_node(
            "Slice", ["col_sum", "starts0", "end9", "axis0"], ["curr9"]
        ),
        helper.make_node(
            "Slice", ["row_sum", "starts0", "end9", "axis0"], ["row_curr9"]
        ),
        helper.make_node(
            "Slice", ["row_sum", "starts1", "end10", "axis0"], ["row_next9"]
        ),
        helper.make_node("Equal", ["row_curr9", "row_next9"], ["same_row"]),
        helper.make_node(
            "Where", ["same_row", "curr9", "zero_f"], ["left_score"]
        ),
        helper.make_node("TopK", ["left_score", "k3"], ["top_vals", "top_idx"]),
        helper.make_node("Greater", ["top_vals", "zero_f"], ["valid"]),
        helper.make_node("Cast", ["top_idx"], ["col_i32"], to=TensorProto.INT32),
        helper.make_node("Add", ["col_i32", "one_i32"], ["right_idx"]),
        helper.make_node("Gather", ["row_color_sum", "top_idx"], ["t_left"]),
        helper.make_node("Gather", ["row_sum", "top_idx"], ["r_left"]),
        helper.make_node("Gather", ["col_sum", "right_idx"], ["s_right"]),
        helper.make_node("Gather", ["row_color_sum", "right_idx"], ["t_right"]),
        # Since T = r*S + bottom_color, floating fmod recovers the bottom
        # color directly.  Max makes unused zero-score slots deterministic.
        helper.make_node("Max", ["top_vals", "one_f"], ["s_left_safe"]),
        helper.make_node("Mod", ["t_left", "s_left_safe"], ["c"], fmod=1),
        helper.make_node("Sub", ["top_vals", "c"], ["a"]),
        helper.make_node("Max", ["s_right", "one_f"], ["s_right_safe"]),
        helper.make_node("Mod", ["t_right", "s_right_safe"], ["d"], fmod=1),
        helper.make_node("Sub", ["s_right", "d"], ["b"]),
        helper.make_node("Mul", ["r_left", "half_f"], ["row_half"]),
        helper.make_node("Cast", ["row_half"], ["row_i32"], to=TensorProto.INT32),
        helper.make_node("Mul", ["row_i32", "ten_i32"], ["row10"]),
        helper.make_node("Add", ["row10", "col_i32"], ["base_idx"]),
    ]

    # Reuse the compact, proven distinct-color Boolean circuit unchanged,
    # excluding the old ids_flat offset/Gather nodes because a,b,c,d are now
    # recovered directly from the two moments.
    equality_logic = [
        node
        for node in (
            source_nodes[18:20] + source_nodes[22:27] + source_nodes[29:44]
        )
        if not {"u2", "u3"}.intersection(node.output)
    ]

    # Direct update order for cumulative height flags laid out as
    # [u1(objects), u2(objects), u3(objects), u4(objects)].
    order = []
    for obj in range(3):
        for height in range(4):
            order.extend([height * 3 + obj, height * 3 + obj])
    _replace_initializer(model, "update_order", np.asarray(order, dtype=np.int64))
    # Bars occupy output rows 2..9.  An 8x11 canvas removes the permanently
    # empty first row while reserving column 10 as a collision-free sentinel
    # for false update slots.
    _replace_initializer(model, "ten_i32", np.asarray([11], dtype=np.int32))
    _replace_initializer(model, "zero_i32", np.asarray([10], dtype=np.int32))
    _replace_initializer(
        model,
        "flat_offsets",
        np.asarray([[0, 1, 11, 12, 22, 23, 33, 34]], dtype=np.int32),
    )
    _replace_initializer(model, "zero100_bool", np.zeros(88, dtype=np.bool_))
    _replace_initializer(
        model, "shape1110", np.asarray([1, 1, 8, 11], dtype=np.int64)
    )
    _replace_initializer(
        model,
        "pad_cond",
        np.asarray([0, 0, 2, 0, 0, 0, 20, 19], dtype=np.int64),
    )

    draw_tail = [
        helper.make_node(
            "Concat", ["valid", "m2", "m3", "three_new"], ["flags12"], axis=0
        ),
        helper.make_node("Gather", ["flags12", "update_order"], ["updates"], axis=0),
        helper.make_node("Reshape", ["base_idx", "shape31"], ["base31"]),
        helper.make_node("Add", ["base31", "flat_offsets"], ["scatter_idx38"]),
        helper.make_node("Reshape", ["scatter_idx38", "shape24"], ["scatter_idx"]),
        helper.make_node(
            "Where", ["updates", "scatter_idx", "zero_i32"], ["scatter_idx_safe"]
        ),
        helper.make_node(
            "ScatterElements",
            ["zero100_bool", "scatter_idx_safe", "updates"],
            ["mask_flat"],
            axis=0,
        ),
        helper.make_node("Reshape", ["mask_flat", "shape1110"], ["bar_b"]),
        helper.make_node("Pad", ["bar_b", "pad_cond"], ["cond30"]),
        helper.make_node("Where", ["cond30", "green", "input"], ["output"]),
    ]

    del model.graph.node[:]
    model.graph.node.extend(moment_prefix + equality_logic + draw_tail)
    _drop_unused_initializers(model)
    del model.graph.value_info[:]
    # ONNX's variadic Einsum shape inference does not propagate these two
    # simple width-only outputs far enough for downstream Gather inference.
    model.graph.value_info.extend(
        [
            helper.make_tensor_value_info("col_sum", TensorProto.FLOAT, [30]),
            helper.make_tensor_value_info(
                "row_color_sum", TensorProto.FLOAT, [30]
            ),
            helper.make_tensor_value_info("row_sum", TensorProto.FLOAT, [30]),
        ]
    )

    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(model)
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    candidate = build()
    onnx.save(candidate, OUTPUT)
    print(OUTPUT)
