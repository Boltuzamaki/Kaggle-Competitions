"""Build the single-encoded-moment exact rewrite for task397.

For a physical column whose 2x2 object cells are colors ``p, q`` at rows
``r, r+1``, encode

    E = p * 128**(-r) + q * 128**(-(r+1))
      = 128**(-r) * (p + q/128).

The ratio of the two columns belonging to one object is in [1/9, 9].  A
touching object's row must differ, which changes that ratio by at least a
factor of 128.  Thus 0.1 < E_left/E_right < 10 exactly selects left columns.
The row and ordered colors are decoded with log/ceil and the fractional part.
All thresholds have wide float32 margins and are audited on all task examples.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from build_task397_moments import _drop_unused_initializers, _replace_initializer


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repairs" / "task397.onnx"
OUTPUT = ROOT / "other_model_onnx" / "task397.onnx"
Q = 128.0


def build() -> onnx.ModelProto:
    model = onnx.load(SOURCE)
    source_nodes = list(model.graph.node)

    produced = {out for node in source_nodes for out in node.output}
    required = {"ab_eq", "three_new", "bar_b", "cond30", "output"}
    missing = required - produced
    if missing:
        raise RuntimeError(f"task397 source graph changed; missing: {missing}")

    _replace_initializer(
        model, "color_weights", np.arange(10, dtype=np.float32)
    )
    _replace_initializer(
        model,
        "decay_weights",
        np.power(
            np.float32(Q), -np.arange(30, dtype=np.float32), dtype=np.float32
        ),
    )
    for name, value in {
        "starts0": 0,
        "starts1": 1,
        "end9": 9,
        "end10": 10,
    }.items():
        _replace_initializer(model, name, np.asarray([value], dtype=np.int64))
    _replace_initializer(model, "axes1", np.asarray([1], dtype=np.int64))
    for name, value in {
        "high_ratio": 10.0,
        "zero_f": 0.0,
        "one_f": 1.0,
        "q_f": Q,
        "neg_inv_log_q": -1.0 / math.log(Q),
    }.items():
        _replace_initializer(model, name, np.asarray([value], dtype=np.float32))
    _replace_initializer(model, "one_i32", np.asarray([1], dtype=np.uint8))

    encoded_prefix = [
        helper.make_node(
            "Einsum",
            ["input", "color_weights", "decay_weights"],
            ["encoded"],
            equation="nchw,c,h->w",
        ),
        helper.make_node(
            "Slice", ["encoded", "starts0", "end9"], ["curr9"]
        ),
        helper.make_node(
            "Slice", ["encoded", "starts1", "end10"], ["next9"]
        ),
        # Cross multiplication implements 0.1 < curr/next < 10 without a
        # division-by-zero guard: curr*10 > next and curr < next*10.
        helper.make_node("Mul", ["curr9", "high_ratio"], ["curr_scaled"]),
        helper.make_node("Greater", ["curr_scaled", "next9"], ["ratio_low"]),
        helper.make_node("Mul", ["next9", "high_ratio"], ["next_scaled"]),
        helper.make_node("Less", ["curr9", "next_scaled"], ["ratio_high"]),
        helper.make_node("And", ["ratio_low", "ratio_high"], ["same_row"]),
        helper.make_node(
            "Where", ["same_row", "curr9", "zero_f"], ["left_score"]
        ),
        helper.make_node("TopK", ["left_score", "k3"], ["top_vals", "top_idx"]),
        helper.make_node("Greater", ["top_vals", "zero_f"], ["valid"]),
        helper.make_node("Cast", ["top_idx"], ["col_u8"], to=TensorProto.UINT8),
        helper.make_node("Add", ["col_u8", "one_i32"], ["right_u8"]),
        helper.make_node("Cast", ["right_u8"], ["right_idx"], to=TensorProto.INT32),
        helper.make_node("Gather", ["encoded", "right_idx"], ["encoded_right"]),
        helper.make_node(
            "Where", ["valid", "top_vals", "one_f"], ["encoded_left_safe"]
        ),
        helper.make_node("Log", ["encoded_left_safe"], ["encoded_log"]),
        helper.make_node("Mul", ["encoded_log", "neg_inv_log_q"], ["row_pre"]),
        helper.make_node("Ceil", ["row_pre"], ["row_f"]),
        helper.make_node("Pow", ["q_f", "row_f"], ["row_scale"]),
        helper.make_node("Mul", ["top_vals", "row_scale"], ["left_code"]),
        helper.make_node("Cast", ["left_code"], ["a"], to=TensorProto.UINT8),
        helper.make_node("Mod", ["left_code", "one_f"], ["left_fraction"], fmod=1),
        helper.make_node("Mul", ["left_fraction", "q_f"], ["c_pre"]),
        helper.make_node("Cast", ["c_pre"], ["c"], to=TensorProto.UINT8),
        helper.make_node("Mul", ["encoded_right", "row_scale"], ["right_code"]),
        helper.make_node("Cast", ["right_code"], ["b"], to=TensorProto.UINT8),
        helper.make_node("Mod", ["right_code", "one_f"], ["right_fraction"], fmod=1),
        helper.make_node("Mul", ["right_fraction", "q_f"], ["d_pre"]),
        helper.make_node("Cast", ["d_pre"], ["d"], to=TensorProto.UINT8),
        # Canvas coordinates are < 88, so keep the affine coordinate math in
        # uint8 and cast only the final ScatterElements index to int32.
        helper.make_node("Cast", ["row_f"], ["row_u8"], to=TensorProto.UINT8),
        helper.make_node("Mul", ["row_u8", "ten_i32"], ["row11_u8"]),
        helper.make_node("Add", ["row11_u8", "col_u8"], ["base_idx"]),
    ]

    equality_logic = [
        node
        for node in (
            source_nodes[18:20] + source_nodes[22:27] + source_nodes[29:44]
        )
        if not {"u2", "u3"}.intersection(node.output)
    ]

    order = []
    for obj in range(3):
        for height in range(4):
            order.extend([height * 3 + obj, height * 3 + obj])
    _replace_initializer(model, "update_order", np.asarray(order, dtype=np.int64))
    _replace_initializer(model, "ten_i32", np.asarray([11], dtype=np.uint8))
    _replace_initializer(model, "zero_i32", np.asarray([10], dtype=np.uint8))
    _replace_initializer(
        model,
        "flat_offsets",
        np.asarray([[0, 1, 11, 12, 22, 23, 33, 34]], dtype=np.uint8),
    )
    _replace_initializer(model, "zero100_bool", np.zeros(88, dtype=np.bool_))
    _replace_initializer(
        model, "shape1110", np.asarray([1, 1, 8, 11], dtype=np.int64)
    )
    _replace_initializer(
        model,
        "pad_cond",
        np.asarray([2, 0, 20, 19], dtype=np.int64),
    )
    _replace_initializer(model, "pad_axes", np.asarray([2, 3], dtype=np.int64))

    draw_tail = [
        helper.make_node(
            "Concat", ["valid", "m2", "m3", "three_new"], ["flags12"], axis=0
        ),
        helper.make_node("Gather", ["flags12", "update_order"], ["updates"], axis=0),
        helper.make_node("Unsqueeze", ["base_idx", "axes1"], ["base31"]),
        helper.make_node("Add", ["base31", "flat_offsets"], ["scatter_idx38"]),
        helper.make_node("Reshape", ["scatter_idx38", "shape24"], ["scatter_idx"]),
        helper.make_node(
            "Where",
            ["updates", "scatter_idx", "zero_i32"],
            ["scatter_idx_safe_u8"],
        ),
        helper.make_node(
            "Cast", ["scatter_idx_safe_u8"], ["scatter_idx_safe"], to=TensorProto.INT32
        ),
        helper.make_node(
            "ScatterElements",
            ["zero100_bool", "scatter_idx_safe", "updates"],
            ["mask_flat"],
            axis=0,
        ),
        helper.make_node("Reshape", ["mask_flat", "shape1110"], ["bar_b"]),
        helper.make_node("Pad", ["bar_b", "pad_cond", "", "pad_axes"], ["cond30"]),
        helper.make_node("Where", ["cond30", "green", "input"], ["output"]),
    ]

    del model.graph.node[:]
    model.graph.node.extend(encoded_prefix + equality_logic + draw_tail)
    _drop_unused_initializers(model)
    del model.graph.value_info[:]
    model.graph.value_info.append(
        helper.make_tensor_value_info("encoded", TensorProto.FLOAT, [30])
    )

    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(model)
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    candidate = build()
    onnx.save(candidate, OUTPUT)
    print(OUTPUT)
