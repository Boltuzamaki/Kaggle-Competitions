"""Reduce task148's marker and directed-ray pipelines.

Each active marker row contains exactly one color-8 cell and marker columns are
strictly positive.  A weighted contraction therefore gives both the column
and, by comparison with zero, the row-presence bit.  This is algebraically
equivalent to the former ReduceMax + ArgMax chain on the task's structural
domain.  The global left/right choice is then represented as +1/-1, turning
the two comparison branches into one signed comparison.  Destination rows
receive an all-column threshold, so the existing frontier behavior is folded
into the same comparison without materializing a second mask.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submissions" / "neurogolf7300" / "task148.onnx"
OUT = ROOT / "other_model_onnx" / "task148_marker_matmul.onnx"


def build():
    model = onnx.load(SOURCE)
    nodes = list(model.graph.node)

    remove_outputs = {
        # Old right-line slice/reduction and marker slice/ArgMax chains.
        "right_red_area", "right_red_area_b", "right_rows_f", "right_rows",
        "right_col_safe",
        "c8_top", "c8_top_b", "row_c8_f", "row_has_8", "marker_col",
        "marker_col_u8", "label_top",
        "left_rows_f", "left_rows", "left_top_i64", "right_top_i64",
        "row_has_8_i", "masked_coords", "masked_shift", "dest_idx_i8",
        # Paint the two possible red cells per row directly.
        "red23_b", "label23",
        # Direction is cheaper from the two already-computed line tops.
        "left_rows_top", "right_rows_top", "source_left_rows",
        "source_right_rows", "source_left_rows_i", "source_left_flag_i",
        # Two-branch direction and shift selection.
        "source_right_flag_i", "shift_lr", "shift_rl", "shift_lr_sel",
        "shift_rl_sel", "signed_shift",
        # Two-branch ray comparison.
        "ml_inner", "masked_left", "masked_right", "leftfill",
        "rightfill", "fill_pre", "fill_top",
    }
    kept = [n for n in nodes if not any(o in remove_outputs for o in n.output)]

    compact_extract_nodes = [
        # Weighted color/column contractions keep the row coordinate and avoid
        # materializing the large float32 channel slices.
        # Weight column zero by 16.  Color 2 at column zero encodes left-line
        # presence, while the 7..11 contribution still carries the right-line
        # column.  Mod/threshold recover both facts from one row scalar.
        helper.make_node("Einsum", ["input", "sel_c2", "all_col_weights", "unit_i", "unit_k"], ["right_col_all"], equation="nchw,c,w,i,k->nihk"),
        helper.make_node("Slice", ["right_col_all", "starts_top", "ends_tail", "axes_row"], ["right_rows_f"]),
        helper.make_node("Cast", ["right_rows_f"], ["red_code_u8"], to=TensorProto.UINT8),
        helper.make_node("Mod", ["red_code_u8", "sixteen_u8"], ["right_col_u8"]),
        helper.make_node("Greater", ["red_code_u8", "fifteen_u8"], ["left_rows"]),
        helper.make_node("Greater", ["right_col_u8", "zero_u8"], ["right_rows"]),

        helper.make_node("Einsum", ["input", "sel_c8", "all_col_weights", "unit_i", "unit_k"], ["marker_col_all"], equation="nchw,c,w,i,k->nihk"),
        helper.make_node("Slice", ["marker_col_all", "starts_top", "ends_top", "axes_row"], ["marker_col_f"]),
        helper.make_node("Cast", ["marker_col_f"], ["marker_col_u8"], to=TensorProto.UINT8),
        helper.make_node("Greater", ["marker_col_u8", "zero_u8"], ["row_has_8"]),
    ]

    # Insert after col_valid so every later row/direction consumer is topological.
    insert_at = next(i for i, n in enumerate(kept) if "col_valid" in n.output) + 1
    kept[insert_at:insert_at] = compact_extract_nodes

    top_nodes = [
        # The packed code is >=16 exactly on left-line rows and <=11 on
        # right-only rows, so its first maximum is the left line's top.
        helper.make_node("ArgMax", ["red_code_u8"], ["left_top_i64"], axis=2, keepdims=1),
        helper.make_node("ArgMax", ["right_col_u8"], ["right_top_i64"], axis=2, keepdims=1),
    ]
    top_at = next(i for i, n in enumerate(kept) if "row_has_8" in n.output) + 1
    kept[top_at:top_at] = top_nodes

    direction_nodes = [
        helper.make_node("Less", ["left_top", "right_top"], ["source_left_flag_b"]),
        helper.make_node("Sub", ["right_top", "left_top"], ["top_delta"]),
        helper.make_node("Abs", ["top_delta"], ["signed_shift"]),
        helper.make_node("Add", ["row_coords_top", "signed_shift"], ["dest_idx_raw"]),
        helper.make_node("Cast", ["row_has_8"], ["row_has_8_i"], to=TensorProto.INT8),
        helper.make_node("Mul", ["dest_idx_raw", "row_has_8_i"], ["dest_idx_i8"]),
    ]
    direction_at = next(i for i, n in enumerate(kept) if "right_top" in n.output) + 1
    kept[direction_at:direction_at] = direction_nodes

    ray_nodes = [
        helper.make_node("Where", ["source_left_flag_b", "col_coords", "rev_col_coords"], ["directed_cols"]),
        helper.make_node("Sub", ["eleven_u8", "marker_col_u8"], ["marker_col_rev"]),
        helper.make_node("Where", ["source_left_flag_b", "marker_col_u8", "marker_col_rev"], ["directed_marker"]),
        helper.make_node("Where", ["row_has_8", "directed_marker", "zero_u8"], ["ray_threshold"]),
        helper.make_node("Where", ["dest_top", "c12_u8", "ray_threshold"], ["fill_threshold"]),
        helper.make_node("Where", ["col_valid", "directed_cols", "c12_u8"], ["valid_directed_cols"]),
        helper.make_node("Less", ["valid_directed_cols", "fill_threshold"], ["fill_top"]),
    ]
    ray_at = next(i for i, n in enumerate(kept) if "label_base_top" in n.output) + 1
    kept[ray_at:ray_at] = ray_nodes

    # A source row contains at most one marker.  Place that cell directly
    # instead of expanding marker_col to a charged [10, 12] equality mask.
    # Rows without a marker scatter their existing column-zero value back, so
    # destination-ray rows and background rows are both preserved exactly.
    marker_overlay_nodes = [
        helper.make_node("Cast", ["marker_col_u8"], ["marker_col_i32"], to=TensorProto.INT32),
        helper.make_node("GatherElements", ["label_fill_top", "marker_col_i32"], ["marker_old"], axis=3),
        helper.make_node("Where", ["row_has_8", "four_u8", "marker_old"], ["marker_updates"]),
        helper.make_node("ScatterElements", ["label_fill_top", "marker_col_i32", "marker_updates"], ["label_top"], axis=3),
    ]
    marker_at = next(i for i, n in enumerate(kept) if "label_fill_top" in n.output) + 1
    kept[marker_at:marker_at] = marker_overlay_nodes

    # Each row has at most a left red cell at column zero and a right red cell
    # at right_col_u8.  Gather the old values and scatter either red or the old
    # value back.  This replaces a [24, 12] boolean mask and its stored false
    # middle block with two coordinates per row.
    red_overlay_nodes = [
        helper.make_node("Where", ["right_rows", "right_col_u8", "eleven_u8"], ["right_red_col_safe"]),
        helper.make_node("Concat", ["zero_left_cols", "right_red_col_safe"], ["red_cols_u8"], axis=3),
        helper.make_node("Cast", ["red_cols_u8"], ["red_cols_i32"], to=TensorProto.INT32),
        helper.make_node("Concat", ["left_rows", "right_rows"], ["red_present"], axis=3),
        helper.make_node("GatherElements", ["label_pre23", "red_cols_i32"], ["red_old"], axis=3),
        helper.make_node("Where", ["red_present", "two_u8", "red_old"], ["red_updates"]),
        helper.make_node("ScatterElements", ["label_pre23", "red_cols_i32", "red_updates"], ["label23"], axis=3),
    ]
    red_at = next(i for i, n in enumerate(kept) if "label_pre23" in n.output) + 1
    kept[red_at:red_at] = red_overlay_nodes

    del model.graph.node[:]
    model.graph.node.extend(kept)
    model.graph.initializer.extend([
        numpy_helper.from_array(np.asarray(2, dtype=np.int8), "two_i"),
        numpy_helper.from_array(np.asarray(11, dtype=np.uint8), "eleven_u8"),
        numpy_helper.from_array(np.arange(11, -1, -1, dtype=np.uint8).reshape(1, 1, 1, 12), "rev_col_coords"),
        numpy_helper.from_array(np.eye(10, dtype=np.float32)[2], "sel_c2"),
        numpy_helper.from_array(np.eye(10, dtype=np.float32)[8], "sel_c8"),
        numpy_helper.from_array(np.asarray([16, *range(1, 30)], dtype=np.float32), "all_col_weights"),
        numpy_helper.from_array(np.ones(1, dtype=np.float32), "unit_i"),
        numpy_helper.from_array(np.ones(1, dtype=np.float32), "unit_k"),
        numpy_helper.from_array(np.zeros((1, 1, 24, 1), dtype=np.uint8), "zero_left_cols"),
        numpy_helper.from_array(np.asarray(16, dtype=np.uint8), "sixteen_u8"),
        numpy_helper.from_array(np.asarray(15, dtype=np.uint8), "fifteen_u8"),
    ])

    # The two direct contractions make the old right/c8 Slice metadata dead.
    used = {name for node in model.graph.node for name in node.input}
    live = [x for x in model.graph.initializer if x.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(live)

    # Keep the artifact directly loadable as well as sanitizer-loadable.
    for node in model.graph.node:
        node.name = node.output[0]

    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUT)
    print(OUT)
