"""Build a compact, rule-preserving task096 decoder.

The repairs model first materializes row/column occupancy for all ten colors.
This rewrite projects only the five TopK candidate colors.  Column bounds are
encoded directly as two exponential moments, replacing the full 10x30 column
projection with ten scalars.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repairs" / "task096.onnx"
OUTPUT = ROOT / "other_model_onnx" / "task096.onnx"


def tensor(name, value, dtype):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def vi(name, dtype, dims):
    return helper.make_tensor_value_info(name, dtype, dims)


def build():
    model = onnx.load(SOURCE)
    graph = model.graph

    depth10 = tensor("opt_depth10", np.int64(10), np.int64)
    oh_values = tensor("opt_oh_values", [0.0, 1.0], np.float32)
    zero_f = tensor("opt_zero_f", np.float32(0.0), np.float32)
    one_f = tensor("opt_one_f", np.float32(1.0), np.float32)
    inv_log64 = tensor("opt_inv_log64", np.float32(1.0 / np.log(64.0)), np.float32)
    dir0 = tensor("opt_dir0", np.int64(0), np.int64)
    dir1 = tensor("opt_dir1", np.int64(1), np.int64)
    eighteen = tensor("opt_eighteen", np.uint8(18), np.uint8)

    # Base 64 is larger than the maximum number of pixels of one color.  Thus
    # the leading exponent identifies the extreme occupied coordinate even
    # when several pixels share a column.  The 0.02 offset keeps Cast safely
    # away from an integer boundary after Log.
    eps = np.float32(0.02)
    weights = np.zeros((2, 30), dtype=np.float32)
    weights[0, :19] = np.power(np.float32(64.0), np.arange(19, dtype=np.float32) + eps)
    weights[1, :19] = np.power(np.float32(64.0), np.arange(18, -1, -1, dtype=np.float32) + eps)
    col_weights = tensor("opt_col_weights", weights, np.float32)
    graph.initializer.extend(
        [depth10, oh_values, zero_f, one_f, inv_log64, dir0, dir1, eighteen, col_weights]
    )

    # Outputs removed from the original all-color row/column projections.
    removed = {
        "row_sum_full", "row_present_full", "row_present_selected",
        "col_sum_full", "col_present_full", "col_any_u8",
        "col_bottom_i64", "col_bottom", "col_top_i64", "col_top",
        "col_span_minus_one", "col_span",
    }

    kept = []
    for node in graph.node:
        if any(name in removed for name in node.output):
            continue
        kept.append(node)

    # Insert candidate projection immediately after TopK.  The existing Slice
    # producing row_any_u8 remains unchanged.
    row_nodes = [
        helper.make_node(
            "OneHot", ["top_colors_1", "opt_depth10", "opt_oh_values"], ["opt_sel_oh"], axis=-1
        ),
        helper.make_node(
            "Einsum", ["input", "opt_sel_oh"], ["opt_row_counts"], equation="nkhw,ek->eh"
        ),
        # Casting nonzero counts to bool is the occupancy test.  Crop this bool
        # tensor before the uint8 conversion to save one 5x30 activation.
        helper.make_node("Cast", ["opt_row_counts"], ["opt_row_present_bool"], to=TensorProto.BOOL),
    ]

    # Four natural-log units per coordinate leave enough room for the maximum
    # pixel multiplicity.  Integer division after Cast replaces a float Mul.
    weights[0, :19] = np.exp(np.float32(4.0) * np.arange(19, dtype=np.float32) + eps)
    weights[1, :19] = np.exp(np.float32(4.0) * np.arange(18, -1, -1, dtype=np.float32) + eps)
    for initializer in graph.initializer:
        if initializer.name == "opt_col_weights":
            initializer.CopyFrom(tensor("opt_col_weights", weights, np.float32))
            break
    four_u8 = tensor("opt_four_u8", np.uint8(4), np.uint8)
    # Pack count and coordinate sum into one integer: 172 exceeds the largest
    # possible coordinate sum (0+...+18), while 172+r still fits uint8.
    row_moment_weights = np.uint8(172) + np.arange(19, dtype=np.uint8)
    graph.initializer.extend(
        [
            four_u8,
            tensor("opt_row_moment_weights", row_moment_weights, np.uint8),
            tensor("opt_moment_base", np.uint16(172), np.uint16),
            tensor("opt_zero_u16", np.uint16(0), np.uint16),
            tensor("opt_one_u16", np.uint16(1), np.uint16),
            tensor("opt_two_u16", np.uint16(2), np.uint16),
            tensor("opt_seventeen", np.uint8(17), np.uint8),
        ]
    )

    col_nodes = [
        helper.make_node(
            "Einsum", ["input", "opt_sel_oh", "opt_col_weights"], ["opt_col_code"],
            equation="nkhw,ek,dw->ed",
        ),
        # Empty TopK slots yield Log(0)=-inf; ORT's following uint8 Cast maps
        # those ignored slots to zero.  Present slots are strictly positive.
        helper.make_node("Log", ["opt_col_code"], ["opt_col_log"]),
        helper.make_node("Cast", ["opt_col_log"], ["opt_col_q"], to=TensorProto.UINT8),
        helper.make_node("BitShift", ["opt_col_q", "two_u8"], ["opt_col_pos"], direction="RIGHT"),
        helper.make_node("Gather", ["opt_col_pos", "opt_dir0"], ["opt_col_right"], axis=1),
        helper.make_node("Gather", ["opt_col_pos", "opt_dir1"], ["opt_col_rev"], axis=1),
    ]

    # Replace five full 5x19 span tensors by one unsigned interval test.
    removed_span = {
        "row_top_minus_one_vec", "row_ge_top", "row_bottom_plus_one_vec",
        "row_le_bottom", "row_in_span", "row_in_span_u8", "row_zero_band",
        "row_span_minus_one", "row_gap_exists_u8", "row_gap_exists",
        "row_first_gap_i64", "row_first_gap", "row_last_gap_i64", "row_last_gap",
    }
    removed_decode = {
        "code_pre_i32", "code_i32", "radius_raw", "maxd", "length_i32", "length_u8",
    }
    removed_ambiguous = {"count_lt_seven", "count_lt_ten", "s3_adjust_hi", "s3_adjust"}
    kept = [
        node for node in kept
        if not any(
            name in removed_span or name in removed_decode or name in removed_ambiguous
            for name in node.output
        )
    ]

    compact_s3_nodes = [
        helper.make_node("Add", ["selected_count_i32", "two_u8"], ["opt_count_plus_two"]),
        helper.make_node("Div", ["opt_count_plus_two", "three_u8"], ["opt_count_thirds"]),
        helper.make_node("Sub", ["three_u8", "opt_count_thirds"], ["s3_adjust"]),
    ]

    span_nodes = [
        helper.make_node("Sub", ["row_bottom", "row_top"], ["row_span_minus_one"]),
    ]

    # The occupied rows form one or two intervals.  For a gap of length G,
    # its endpoints follow exactly from G and the sum of missing coordinates:
    #   2*mean(gap) = (span*(top+bottom) - 2*sum(occupied)) / G.
    # MatMulInteger obtains occupied-count and coordinate-sum together without
    # materializing another float/bool row tensor.
    moment_nodes = [
        helper.make_node(
            "MatMulInteger", ["row_any_u8", "opt_row_moment_weights"],
            ["opt_row_moments_i32"],
        ),
        helper.make_node("Cast", ["opt_row_moments_i32"], ["opt_row_moments_u16"], to=TensorProto.UINT16),
        helper.make_node("Div", ["opt_row_moments_u16", "opt_moment_base"], ["opt_row_occupied"]),
        helper.make_node("Mod", ["opt_row_moments_u16", "opt_moment_base"], ["opt_row_sum"], fmod=0),
        helper.make_node("Cast", ["row_top"], ["opt_row_top_u16"], to=TensorProto.UINT16),
        helper.make_node("Cast", ["row_bottom"], ["opt_row_bottom_u16"], to=TensorProto.UINT16),
        helper.make_node("Cast", ["row_span"], ["opt_row_span_u16"], to=TensorProto.UINT16),
        helper.make_node("Sub", ["opt_row_span_u16", "opt_row_occupied"], ["opt_gap_len"]),
        helper.make_node("Equal", ["opt_gap_len", "opt_zero_u16"], ["opt_no_gap"]),
        helper.make_node("Add", ["opt_row_top_u16", "opt_row_bottom_u16"], ["opt_span_end_sum"]),
        helper.make_node("Mul", ["opt_row_span_u16", "opt_span_end_sum"], ["opt_all_twice"]),
        helper.make_node("Add", ["opt_row_sum", "opt_row_sum"], ["opt_occupied_twice"]),
        helper.make_node("Sub", ["opt_all_twice", "opt_occupied_twice"], ["opt_gap_twice"]),
        helper.make_node("Cast", ["opt_no_gap"], ["opt_no_gap_u16"], to=TensorProto.UINT16),
        helper.make_node("Add", ["opt_gap_len", "opt_no_gap_u16"], ["opt_gap_safe"]),
        helper.make_node("Div", ["opt_gap_twice", "opt_gap_safe"], ["opt_gap_mean2"]),
        helper.make_node("Sub", ["opt_gap_safe", "opt_one_u16"], ["opt_gap_minus_one"]),
        helper.make_node("Sub", ["opt_gap_mean2", "opt_gap_minus_one"], ["opt_gap_first_twice"]),
        helper.make_node("Div", ["opt_gap_first_twice", "opt_two_u16"], ["opt_gap_first_u16"]),
        helper.make_node("Add", ["opt_gap_first_u16", "opt_gap_minus_one"], ["opt_gap_last_u16"]),
        helper.make_node("Cast", ["opt_gap_first_u16"], ["row_first_gap"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["opt_gap_last_u16"], ["row_last_gap"], to=TensorProto.UINT8),
    ]

    decode_nodes = [
        helper.make_node("Where", ["present_bool", "code_pre", "zero_u8"], ["opt_code_u8"]),
        helper.make_node("Div", ["opt_code_u8", "eight_u8"], ["opt_radius_u8"]),
        helper.make_node("Cast", ["opt_radius_u8"], ["radius_raw"], to=TensorProto.INT32),
        helper.make_node("ReduceMax", ["radius_raw"], ["maxd"], axes=[0], keepdims=0),
        helper.make_node("Mod", ["opt_code_u8", "eight_u8"], ["length_u8"], fmod=0),
    ]

    final_nodes = []
    row_inserted = False
    col_inserted = False
    for node in kept:
        if "channel_count" in node.output:
            node.op_type = "Einsum"
            del node.input[:]
            node.input.extend(["input"])
            del node.attribute[:]
            node.attribute.extend([helper.make_attribute("equation", "nkhw->k")])
        if not row_inserted and node.op_type == "Cast" and "top_colors_0" in node.input:
            # This is present_bool, immediately following TopK in the source.
            final_nodes.extend(row_nodes)
            row_inserted = True
        if not col_inserted and "selected_count_i32" in node.output:
            final_nodes.extend(col_nodes)
            col_inserted = True
        if "row_any_u8" in node.output:
            node.input[0] = "opt_row_present_bool"
            node.output[0] = "opt_row_any_bool"
            final_nodes.append(node)
            final_nodes.append(
                helper.make_node("Cast", ["opt_row_any_bool"], ["row_any_u8"], to=TensorProto.UINT8)
            )
            continue
        if "row_code" in node.output:
            node.input[0] = "opt_no_gap"
            node.input[1], node.input[2] = node.input[2], node.input[1]
        if "ambiguous_radius_num" in node.output:
            final_nodes.append(
                helper.make_node(
                    "Add", ["opt_col_right", "opt_col_rev"], ["opt_col_sum"],
                )
            )
            final_nodes.append(
                helper.make_node(
                    "Add", ["opt_col_sum", "ambiguous_adjust"],
                    ["opt_ambiguous_sum"],
                )
            )
            final_nodes.append(
                helper.make_node(
                    "Sub", ["opt_ambiguous_sum", "opt_seventeen"],
                    ["ambiguous_radius_num"],
                )
            )
            continue
        final_nodes.append(node)
        if "row_bottom" in node.output:
            final_nodes.extend(span_nodes)
        if "row_span" in node.output:
            final_nodes.extend(moment_nodes)
        if "selected_count_i32" in node.output:
            final_nodes.extend(compact_s3_nodes)
        if "code_pre" in node.output:
            final_nodes.extend(decode_nodes)
    if not row_inserted or not col_inserted:
        raise RuntimeError((row_inserted, col_inserted))
    del graph.node[:]
    graph.node.extend(final_nodes)

    removed_vi = removed | removed_span | removed_decode | removed_ambiguous | {
        "opt_sel_oh", "opt_row_counts", "opt_row_present_bool", "opt_row_any_bool",
        "opt_col_code", "opt_col_safe", "opt_col_log", "opt_col_q", "opt_col_pos",
        "opt_col_right", "opt_col_rev", "opt_col_left",
        "opt_row_top_col", "opt_row_delta", "opt_row_width_col",
        "opt_row_moments_i32", "opt_row_moments_u16", "opt_row_occupied",
        "opt_row_sum", "opt_row_top_u16", "opt_row_bottom_u16", "opt_row_span_u16",
        "opt_gap_len", "opt_span_end_sum", "opt_all_twice", "opt_occupied_twice",
        "opt_gap_twice", "opt_no_gap", "opt_no_gap_u16", "opt_gap_safe", "opt_gap_mean2", "opt_gap_minus_one",
        "opt_gap_first_twice", "opt_gap_first_u16", "opt_gap_last_u16",
        "opt_count_plus_two", "opt_count_thirds",
        "opt_col_sum", "opt_ambiguous_sum",
        "opt_code_u8", "opt_radius_u8", "opt_maxd_u8",
    }
    old_vi = [x for x in graph.value_info if x.name not in removed_vi]
    del graph.value_info[:]
    graph.value_info.extend(old_vi)
    graph.value_info.extend(
        [
            vi("opt_sel_oh", TensorProto.FLOAT, [5, 10]),
            vi("opt_row_counts", TensorProto.FLOAT, [5, 30]),
            vi("opt_row_present_bool", TensorProto.BOOL, [5, 30]),
            vi("opt_row_any_bool", TensorProto.BOOL, [5, 19]),
            vi("opt_col_code", TensorProto.FLOAT, [5, 2]),
            vi("opt_col_log", TensorProto.FLOAT, [5, 2]),
            vi("opt_col_q", TensorProto.UINT8, [5, 2]),
            vi("opt_col_pos", TensorProto.UINT8, [5, 2]),
            vi("opt_col_right", TensorProto.UINT8, [5]),
            vi("opt_col_rev", TensorProto.UINT8, [5]),
            vi("opt_row_moments_i32", TensorProto.INT32, [5]),
            vi("opt_row_moments_u16", TensorProto.UINT16, [5]),
            vi("opt_row_occupied", TensorProto.UINT16, [5]),
            vi("opt_row_sum", TensorProto.UINT16, [5]),
            vi("opt_row_top_u16", TensorProto.UINT16, [5]),
            vi("opt_row_bottom_u16", TensorProto.UINT16, [5]),
            vi("opt_row_span_u16", TensorProto.UINT16, [5]),
            vi("opt_gap_len", TensorProto.UINT16, [5]),
            vi("opt_span_end_sum", TensorProto.UINT16, [5]),
            vi("opt_all_twice", TensorProto.UINT16, [5]),
            vi("opt_occupied_twice", TensorProto.UINT16, [5]),
            vi("opt_gap_twice", TensorProto.UINT16, [5]),
            vi("opt_no_gap", TensorProto.BOOL, [5]),
            vi("opt_no_gap_u16", TensorProto.UINT16, [5]),
            vi("opt_gap_safe", TensorProto.UINT16, [5]),
            vi("opt_gap_mean2", TensorProto.UINT16, [5]),
            vi("opt_gap_minus_one", TensorProto.UINT16, [5]),
            vi("opt_gap_first_twice", TensorProto.UINT16, [5]),
            vi("opt_gap_first_u16", TensorProto.UINT16, [5]),
            vi("opt_gap_last_u16", TensorProto.UINT16, [5]),
            vi("opt_count_plus_two", TensorProto.UINT8, [5]),
            vi("opt_count_thirds", TensorProto.UINT8, [5]),
            vi("opt_col_sum", TensorProto.UINT8, [5]),
            vi("opt_ambiguous_sum", TensorProto.UINT8, [5]),
            vi("opt_code_u8", TensorProto.UINT8, [5]),
            vi("opt_radius_u8", TensorProto.UINT8, [5]),
        ]
    )

    # Reuse the existing int64 zero vector for the two radius-zero scatters.
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == "zero_index_vec_i32":
                node.input[index] = "slice_start_0_i64"

    used = {name for node in graph.node for name in node.input if name}
    inits = [x for x in graph.initializer if x.name in used]
    del graph.initializer[:]
    graph.initializer.extend(inits)

    onnx.checker.check_model(model, full_check=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Break a possible repository hard link before writing the candidate.
    if OUTPUT.exists():
        OUTPUT.unlink()
    onnx.save(model, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
