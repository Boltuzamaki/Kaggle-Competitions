"""Build the optimized task096 decoder from the current repairs artifact.

The rewrite is deliberately algebraic and portable:

* row/column occupancy stays uint8 instead of bool -> uint8;
* the column crop is omitted because task096 inputs occupy only coordinates 0..18
  and ArgMax is unchanged by the known-zero trailing coordinates;

No negative padding or unsupported dtype-specific kernels are used.
"""

from pathlib import Path
import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path.cwd()
SOURCE = ROOT / "repairs" / "task096.onnx"
DESTINATION = ROOT / "other_model_onnx" / "task096.onnx"


def producer(model: onnx.ModelProto, output_name: str):
    for node in model.graph.node:
        if output_name in node.output:
            return node
    raise KeyError(output_name)


def set_cast_type(node, elem_type: int) -> None:
    assert node.op_type == "Cast"
    for attribute in node.attribute:
        if attribute.name == "to":
            attribute.i = elem_type
            return
    raise RuntimeError(f"Cast {node.output[0]} has no to attribute")


def set_value_info(model, name: str, elem_type: int, dims) -> None:
    for value_info in model.graph.value_info:
        if value_info.name != name:
            continue
        tensor_type = value_info.type.tensor_type
        tensor_type.elem_type = elem_type
        del tensor_type.shape.dim[:]
        for size in dims:
            tensor_type.shape.dim.add().dim_value = size
        return
    raise KeyError(name)


def build() -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load(SOURCE))

    # Row occupancy: float counts -> uint8 once.  The sliced tensor is already
    # exactly the uint8 0/1 tensor required by ArgMax and the gap arithmetic.
    row_reduce = producer(model, "row_sum_full")
    row_reduce.op_type = "ReduceMax"
    del row_reduce.input[:]
    row_reduce.input.extend(["input"])
    del row_reduce.attribute[:]
    row_reduce.attribute.extend(
        [helper.make_attribute("axes", [0, 3]), helper.make_attribute("keepdims", 0)]
    )
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.array([0, 2, 3], dtype=np.int64), "channel_count_axes_i64"
        )
    )
    channel_count = producer(model, "channel_count")
    del channel_count.input[:]
    channel_count.input.extend(["input", "channel_count_axes_i64"])
    bg_index = producer(model, "bg_idx_i64")
    for attribute in bg_index.attribute:
        if attribute.name == "keepdims":
            attribute.i = 1
    non_bg_count = producer(model, "non_bg_count")
    non_bg_count.op_type = "ScatterElements"
    del non_bg_count.input[:]
    non_bg_count.input.extend(["channel_count", "bg_idx_i64", "neg_one_f32"])
    del non_bg_count.attribute[:]
    non_bg_count.attribute.extend([helper.make_attribute("axis", 0)])
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == "neg_one_f32":
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(
                    np.array([-1.0], dtype=np.float32), "neg_one_f32"
                )
            )
    row_cast = producer(model, "row_present_full")
    set_cast_type(row_cast, TensorProto.UINT8)
    row_slice = producer(model, "row_present19")
    row_slice.output[0] = "row_any_u8"

    row_gap = producer(model, "row_zero_band")
    row_gap.op_type = "Sub"
    del row_gap.input[:]
    row_gap.input.extend(["row_in_span_u8", "row_any_u8"])
    del row_gap.attribute[:]

    # radius_num = (row_span - 1) + (long_gap_offset - short_gap_offset).
    # Reuse the existing span-minus-one tensor instead of materializing
    # row_span + length and subtracting one again.
    radius_delta = producer(model, "row_radius_num_plus_one")
    radius_delta.input[0] = "row_length"
    radius_delta.output[0] = "row_radius_delta"
    radius_num = producer(model, "row_radius_num")
    radius_num.op_type = "Add"
    del radius_num.input[:]
    radius_num.input.extend(["row_span_minus_one", "row_radius_delta"])
    del radius_num.attribute[:]

    # Column occupancy needs only first/last occupied coordinates.  Keeping the
    # selected 30-position uint8 tensor is cheaper than crop + another Cast;
    # positions 19..29 are guaranteed zero by the task's <=19 input contract.
    col_reduce = producer(model, "col_sum_full")
    col_reduce.op_type = "ReduceMax"
    del col_reduce.input[:]
    col_reduce.input.extend(["input"])
    del col_reduce.attribute[:]
    col_reduce.attribute.extend(
        [helper.make_attribute("axes", [0, 2]), helper.make_attribute("keepdims", 0)]
    )
    col_cast = producer(model, "col_present_full")
    set_cast_type(col_cast, TensorProto.UINT8)
    col_gather = producer(model, "col_present_selected")
    col_gather.output[0] = "col_any_u8"

    # Replace separate radius/corner grids and four full-grid render tensors by
    # a single static pair index.  A tiny [8,9] table is rebuilt per example and
    # gathered once to render the complete color grid.
    radius_big = next(
        numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
        if initializer.name == "radius_big"
    )
    corner_big = next(
        numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
        if initializer.name == "corner_big"
    )
    # Real rings use radii 0..5 and corner depths 1..6.  Radius 6 is the
    # sentinel row for both trailing canvas cells and absent TopK slots.
    compact_radius = np.minimum(radius_big.astype(np.int32), 6)
    compact_corner = np.minimum(corner_big.astype(np.int32), 6) - 1
    combined_big = compact_radius * 6 + compact_corner
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(combined_big, "combined_big"),
            numpy_helper.from_array(
                np.arange(1, 7, dtype=np.uint8).reshape(1, 6),
                "corner_levels_u8",
            ),
            numpy_helper.from_array(
                np.arange(7, dtype=np.int32), "radius_levels_i32"
            ),
            numpy_helper.from_array(np.array([42], dtype=np.int64), "shape42_i64"),
            numpy_helper.from_array(np.array(255, dtype=np.uint8), "minus_one_u8"),
        ]
    )
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == "zero_color_by_radius_u8":
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(
                    np.zeros(7, dtype=np.uint8), "zero_color_by_radius_u8"
                )
            )
        elif initializer.name == "seven_i32":
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(np.array(6, dtype=np.int32), "seven_i32")
            )
    combined_slice = producer(model, "radius_i32")
    combined_slice.input[0] = "combined_big"
    combined_slice.output[0] = "combined_idx"

    removed_outputs = {
        "row_present19",
        "col_present_selected",
        "col_present19",
        "radius_i32",
        "corner_depth_u8",
        "box_limit_i32",
        "in_box",
        "length_grid",
        "stroke_active",
        "color_grid_fg",
        "color_grid_in_box",
        "color_grid",
        "col_half",
        "ambiguous_s2_radius_code",
        "ambiguous_code_s2",
        "row_span_eq_three",
        "selected_count_i32",
        "count_lt_seven",
        "count_lt_ten",
        "col_span_plus_one",
        "s3_num_hi",
        "ambiguous_s3_radius_num",
        "ambiguous_s3_radius",
        "ambiguous_s3_radius_code",
        "ambiguous_code_s3_calc",
        "s4_m11",
        "s4_r11",
        "ambiguous_s4_radius_num",
        "ambiguous_s4_radius",
        "ambiguous_s4_radius_code",
        "ambiguous_code_s4",
        "ambiguous_code_not_s2",
        "ambiguous_code",
        "row_span_plus_length",
        "row_radius_num_plus_one",
        "bg_channel",
    }
    kept_nodes = [
        node
        for node in model.graph.node
        if not any(output in removed_outputs for output in node.output)
        and not (
            node.op_type == "Cast"
            and any(name in {"row_present19", "col_present19"} for name in node.input)
        )
    ]
    del model.graph.node[:]
    table_nodes = [
        helper.make_node(
            "Unsqueeze",
            ["length_by_radius", "slice_axis_1_i64"],
            ["length_col"],
        ),
        helper.make_node(
            "LessOrEqual",
            ["corner_levels_u8", "length_col"],
            ["table_active"],
        ),
        helper.make_node(
            "Unsqueeze",
            ["color_by_radius", "slice_axis_1_i64"],
            ["color_col"],
        ),
        helper.make_node(
            "LessOrEqual",
            ["radius_levels_i32", "maxd"],
            ["radius_present"],
        ),
        helper.make_node(
            "Where",
            ["radius_present", "bg_idx_u8", "ten_u8"],
            ["radius_fill"],
        ),
        helper.make_node(
            "Unsqueeze",
            ["radius_fill", "slice_axis_1_i64"],
            ["radius_fill_col"],
        ),
        helper.make_node(
            "Where",
            ["table_active", "color_col", "radius_fill_col"],
            ["render_table_2d"],
        ),
        helper.make_node(
            "Reshape", ["render_table_2d", "shape42_i64"], ["render_table"]
        ),
        helper.make_node(
            "Gather", ["render_table", "combined_idx"], ["color_grid"], axis=0
        ),
    ]
    ambiguous_nodes = [
        helper.make_node(
            "Cast", ["top_colors_0"], ["selected_count_i32"], to=TensorProto.UINT8
        ),
        helper.make_node(
            "Less", ["selected_count_i32", "seven_u8"], ["count_lt_seven"]
        ),
        helper.make_node(
            "Less", ["selected_count_i32", "ten_u8"], ["count_lt_ten"]
        ),
        helper.make_node(
            "Where",
            ["count_lt_ten", "zero_u8", "minus_one_u8"],
            ["s3_adjust_hi"],
        ),
        helper.make_node(
            "Where",
            ["count_lt_seven", "one_u8", "s3_adjust_hi"],
            ["s3_adjust"],
        ),
        helper.make_node(
            "Min", ["selected_count_i32", "eleven_u8"], ["count_cap_11"]
        ),
        helper.make_node(
            "Sub", ["eleven_u8", "count_cap_11"], ["s4_adjust"]
        ),
        helper.make_node(
            "Equal", ["row_span", "three_u8"], ["row_span_eq_three"]
        ),
        helper.make_node(
            "Where",
            ["row_span_eq_three", "s3_adjust", "s4_adjust"],
            ["adjust_not_s2"],
        ),
        helper.make_node(
            "Where",
            ["row_span_eq_two", "zero_u8", "adjust_not_s2"],
            ["ambiguous_adjust"],
        ),
        helper.make_node(
            "Add", ["col_span", "ambiguous_adjust"], ["ambiguous_radius_num"]
        ),
        helper.make_node(
            "Div", ["ambiguous_radius_num", "two_u8"], ["ambiguous_radius"]
        ),
        helper.make_node(
            "Mul", ["ambiguous_radius", "eight_u8"], ["ambiguous_radius_code"]
        ),
        helper.make_node(
            "Add", ["ambiguous_radius_code", "row_span"], ["ambiguous_code"]
        ),
    ]
    final_nodes = []
    for node in kept_nodes:
        if "code_pre" in node.output:
            final_nodes.extend(ambiguous_nodes)
        if "padded_color" in node.output:
            final_nodes.extend(table_nodes)
        final_nodes.append(node)
    model.graph.node.extend(final_nodes)

    # Retain the source's baked maximum shapes, updating only tensors changed
    # above.  Those baked maxima are part of the competition scoring contract.
    kept_value_info = [
        value_info
        for value_info in model.graph.value_info
        if value_info.name not in removed_outputs
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    set_value_info(model, "row_present_full", TensorProto.UINT8, [10, 30])
    set_value_info(model, "row_present_selected", TensorProto.UINT8, [5, 30])
    set_value_info(model, "row_any_u8", TensorProto.UINT8, [5, 19])
    set_value_info(model, "col_present_full", TensorProto.UINT8, [10, 30])
    set_value_info(model, "col_any_u8", TensorProto.UINT8, [5, 30])
    set_value_info(model, "bg_idx_i64", TensorProto.INT64, [1])
    set_value_info(model, "bg_idx_u8", TensorProto.UINT8, [1])
    new_value_info = [
        helper.make_tensor_value_info("combined_idx", TensorProto.INT32, [11, 11]),
        helper.make_tensor_value_info("length_col", TensorProto.UINT8, [7, 1]),
        helper.make_tensor_value_info("table_active", TensorProto.BOOL, [7, 6]),
        helper.make_tensor_value_info("color_col", TensorProto.UINT8, [7, 1]),
        helper.make_tensor_value_info("radius_present", TensorProto.BOOL, [7]),
        helper.make_tensor_value_info("radius_fill", TensorProto.UINT8, [7]),
        helper.make_tensor_value_info("radius_fill_col", TensorProto.UINT8, [7, 1]),
        helper.make_tensor_value_info("render_table_2d", TensorProto.UINT8, [7, 6]),
        helper.make_tensor_value_info("render_table", TensorProto.UINT8, [42]),
        helper.make_tensor_value_info("color_grid", TensorProto.UINT8, [11, 11]),
    ]
    model.graph.value_info.extend(new_value_info)
    compact_decoder_value_info = [
        helper.make_tensor_value_info("selected_count_i32", TensorProto.UINT8, [5]),
        helper.make_tensor_value_info("count_lt_seven", TensorProto.BOOL, [5]),
        helper.make_tensor_value_info("count_lt_ten", TensorProto.BOOL, [5]),
        helper.make_tensor_value_info("s3_adjust_hi", TensorProto.UINT8, [5]),
        helper.make_tensor_value_info("s3_adjust", TensorProto.UINT8, [5]),
        helper.make_tensor_value_info("count_cap_11", TensorProto.UINT8, [5]),
        helper.make_tensor_value_info("s4_adjust", TensorProto.UINT8, [5]),
        helper.make_tensor_value_info("row_span_eq_three", TensorProto.BOOL, [5]),
        helper.make_tensor_value_info("adjust_not_s2", TensorProto.UINT8, [5]),
        helper.make_tensor_value_info("ambiguous_adjust", TensorProto.UINT8, [5]),
        helper.make_tensor_value_info("ambiguous_radius_num", TensorProto.UINT8, [5]),
        helper.make_tensor_value_info("ambiguous_radius", TensorProto.UINT8, [5]),
        helper.make_tensor_value_info("ambiguous_radius_code", TensorProto.UINT8, [5]),
        helper.make_tensor_value_info("ambiguous_code", TensorProto.UINT8, [5]),
    ]
    model.graph.value_info.extend(compact_decoder_value_info)
    model.graph.value_info.append(
        helper.make_tensor_value_info("row_radius_delta", TensorProto.UINT8, [5])
    )
    for name in ["length_seed", "length_by_radius", "color_seed", "color_by_radius"]:
        set_value_info(model, name, TensorProto.UINT8, [7])

    # Drop constants that became unreachable with the two seed subgraphs.
    referenced = {name for node in model.graph.node for name in node.input}
    kept_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in referenced
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

    onnx.checker.check_model(model, full_check=True)
    return model


model = build()


if __name__ == "__main__":
    onnx.save(model, DESTINATION)
    print(DESTINATION)
