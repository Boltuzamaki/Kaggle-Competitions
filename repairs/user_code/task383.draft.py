# task383.py
# Checker-friendly: exposes top-level `model`.
# Rule: find the inner color defects in the outer rectangular border.  Rows/cols
# containing those defects become cuts through the rectangle; the inner color is
# extended outside the rectangle along those rows/cols.

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

F = TensorProto.FLOAT
I64 = np.int64
FP = np.float32


def _K(name, value, dtype):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def build_onnx_model():
    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

    init = [
        _K("c1s", [1], I64),
        _K("c10e", [10], I64),
        _K("ax1", [1], I64),
        _K("r0", [0], I64),
        _K("r1", [1], I64),
        _K("r30", [30], I64),
        _K("r31", [31], I64),
        _K("ax2", [2], I64),
        _K("ax3", [3], I64),
        _K("zero", [0.0], FP),
        _K("one", [1.0], FP),
        _K("half", [0.5], FP),
        _K("three", [3.0], FP),
        _K("pads_top", [0, 0, 1, 0, 0, 0, 0, 0], I64),
        _K("pads_bottom", [0, 0, 0, 0, 0, 0, 1, 0], I64),
        _K("pads_left", [0, 0, 0, 1, 0, 0, 0, 0], I64),
        _K("pads_right", [0, 0, 0, 0, 0, 0, 0, 1], I64),
        # channel gate: channel 0 is background and must never be treated as a color
        _K("nz_gate", np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=FP).reshape(1, 10, 1, 1), FP),
    ]

    n = []

    # valid real canvas cells: one-hot sum is 1 inside the real HxW area and 0 in padding.
    n += [
        helper.make_node("ReduceSum", ["input"], ["valid_sum"], axes=[1], keepdims=1),
        helper.make_node("Greater", ["valid_sum", "half"], ["valid_bool"]),
        helper.make_node("Cast", ["valid_bool"], ["valid"], to=TensorProto.FLOAT),
    ]

    # Nonzero object mask from channels 1..9.
    n += [
        helper.make_node("Slice", ["input", "c1s", "c10e", "ax1"], ["input_nz"]),
        helper.make_node("ReduceSum", ["input_nz"], ["obj_sum"], axes=[1], keepdims=1),
        helper.make_node("Greater", ["obj_sum", "half"], ["obj_bool"]),
        helper.make_node("Cast", ["obj_bool"], ["obj"], to=TensorProto.FLOAT),
    ]

    # Boundary of the filled rectangular object.  The color appearing on this boundary
    # is the outer color; the other nonzero color is the inner/cut color.
    n += [
        helper.make_node("Pad", ["obj", "pads_top", "zero"], ["p_top"]),
        helper.make_node("Slice", ["p_top", "r0", "r30", "ax2"], ["up"]),
        helper.make_node("Pad", ["obj", "pads_bottom", "zero"], ["p_bottom"]),
        helper.make_node("Slice", ["p_bottom", "r1", "r31", "ax2"], ["down"]),
        helper.make_node("Pad", ["obj", "pads_left", "zero"], ["p_left"]),
        helper.make_node("Slice", ["p_left", "r0", "r30", "ax3"], ["left"]),
        helper.make_node("Pad", ["obj", "pads_right", "zero"], ["p_right"]),
        helper.make_node("Slice", ["p_right", "r1", "r31", "ax3"], ["right"]),
        helper.make_node("Mul", ["obj", "up"], ["er1"]),
        helper.make_node("Mul", ["er1", "down"], ["er2"]),
        helper.make_node("Mul", ["er2", "left"], ["er3"]),
        helper.make_node("Mul", ["er3", "right"], ["interior"]),
        helper.make_node("Sub", ["obj", "interior"], ["boundary"]),
    ]

    # Row/column presence of the object; because the object is a solid rectangle,
    # broadcasting these gives the object bbox.
    n += [
        helper.make_node("ReduceMax", ["obj"], ["row_has"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["obj"], ["col_has"], axes=[2], keepdims=1),
        helper.make_node("Sub", ["one", "row_has"], ["not_row_has"]),
        helper.make_node("Sub", ["one", "col_has"], ["not_col_has"]),
        helper.make_node("Mul", ["row_has", "col_has"], ["bbox"]),
    ]

    # Per-channel gates for outer/inner colors.
    n += [
        helper.make_node("ReduceSum", ["input"], ["color_total"], axes=[2, 3], keepdims=1),
        helper.make_node("Greater", ["color_total", "half"], ["has_color_bool"]),
        helper.make_node("Cast", ["has_color_bool"], ["has_color"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["input", "boundary"], ["on_boundary"]),
        helper.make_node("ReduceSum", ["on_boundary"], ["boundary_count"], axes=[2, 3], keepdims=1),
        helper.make_node("Less", ["boundary_count", "half"], ["boundary_zero_bool"]),
        helper.make_node("Greater", ["boundary_count", "half"], ["boundary_pos_bool"]),
        helper.make_node("Cast", ["boundary_zero_bool"], ["boundary_zero"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["boundary_pos_bool"], ["boundary_pos"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["has_color", "boundary_zero"], ["inner_gate0"]),
        helper.make_node("Mul", ["inner_gate0", "nz_gate"], ["inner_gate"]),
        helper.make_node("Mul", ["has_color", "boundary_pos"], ["outer_gate0"]),
        helper.make_node("Mul", ["outer_gate0", "nz_gate"], ["outer_gate"]),
    ]

    # Detect inner-color defect rows/cols.  Main inner rows/cols have more than 3
    # inner-color cells.  Inner cells outside those main rows/cols mark the cuts.
    n += [
        helper.make_node("ReduceSum", ["input"], ["row_count"], axes=[3], keepdims=1),
        helper.make_node("ReduceSum", ["input"], ["col_count"], axes=[2], keepdims=1),
        helper.make_node("Greater", ["row_count", "three"], ["main_rows_bool"]),
        helper.make_node("Greater", ["col_count", "three"], ["main_cols_bool"]),
        helper.make_node("Cast", ["main_rows_bool"], ["main_rows"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["main_cols_bool"], ["main_cols"], to=TensorProto.FLOAT),
        helper.make_node("Sub", ["one", "main_rows"], ["not_main_rows"]),
        helper.make_node("Sub", ["one", "main_cols"], ["not_main_cols"]),
        helper.make_node("Mul", ["input", "inner_gate"], ["inner_input"]),
        helper.make_node("Mul", ["inner_input", "not_main_cols"], ["inner_bad_cols"]),
        helper.make_node("Mul", ["inner_input", "not_main_rows"], ["inner_bad_rows"]),
        helper.make_node("ReduceMax", ["inner_bad_cols"], ["sel_rows"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["inner_bad_rows"], ["sel_cols"], axes=[2], keepdims=1),
        helper.make_node("ReduceMax", ["sel_rows"], ["sel_rows_any"], axes=[1], keepdims=1),
        helper.make_node("ReduceMax", ["sel_cols"], ["sel_cols_any"], axes=[1], keepdims=1),
        helper.make_node("Max", ["sel_rows_any", "sel_cols_any"], ["line_any"]),
        helper.make_node("Mul", ["line_any", "bbox"], ["cut_inside"]),
    ]

    # Inner color extensions outside the rectangle.
    n += [
        helper.make_node("Mul", ["sel_rows", "not_col_has"], ["row_ext0"]),
        helper.make_node("Mul", ["row_ext0", "valid"], ["row_ext"]),
        helper.make_node("Mul", ["sel_cols", "not_row_has"], ["col_ext0"]),
        helper.make_node("Mul", ["col_ext0", "valid"], ["col_ext"]),
        helper.make_node("Add", ["row_ext", "col_ext"], ["inner_ext_sum"]),
        helper.make_node("Min", ["inner_ext_sum", "one"], ["inner_ext"]),
    ]

    # Compose non-background channels.
    # Inner cells on a cut become outer; outside the bbox the inner color is extended.
    n += [
        helper.make_node("Sub", ["one", "cut_inside"], ["not_cut_inside"]),
        helper.make_node("Mul", ["inner_input", "not_cut_inside"], ["inner_kept"]),
        helper.make_node("Max", ["inner_kept", "inner_ext"], ["inner_out"]),
        helper.make_node("Mul", ["input", "outer_gate"], ["outer_input"]),
        helper.make_node("Mul", ["cut_inside", "outer_gate"], ["outer_cut"]),
        helper.make_node("Max", ["outer_input", "outer_cut"], ["outer_out"]),
        helper.make_node("Add", ["inner_out", "outer_out"], ["colors_all"]),
    ]

    # Background only inside the real canvas, never in padded 30x30 area.
    n += [
        helper.make_node("Slice", ["colors_all", "c1s", "c10e", "ax1"], ["colors_nz"]),
        helper.make_node("ReduceMax", ["colors_nz"], ["occ"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["valid", "occ"], ["bg"]),
        helper.make_node("Concat", ["bg", "colors_nz"], ["output"], axis=1),
    ]

    graph = helper.make_graph(n, "task383", [x], [y], init)
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model)
    return model


model = build_onnx_model()
