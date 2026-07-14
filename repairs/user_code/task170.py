"""Exact scorer-aware simplifications for repairs/task170.onnx."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(os.environ.get("PROJECT_DIR", Path.cwd()))
SOURCE = ROOT / "repairs" / "task170.onnx"
OUT = ROOT / "scratch_onnx" / "task170_compact.onnx"


def topo_sort(nodes, initializer_names, input_names):
    available = set(initializer_names) | set(input_names) | {""}
    remaining = list(nodes)
    ordered = []
    while remaining:
        for i, node in enumerate(remaining):
            if all(x in available for x in node.input):
                ordered.append(node)
                available.update(x for x in node.output if x)
                remaining.pop(i)
                break
        else:
            missing = [(n.op_type, [x for x in n.input if x not in available]) for n in remaining]
            raise RuntimeError(f"topological-sort failure: {missing[:8]}")
    return ordered


def build():
    model = onnx.load(SOURCE)

    # Stable after promotion: do not apply the same rewrite twice.
    if any(n.output and n.output[0] == "row_count_1d" for n in model.graph.node):
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, OUT)
        print(OUT)
        return model

    nodes = [copy.deepcopy(n) for n in model.graph.node]

    # Sum foreground channels per row exactly as before, but omit the known
    # singleton batch dimension inside Einsum. This removes the charged boolean
    # Squeeze tensor without changing the foreground/background semantics.
    nodes = [n for n in nodes if "row_any" not in n.output]
    for node in nodes:
        if "row_count_2d" in node.output:
            node.output[0] = "row_count_1d"
            for attr in node.attribute:
                if attr.name == "equation":
                    attr.s = b"nchw,c->h"
        if "row_any_2d" in node.output:
            node.input[0] = "row_count_1d"
            node.output[0] = "row_any"

    # Reduce all singleton/row axes at once; the following Squeeze carried an
    # additional charged INT64 scalar without changing the value.
    nodes = [n for n in nodes if "shape_left_i64" not in n.output]
    for node in nodes:
        if "shape_left_2d" in node.output:
            node.output[0] = "shape_left_i64"
            for attr in node.attribute:
                if attr.name == "axes":
                    del attr.ints[:]
                    attr.ints.extend([0, 1, 2])
                elif attr.name == "keepdims":
                    attr.i = 0

    # A strided Slice is the same column selection as Range+Gather but avoids
    # materializing the four INT32 column indices.
    nodes = [n for n in nodes if not any(o in {"shape_col_idx", "shape_bg"} for o in n.output)]
    nodes.extend(
        [
            helper.make_node("Unsqueeze", ["shape_left", "i64_0_vec"], ["shape_left_vec"]),
            helper.make_node(
                "Unsqueeze", ["shape_sample_col_end", "i64_0_vec"], ["shape_col_end_vec"]
            ),
            helper.make_node(
                "Slice",
                ["shape_bg_rows", "shape_left_vec", "shape_col_end_vec", "i32_3_vec", "smag_vec"],
                ["shape_bg"],
            ),
        ]
    )

    # [4,1] AND [4] broadcasts to [4,4]; the second Unsqueeze and both bulky
    # three-axis initializers are unnecessary.
    nodes = [
        n
        for n in nodes
        if not any(o in {"valid_row_4d", "valid_col_4d", "valid_4x4"} for o in n.output)
    ]
    nodes.extend(
        [
            helper.make_node("Unsqueeze", ["valid4", "axis1_i64"], ["valid_row_2d"]),
            helper.make_node("And", ["valid_row_2d", "valid4"], ["valid_4x4"]),
        ]
    )

    # Keep coordinates that are only consumed as Slice/Concat vectors in their
    # one-element form from the start, eliminating redundant Unsqueeze outputs.
    vector_arg_outputs = {"shape_top_i64", "last_occ_row_i64", "patch_left_i64"}
    for node in nodes:
        if any(o in vector_arg_outputs for o in node.output):
            for attr in node.attribute:
                if attr.name == "keepdims":
                    attr.i = 1
    drop_vectors = {
        "row_window_start_vec", "row_window_end_vec", "shape_sample_end_vec", "smag_vec",
        "last_occ_row_vec", "patch_top_vec", "patch_left_vec", "shape_col_end_vec",
    }
    nodes = [n for n in nodes if not any(o in drop_vectors for o in n.output)]
    replacements = {
        "row_window_start_vec": "shape_top",
        "row_window_end_vec": "row_window_end",
        "shape_sample_end_vec": "shape_sample_end",
        "smag_vec": "smag",
        "last_occ_row_vec": "last_occ_row_i64",
        "patch_top_vec": "patch_top",
        "patch_left_vec": "patch_left",
        "shape_col_end_vec": "shape_sample_col_end",
    }
    for node in nodes:
        for i, name in enumerate(node.input):
            if name in replacements:
                node.input[i] = replacements[name]

    # Width 29 includes the first padded zero after every generator grid
    # (maximum true width 28), which ArgMin needs on masked placeholder rows.
    for node in nodes:
        if "shape_starts" in node.output:
            node.input.append("i32_0_vec")
        if "shape_ends" in node.output:
            node.input.append("i32_29_vec")
        if "shape_steps" in node.output:
            node.input.append("i32_1_vec")

    # ArgMin(row_window) finds the first gap except for the all-occupied length
    # 16 case. Handle that case with a one-byte ReduceMin predicate instead of
    # concatenating and charging a 17-element sentinel window.
    nodes = [n for n in nodes if "row_window_z" not in n.output]
    for node in nodes:
        if "row_len_i64" in node.output:
            node.input[0] = "row_window"
            node.output[0] = "row_zero_i64"
    nodes.extend(
        [
            helper.make_node("ReduceMin", ["row_window"], ["row_window_min"], axes=[0], keepdims=0),
            helper.make_node("Cast", ["row_window_min"], ["row_window_full"], to=TensorProto.BOOL),
            helper.make_node(
                "Where", ["row_window_full", "i64_16", "row_zero_i64"], ["row_len_i64"]
            ),
        ]
    )

    inits = [
        copy.deepcopy(x)
        for x in model.graph.initializer
        if x.name not in {"_axes_4", "_axes_6", "_axes_7", "shape_axes"}
    ]
    inits = [x for x in inits if x.name != "zero_u8_vec"]
    inits.extend(
        [
            numpy_helper.from_array(np.asarray([3], dtype=np.int32), "i32_3_vec"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axis1_i64"),
            numpy_helper.from_array(np.asarray(16, dtype=np.int64), "i64_16"),
            numpy_helper.from_array(np.asarray([29], dtype=np.int32), "i32_29_vec"),
            numpy_helper.from_array(np.asarray([1, 2, 3], dtype=np.int32), "shape_axes"),
        ]
    )
    del model.graph.initializer[:]
    model.graph.initializer.extend(inits)

    removed_vi = {
        "row_count_2d", "row_any_2d", "row_any", "shape_left_2d", "shape_col_idx",
        "valid_row_4d", "valid_col_4d", "valid_4x4",
    }
    removed_vi.update(drop_vectors)
    removed_vi.update(
        {
            "shape_top_i64", "shape_top", "row_window_end", "shape_sample_end",
            "last_occ_row_i64", "last_occ_row", "last_minus3", "has_four_rows",
            "out_size", "out_size_minus1", "patch_top", "smag", "shape_sample_span",
            "shape_sample_col_end", "patch_left_i64", "patch_left",
            "row_window_z", "shape_bg_rows", "shape_starts", "shape_ends", "shape_steps",
        }
    )
    kept_vi = [copy.deepcopy(x) for x in model.graph.value_info if x.name not in removed_vi]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)
    model.graph.value_info.extend(
        [
            helper.make_tensor_value_info("row_count_1d", TensorProto.FLOAT, [30]),
            helper.make_tensor_value_info("row_any", TensorProto.BOOL, [30]),
            helper.make_tensor_value_info("shape_left_vec", TensorProto.INT32, [1]),
            helper.make_tensor_value_info("shape_col_end_vec", TensorProto.INT32, [1]),
            helper.make_tensor_value_info("valid_row_2d", TensorProto.BOOL, [4, 1]),
            helper.make_tensor_value_info("valid_4x4", TensorProto.BOOL, [4, 4]),
            helper.make_tensor_value_info("row_zero_i64", TensorProto.INT64, []),
            helper.make_tensor_value_info("row_window_min", TensorProto.UINT8, []),
            helper.make_tensor_value_info("row_window_full", TensorProto.BOOL, []),
            helper.make_tensor_value_info("shape_bg_rows", TensorProto.FLOAT, [1, 1, 4, 29]),
            helper.make_tensor_value_info("shape_starts", TensorProto.INT32, [3]),
            helper.make_tensor_value_info("shape_ends", TensorProto.INT32, [3]),
            helper.make_tensor_value_info("shape_steps", TensorProto.INT32, [3]),
        ]
    )

    nodes = topo_sort(
        nodes,
        {x.name for x in model.graph.initializer},
        {x.name for x in model.graph.input},
    )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUT)
    print(OUT)
    return model


model = build()
