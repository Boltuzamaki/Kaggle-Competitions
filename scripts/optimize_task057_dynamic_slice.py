"""Build the lower-cost task057 dynamic-Slice rewrite.

The source repair locates the 3x3 bounding box correctly, but constructs three
row and three column indices and gathers twice.  Keeping ArgMin's singleton
dimension lets one concatenate [top, left] directly and crop with one Slice.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(source: Path) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load(source))
    graph = model.graph

    by_output = {output: node for node in graph.node for output in node.output if output}
    top = by_output["top_i64"]
    left = by_output["left_i64"]
    for node in (top, left):
        for attr in node.attribute:
            if attr.name == "keepdims":
                attr.i = 1

    replacement = [
        helper.make_node("Concat", ["top_i64", "left_i64"], ["crop_starts"], axis=0),
        helper.make_node("Add", ["crop_starts", "crop_three_i64"], ["crop_ends"]),
        helper.make_node(
            "Slice",
            ["bg", "crop_starts", "crop_ends", "crop_axes_i64"],
            ["bg_patch"],
        ),
    ]

    removed_outputs = {
        "top",
        "left",
        "row_idx",
        "col_idx",
        "bg_rows",
        "bg_patch",
    }
    kept = [
        node
        for node in graph.node
        if not any(output in removed_outputs for output in node.output)
    ]
    insert_at = next(i for i, node in enumerate(kept) if "fgmask3" in node.output)
    kept[insert_at:insert_at] = replacement
    del graph.node[:]
    graph.node.extend(kept)

    kept_initializers = [init for init in graph.initializer if init.name != "offsets3_i32"]
    kept_initializers.extend(
        [
            numpy_helper.from_array(np.asarray(3, dtype=np.int64), "crop_three_i64"),
            numpy_helper.from_array(np.asarray([2, 3], dtype=np.int64), "crop_axes_i64"),
        ]
    )
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)

    replaced_names = {"crop_starts", "crop_ends", "bg_patch"}
    value_info = [vi for vi in graph.value_info if vi.name not in replaced_names]
    value_info.extend(
        [
            helper.make_tensor_value_info("crop_starts", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("crop_ends", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("bg_patch", TensorProto.UINT8, [1, 1, 3, 3]),
        ]
    )
    del graph.value_info[:]
    graph.value_info.extend(value_info)

    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=Path("repairs/task057.onnx"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build(args.source), args.output)


if __name__ == "__main__":
    main()
