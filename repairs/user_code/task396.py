"""Build the compact exact task396 renderer from repairs/task396.onnx.

Task invariants used by the repair and confirmed across the full generated set:

* background is the strictly most frequent color;
* exactly two non-background colors are present with distinct counts;
* the vertical threshold vector is already the prefix mask for output rows.

The rewrite therefore selects both colors with one float32 TopK, uses the
vertical prefix directly, and removes locator tensors whose reduced axes are
known singleton dimensions.  It uses no negative pads.
"""

from pathlib import Path
import copy
import math

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path.cwd()
SOURCE = ROOT / "repairs" / "task396.onnx"
DESTINATION = ROOT / "other_model_onnx" / "task396.onnx"


def producer(model: onnx.ModelProto, output_name: str):
    for node in model.graph.node:
        if output_name in node.output:
            return node
    raise KeyError(output_name)


def replace_attributes(node, **attributes) -> None:
    del node.attribute[:]
    node.attribute.extend(
        helper.make_attribute(name, value) for name, value in attributes.items()
    )


def value_info(name: str, elem_type: int, dims):
    return helper.make_tensor_value_info(name, elem_type, dims)


def build() -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load(SOURCE))

    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([3], dtype=np.int64), "top3_i64"),
            numpy_helper.from_array(np.array(1, dtype=np.int64), "rank1_i64"),
            numpy_helper.from_array(np.array(2, dtype=np.int64), "rank2_i64"),
            numpy_helper.from_array(
                np.array([1, 1, 8, 1], dtype=np.int64), "shape1181_i64"
            ),
            numpy_helper.from_array(np.array([18], dtype=np.int64), "eighteen_i64"),
        ]
    )
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == "mpow":
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(
                    np.array([128, 192, 224, 240, 248, 252, 254, 255], dtype=np.uint8),
                    "mpow",
                )
            )
        elif initializer.name == "Kh":
            compact_kernel = np.array(
                [[[[128, 64, 32, 16, 8, 4, 2, 1]]]], dtype=np.uint8
            )
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(compact_kernel, "Kh")
            )
        elif initializer.name == "c1818":
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(
                    np.array([14, 18], dtype=np.int64), "c1818"
                )
            )

    # All selected frame starts lie in coordinates 0..13.  Positions beyond
    # that cannot win either maximum, so do not materialize them.
    hbits = producer(model, "Hbits")
    replace_attributes(hbits, pads=[0, 0, 0, 3])
    vbits = producer(model, "Vbits")
    replace_attributes(vbits, pads=[0, 0, 3, 0])
    producer(model, "Sh").input[0] = "Qh"
    producer(model, "Sv").input[0] = "Qv"

    # The threshold comparison produces [True] * height + [False] * remainder.
    # Reshape it directly into the row mask instead of recounting and comparing.
    sv_to_row = producer(model, "Svf")
    sv_to_row.op_type = "Transpose"
    del sv_to_row.input[:]
    sv_to_row.input.extend(["Sv"])
    sv_to_row.output[0] = "rowok"
    replace_attributes(sv_to_row, perm=[0, 1, 3, 2])

    # Horizontal location: reduce all singleton/batch/row axes so ArgMax can
    # directly produce the rank-one Slice/Gather index.
    hcol = producer(model, "HcolMax")
    replace_attributes(hcol, axes=[0, 1, 2], keepdims=0)
    bcol = producer(model, "bcol")
    replace_attributes(bcol, axis=0, keepdims=1)
    bcol.output[0] = "bcol1"

    # Read the vertical signature as a separate 18x1 float Slice from the free
    # input, then quantize it.  This lets the main selected-color crop be only
    # 14x18 instead of 18x18.
    colvec_slice = producer(model, "colvec")
    colvec_slice.op_type = "Slice"
    del colvec_slice.input[:]
    colvec_slice.input.extend(["input", "v_starts", "v_ends", "axes123"])
    colvec_slice.output[0] = "colvec_f"
    del colvec_slice.attribute[:]

    # Vbits has width one, so reducing that axis before ArgMax is an identity.
    brow = producer(model, "brow")
    brow.input[0] = "Vbits"
    replace_attributes(brow, axis=2, keepdims=0)

    removed_outputs = {
        "cnt_box",
        "boxcolor",
        "zerocnt",
        "cnt_lc",
        "lc",
        "lc_u8",
        "tall_f",
        "tall_i",
        "VrowMax",
        "bcol",
        "Qhf",
        "Qvf",
    }
    kept_nodes = []
    for node in model.graph.node:
        if any(output in removed_outputs for output in node.output):
            continue
        if node.op_type == "Reshape" and "bcol" in node.input:
            continue
        if node.op_type == "Less" and "rowiota8" in node.input:
            continue
        kept_nodes.append(node)

    color_nodes = [
        helper.make_node(
            "TopK",
            ["cnt", "top3_i64"],
            ["count_top3", "color_top3"],
            axis=1,
            largest=1,
            sorted=1,
        ),
        helper.make_node(
            "Gather", ["color_top3", "rank1_i64"], ["boxcolor"], axis=1
        ),
        helper.make_node(
            "Cast", ["color_top3"], ["color_top3_u8"], to=TensorProto.UINT8
        ),
        helper.make_node(
            "Gather", ["color_top3_u8", "rank2_i64"], ["lc_u8"], axis=1
        ),
    ]
    final_nodes = []
    for node in kept_nodes:
        if "colvec_f" in node.output:
            final_nodes.extend(
                [
                    helper.make_node(
                        "Add", ["bcol1", "one_i"], ["bcolp1"]
                    ),
                    helper.make_node(
                        "Concat",
                        ["boxcolor", "ch0", "bcol1"],
                        ["v_starts"],
                        axis=0,
                    ),
                    helper.make_node(
                        "Concat",
                        ["boxp1", "eighteen_i64", "bcolp1"],
                        ["v_ends"],
                        axis=0,
                    ),
                ]
            )
        final_nodes.append(node)
        if "colvec_f" in node.output:
            final_nodes.append(
                helper.make_node("Cast", ["colvec_f"], ["colvec"], to=TensorProto.UINT8)
            )
        if "cnt" in node.output:
            final_nodes.extend(color_nodes)
    del model.graph.node[:]
    model.graph.node.extend(final_nodes)

    stale_value_info = removed_outputs | {
        "Svf",
        "rowok",
        "HcolMax",
        "bcol1",
        "brow",
        "Hbits",
        "Vbits",
        "M0",
        "Mu8",
        "colvec",
    }
    kept_value_info = [
        item for item in model.graph.value_info if item.name not in stale_value_info
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    model.graph.value_info.extend(
        [
            value_info("count_top3", TensorProto.FLOAT, [1, 3]),
            value_info("color_top3", TensorProto.INT64, [1, 3]),
            value_info("boxcolor", TensorProto.INT64, [1]),
            value_info("color_top3_u8", TensorProto.UINT8, [1, 3]),
            value_info("lc_u8", TensorProto.UINT8, [1]),
            value_info("rowok", TensorProto.BOOL, [1, 1, 8, 1]),
            value_info("HcolMax", TensorProto.UINT8, [14]),
            value_info("bcol1", TensorProto.INT64, [1]),
            value_info("brow", TensorProto.INT64, [1, 1, 1]),
            value_info("Hbits", TensorProto.UINT8, [1, 1, 14, 14]),
            value_info("Vbits", TensorProto.UINT8, [1, 1, 14, 1]),
            value_info("M0", TensorProto.FLOAT, [1, 1, 14, 18]),
            value_info("Mu8", TensorProto.UINT8, [1, 1, 14, 18]),
            value_info("bcolp1", TensorProto.INT64, [1]),
            value_info("v_starts", TensorProto.INT64, [3]),
            value_info("v_ends", TensorProto.INT64, [3]),
            value_info("colvec_f", TensorProto.FLOAT, [1, 1, 18, 1]),
            value_info("colvec", TensorProto.UINT8, [1, 1, 18, 1]),
        ]
    )

    produced = {output for node in model.graph.node for output in node.output if output}
    live_value_info = [
        item for item in model.graph.value_info if item.name in produced
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(live_value_info)

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
    params = sum(math.prod(initializer.dims) for initializer in model.graph.initializer)
    print(f"{DESTINATION} nodes={len(model.graph.node)} params={params}")
