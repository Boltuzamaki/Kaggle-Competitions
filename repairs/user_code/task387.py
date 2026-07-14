"""Exact scorer-aware rewrites for repairs/task387.onnx."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(os.environ.get("PROJECT_DIR", Path.cwd()))
SOURCE = ROOT / "repairs" / "task387.onnx"
OUT = ROOT / "scratch_onnx" / "task387_compact.onnx"


def K(name, value, dtype):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


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
    # The promoted generator reads repairs/task387.onnx.  Make reruns stable
    # instead of attempting to apply the same graph rewrite a second time.
    if any(x.name == "coord30" for x in model.graph.initializer):
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, OUT)
        print(OUT)
        return model
    nodes = [copy.deepcopy(n) for n in model.graph.node]

    # The top-left/bottom-right diagonal has the larger sum of row*column:
    # (tl + br) - (tr + bl) = (bottom-top)*(right-left) > 0.  One atomic
    # spatial-moment Einsum therefore identifies that color without any dynamic
    # row/column selector. The other color follows from the global color sum.
    drop = {
        "rowsel_b", "rowsel", "lr", "lr2", "colsel2_b", "colsel2",
        "cvec", "cc64", "cc1", "cc1f", "a1", "b1",
    }
    nodes = [n for n in nodes if not any(o in drop for o in n.output)]
    nodes.extend(
        [
            helper.make_node(
                "Einsum",
                ["input", "coord30", "coord30", "fgmask"],
                ["left_score"],
                equation="nchw,h,w,c->nc",
            ),
            helper.make_node("ArgMax", ["left_score"], ["left_color64"], axis=1, keepdims=0),
            helper.make_node("Einsum", ["input", "color_half"], ["color_sum"], equation="nchw,c->n"),
            helper.make_node("Cast", ["color_sum"], ["color_sum64"], to=TensorProto.INT64),
            helper.make_node("Sub", ["color_sum64", "left_color64"], ["right_color64"]),
            helper.make_node("Gather", ["color_ids", "left_color64"], ["a1_full"], axis=1),
            helper.make_node("Squeeze", ["a1_full"], ["a1"]),
            helper.make_node("Gather", ["color_ids", "right_color64"], ["b1_full"], axis=1),
            helper.make_node("Squeeze", ["b1_full"], ["b1"]),
        ]
    )

    # All flattened coordinates are below 324 and are exactly representable in
    # FLOAT16. Keep the geometry branch at two bytes/element, explicitly floor
    # the two divisions to preserve INT32 truncation, then cast only the final
    # ScatterElements index vector back to its required INT32 type.
    extra = []
    for node in nodes:
        if any(o in {"top", "bottom", "left", "right"} for o in node.output):
            for attr in node.attribute:
                if attr.name == "to":
                    attr.i = TensorProto.FLOAT16
        if "wq" in node.output:
            node.output[0] = "wq_raw"
            extra.append(helper.make_node("Floor", ["wq_raw"], ["wq"]))
        if "tq" in node.output:
            node.output[0] = "tq_raw"
            extra.append(helper.make_node("Floor", ["tq_raw"], ["tq"]))
        if "all_idx" in node.output:
            node.output[0] = "all_idx16"
            extra.append(helper.make_node("Cast", ["all_idx16"], ["all_idx"], to=TensorProto.INT32))
    nodes.extend(extra)

    # Form the valid-grid base directly in UINT8. QLinearMatMul is an exact
    # outer product for binary masks with unit scales and zero zero-points.
    # The old And+Cast path materialized both BOOL and UINT8 18x18 canvases.
    drop = {"base2d_b", "base2d"}
    nodes = [n for n in nodes if not any(o in drop for o in n.output)]
    nodes.extend(
        [
            helper.make_node("Cast", ["rmask_c"], ["rmask_u8"], to=TensorProto.UINT8),
            helper.make_node("Cast", ["cmask_b"], ["cmask_u8"], to=TensorProto.UINT8),
            helper.make_node(
                "QLinearMatMul",
                [
                    "rmask_u8",
                    "qscale",
                    "zero_u8",
                    "cmask_u8",
                    "qscale",
                    "zero_u8",
                    "qscale",
                    "zero_u8",
                ],
                ["base2d"],
            ),
        ]
    )

    remove_inits = {
        "arange30",
        "four_i",
        "two_i",
        "eighteen",
        "thirtysix",
        "corner_offsets",
        "class_ids",
        "sh_2",
        "sh_2_1",
        "idx0",
        "idx1",
    }
    inits = [copy.deepcopy(x) for x in model.graph.initializer if x.name not in remove_inits]
    inits.extend(
        [
            K("qscale", 1.0, np.float32),
            K("zero_u8", 0, np.uint8),
            K("color_half", np.arange(10, dtype=np.float32) / 2.0, np.float32),
            K("coord30", np.arange(30, dtype=np.float32), np.float32),
            K("fgmask", [0.0] + [1.0] * 9, np.float32),
            K("four_i", 4.0, np.float16),
            K("two_i", 2.0, np.float16),
            K("eighteen", 18.0, np.float16),
            K("thirtysix", 36.0, np.float16),
            K("corner_offsets", [-19, -18, -17, -1, 0, 1, 17, 18, 19], np.float16),
        ]
    )
    del model.graph.initializer[:]
    model.graph.initializer.extend(inits)

    nodes = topo_sort(
        nodes,
        {x.name for x in model.graph.initializer},
        {x.name for x in model.graph.input},
    )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUT)
    print(OUT)
    return model


model = build()
