"""Build scorer-golfed task397 variants from the live repairs artifact.

The source graph already exploits the generator's exact structure: every 10x10
input contains two or three isolated 2x2 objects, and the output adds a two-cell
wide bar of height equal to each object's number of distinct colors.

This pass preserves that algorithm and compresses charged scatter plumbing:

* concatenate the four per-object height flags and gather them directly into
  the 24 update slots;
* keep the update, index, and scatter tensors rank-1;
* retain the protective Where used by the live model.  The third TopK slot is
  invalid on two-object examples, so its false updates must be redirected to a
  safe index before ScatterElements.

The original repairs/task397.onnx is read-only.  Artifacts are written beside
this script under other_model_onnx.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repairs" / "task397.onnx"
OUTPUT = ROOT / "other_model_onnx" / "task397.onnx"


def _init_map(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    return {x.name: x for x in model.graph.initializer}


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
    nodes = list(model.graph.node)

    # The live graph's stable tensor names are part of the task identity.  Fail
    # loudly if the source changes instead of silently patching another graph.
    produced = {out for node in nodes for out in node.output}
    required = {
        "valid",
        "m2",
        "m3",
        "three_new",
        "base_idx",
        "scatter_idx38",
        "bar_b",
        "cond30",
        "output",
    }
    missing = required - produced
    if missing:
        raise RuntimeError(f"task397 source graph changed; missing tensors: {missing}")

    # Retain everything through the three per-object distinct-color predicates,
    # while dropping the old per-flag Reshapes (u1/u2/u3).  Those predicates
    # are interleaved with the equality logic in the source graph.
    cut = next(i for i, node in enumerate(nodes) if "three_new" in node.output) + 1
    prefix = [
        node
        for node in nodes[:cut]
        if not {"u1", "u2", "u3"}.intersection(node.output)
    ]

    # Direct update order for flags laid out as
    # [u1(obj0..2), u2(obj0..2), u3(obj0..2), u4(obj0..2)].  This avoids the
    # old four Reshapes plus Unsqueeze and Expand.
    order = []
    for obj in range(3):
        for height in range(4):
            order.extend([height * 3 + obj, height * 3 + obj])
    _replace_initializer(model, "update_order", np.asarray(order, dtype=np.int64))
    _replace_initializer(model, "zero100_bool", np.zeros(90, dtype=np.bool_))

    tail = [
        helper.make_node(
            "Concat",
            ["valid", "m2", "m3", "three_new"],
            ["flags12"],
            axis=0,
        ),
        helper.make_node(
            "Gather", ["flags12", "update_order"], ["updates"], axis=0
        ),
        helper.make_node("Reshape", ["base_idx", "shape31"], ["base31"]),
        helper.make_node(
            "Add", ["base31", "flat_offsets"], ["scatter_idx38"]
        ),
        helper.make_node(
            "Reshape", ["scatter_idx38", "shape24"], ["scatter_idx"]
        ),
        helper.make_node(
            "Where",
            ["updates", "scatter_idx", "zero_i32"],
            ["scatter_idx_safe"],
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
    model.graph.node.extend(prefix + tail)
    _drop_unused_initializers(model)
    # The source carries materialized value_info entries.  Remove entries for
    # deleted tensors before re-inferring, otherwise the scorer still charges
    # those dead intermediates even though no node produces them.
    del model.graph.value_info[:]

    # Keep the original contract/opset.  Shape inference is materialized so the
    # competition scorer does not have to guess any intermediate shape.
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(model)
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    candidate = build()
    onnx.save(candidate, OUTPUT)
    print(OUTPUT)
