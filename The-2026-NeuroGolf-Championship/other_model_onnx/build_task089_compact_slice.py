"""Build a lower-cost exact task089 sprite copier from repairs/task089.onnx.

The expensive padded-17x17 flatten and dynamic 24-index gathers are replaced
with a dynamic 5x5 Slice around each source anchor.  The payload color is read
directly from the already-computed source score, eliminating the redundant
core-color gather.  Red templates use mirrored column offsets; green templates
use direct offsets.
"""

from pathlib import Path
import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
# Immutable copy of the pre-rewrite repairs model.  Keeping it beside the
# builder makes regeneration deterministic after the winner is promoted.
SOURCE = ROOT / "other_model_onnx" / "task089_baseline_source.onnx"
OUTPUT = ROOT / "other_model_onnx" / "task089.onnx"

model = onnx.load(SOURCE)
g = model.graph

# Front end through payload_neighbor4_flat, excluding color17_flat (old node 3).
front = [copy.deepcopy(g.node[i]) for i in [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]]
nodes = list(front)

remove_inits = {
    "offset17",
    "direct_flat_offsets",
    "flipped_flat_offsets",
    "core_indices",
    "shape_289",
    "grid_i32",
    "four_i32",
}
inits = [copy.deepcopy(x) for x in g.initializer if x.name not in remove_inits]

inits.extend(
    [
        numpy_helper.from_array(np.array(13, np.int64), name="grid_i64"),
        numpy_helper.from_array(np.array(5, np.int64), name="five_i64"),
        numpy_helper.from_array(np.array([5], np.int64), name="top5_i64"),
        numpy_helper.from_array(np.array([25], np.int64), name="shape_25"),
    ]
)

offsets = {}
for color in (2, 3):
    vals = []
    for i in range(25):
        dr, dc = divmod(i, 5)
        dr -= 2
        dc -= 2
        vals.append(dr * 13 + (-dc if color == 2 else dc))
    name = f"offsets_c{color}"
    offsets[color] = name
    inits.append(numpy_helper.from_array(np.array(vals, np.int32), name=name))


def node(op, inputs, output, **attrs):
    nodes.append(helper.make_node(op, inputs, [output], **attrs))
    return output


scatter_indices = []
scatter_updates = []
for color, anchor in ((2, "red_anchor_flat"), (3, "green_anchor_flat")):
    p = f"c{color}"
    source = node("Where", [anchor, "payload_neighbor4_flat", "zero_u8"], f"{p}_source")
    payload = node("ReduceMax", [source], f"{p}_payload", keepdims=0)
    present = node("Greater", [payload, "zero_u8"], f"{p}_present")
    source_idx = node("ArgMax", [source], f"{p}_source_idx", axis=0, keepdims=1)

    src_row = node("Div", [source_idx, "grid_i64"], f"{p}_src_row")
    src_col = node("Mod", [source_idx, "grid_i64"], f"{p}_src_col")
    starts = node("Concat", [src_row, src_col], f"{p}_starts", axis=0)
    ends = node("Add", [starts, "five_i64"], f"{p}_ends")
    local_2d = node(
        "Slice", ["color17", starts, ends, "pad18_axes_2_20"], f"{p}_local_2d"
    )
    local = node("Reshape", [local_2d, "shape_25"], f"{p}_local")

    anchor_u8 = node("Cast", [anchor], f"{p}_anchor_u8", to=TensorProto.UINT8)
    source_idx_i32 = node(
        "Cast", [source_idx], f"{p}_source_idx_i32", to=TensorProto.INT32
    )
    target = node(
        "ScatterElements",
        [anchor_u8, source_idx_i32, "zero_u8_vec1"],
        f"{p}_target",
        axis=0,
    )
    first = node(
        "ArgMax", [target], f"{p}_first", axis=0, keepdims=0, select_last_index=0
    )
    last = node(
        "ArgMax", [target], f"{p}_last", axis=0, keepdims=0, select_last_index=1
    )
    first_i32 = node("Cast", [first], f"{p}_first_i32", to=TensorProto.INT32)
    last_i32 = node("Cast", [last], f"{p}_last_i32", to=TensorProto.INT32)

    color_match = node("Equal", [local, payload], f"{p}_color_match")
    valid = node("And", [color_match, present], f"{p}_valid")
    valid_f = node("Cast", [valid], f"{p}_valid_f", to=TensorProto.UINT8)
    top_values = f"{p}_top_values"
    top_indices = f"{p}_top_indices"
    nodes.append(
        helper.make_node(
            "TopK",
            [valid_f, "top5_i64"],
            [top_values, top_indices],
            axis=0,
            largest=1,
            sorted=0,
        )
    )
    selected_offsets = node(
        "Gather", [offsets[color], top_indices], f"{p}_selected_offsets", axis=0
    )
    selected_valid = node(
        "Cast", [top_values], f"{p}_selected_valid", to=TensorProto.BOOL
    )
    safe_offsets = node(
        "Where", [selected_valid, selected_offsets, "zero_i32"], f"{p}_safe_offsets"
    )
    scatter_indices.extend(
        [
            node("Add", [first_i32, safe_offsets], f"{p}_scatter_first"),
            node("Add", [last_i32, safe_offsets], f"{p}_scatter_last"),
        ]
    )
    update = node("Where", [selected_valid, payload, "zero_u8"], f"{p}_updates")
    scatter_updates.extend([update, update])

all_idx = node("Concat", scatter_indices, "all_scatter_idx", axis=0)
all_upd = node("Concat", scatter_updates, "all_scatter_upd", axis=0)
scattered = node(
    "ScatterElements",
    ["color_flat", all_idx, all_upd],
    "scattered",
    axis=0,
    reduction="max",
)
color13_out = node("Reshape", [scattered, "shape_1_1_13_13"], "color13_out")
color30 = node(
    "Pad",
    ["color13_out", "pad18_pads_64_21", "sentinel10_u8", "pad18_axes_2_20"],
    "color30_out",
)
node("Equal", [color30, "color_bank"], "output")

del g.node[:]
g.node.extend(nodes)
del g.initializer[:]
g.initializer.extend(inits)
del g.value_info[:]
out = g.output[0].type.tensor_type
out.elem_type = TensorProto.BOOL
del out.shape.dim[:]
for d in (1, 10, 30, 30):
    out.shape.dim.add().dim_value = d

model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
# Slice starts are dynamic values, but the generator contract and `ends=starts+5`
# make both windows statically [1,1,5,5].  Pin them so the scorer does not
# reject symbolic dimensions after its own strict shape-inference pass.
for vi in model.graph.value_info:
    if vi.name in {"c2_local_2d", "c3_local_2d"}:
        tt = vi.type.tensor_type
        del tt.shape.dim[:]
        for d in (1, 1, 5, 5):
            tt.shape.dim.add().dim_value = d
onnx.checker.check_model(model, full_check=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, OUTPUT)
print(OUTPUT)
