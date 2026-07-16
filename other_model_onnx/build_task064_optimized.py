"""Build the compact, exact task064 model from repairs/task064.onnx.

The original graph materializes the union of the rectangle and marker as a
float32 30x30 tensor solely to obtain its first/last occupied coordinates.
Here, an exponential positional code reduces each row/column directly to two
scalars.  Natural log recovers the extrema with a wide numerical margin.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repairs" / "task064.onnx"
OUTPUT = ROOT / "other_model_onnx" / "task064.onnx"


def init(name, value, dtype):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def build():
    model = onnx.load(SOURCE)
    graph = model.graph

    # Keep color discovery (nodes 0..4) and the exact rectangle/marker logic
    # (old nodes 18 onward).  Replace the large union mask and four ArgMaxes.
    # Node 4 (twohot) was only needed by the removed union-mask Einsum.
    prefix = list(graph.node[:4])
    suffix = list(graph.node[18:])

    eps = np.float32(0.02)
    forward = np.zeros(30, dtype=np.float32)
    reverse = np.zeros(30, dtype=np.float32)
    forward[:24] = np.exp(np.arange(24, dtype=np.float32) + eps)
    reverse[:24] = np.exp(np.arange(23, -1, -1, dtype=np.float32) + eps)
    weights = np.stack([forward, reverse])

    graph.initializer.extend(
        [
            init("exp_pos", weights, np.float32),
            init("one_code", np.float32(1.0), np.float32),
            init("dir0", [0], np.int64),
            init("dir1", [1], np.int64),
            init("u23", np.uint8(23), np.uint8),
            init("shape_h", [1, 1, 24, 1], np.int64),
            init("shape_v", [1, 1, 1, 24], np.int64),
            init("singleton", [1.0], np.float32),
            init("axes23", [2, 3], np.int64),
            init("u0", np.uint8(0), np.uint8),
            init("shape_111", [1, 1, 1], np.int64),
        ]
    )

    replacement = [
        # Sum over both candidate colors and encode rightmost/leftmost columns.
        helper.make_node(
            "Einsum", ["input", "cand_oh", "exp_pos", "singleton"], ["h_code"],
            equation="bchw,ec,dw,z->bdhz",
        ),
        helper.make_node("Max", ["h_code", "one_code"], ["h_safe"]),
        helper.make_node("Log", ["h_safe"], ["h_log"]),
        helper.make_node("Cast", ["h_log"], ["h_exp"], to=TensorProto.UINT8),
        helper.make_node("Slice", ["h_exp", "ax0", "s24", "ax2"], ["h24"]),
        helper.make_node("Gather", ["h24", "dir0"], ["R24"], axis=1),
        helper.make_node("Gather", ["h24", "dir1"], ["Lrev"], axis=1),
        helper.make_node("Sub", ["u23", "Lrev"], ["L24"]),

        # The same positional encoding along rows gives bottom/top coordinates.
        helper.make_node(
            "Einsum", ["input", "cand_oh", "exp_pos", "singleton"], ["v_code"],
            equation="bchw,ec,dh,z->bdzw",
        ),
        helper.make_node("Max", ["v_code", "one_code"], ["v_safe"]),
        helper.make_node("Log", ["v_safe"], ["v_log"]),
        helper.make_node("Cast", ["v_log"], ["v_exp"], to=TensorProto.UINT8),
        helper.make_node("Slice", ["v_exp", "ax0", "s24", "k3"], ["v24"]),
        helper.make_node("Gather", ["v24", "dir0"], ["B24"], axis=1),
        helper.make_node("Gather", ["v24", "dir1"], ["Trev"], axis=1),
        helper.make_node("Sub", ["u23", "Trev"], ["T24"]),
    ]

    # Produce row/column projections with their broadcast singleton already in
    # place.  This removes two charged Reshape tensors later in the graph.
    suffix[0] = helper.make_node(
        "Einsum", ["input", "cand_oh", "singleton"], ["rowproj"],
        equation="bchw,ec,d->behd",
    )
    suffix[1] = helper.make_node(
        "Einsum", ["input", "cand_oh", "singleton"], ["colproj"],
        equation="bchw,ec,d->bedw",
    )
    suffix[6] = helper.make_node(
        "ReduceSum", ["pr_f", "axes23"], ["rspan"], keepdims=0,
    )
    suffix[7] = helper.make_node(
        "ReduceSum", ["pc_f", "axes23"], ["cspan"], keepdims=0,
    )
    # Equal broadcasts [2] candidate counts against [1,2] areas directly.
    suffix[12] = helper.make_node("Equal", ["cc_f", "area"], ["isbox_b"])
    del suffix[11]
    # Giving OneHot a [1,1,1] index directly produces [1,10,1,1] at axis 1.
    suffix[16] = helper.make_node("Reshape", ["mk_idx", "shape_111"], ["mk_idx3"])
    suffix[17] = helper.make_node(
        "OneHot", ["mk_idx3", "oh_depth", "oh_vals"], ["mk_sel"], axis=1,
    )

    # Replace the selected-row/column reshape+cast chain and the small-mask
    # arithmetic with boolean Where gates.  The two large 24x24 range tests are
    # unchanged and remain exact unsigned interval checks.
    compact_tail = [
        helper.make_node("Gather", ["pr_b", "box_pos"], ["brow_b"], axis=1),
        helper.make_node("Slice", ["brow_b", "ax0", "s24", "ax2"], ["BR"]),
        helper.make_node("Gather", ["pc_b", "box_pos"], ["bcol_b"], axis=1),
        helper.make_node("Slice", ["bcol_b", "ax0", "s24", "k3"], ["BC"]),
        helper.make_node("Sub", ["R24", "L24"], ["wid0_h"]),
        helper.make_node("Where", ["BR", "L24", "u30"], ["lo_h"]),
        helper.make_node("Where", ["BR", "wid0_h", "u0"], ["wid_h"]),
        helper.make_node("Where", ["BC", "u100", "col_coord"], ["colb"]),
        helper.make_node("Sub", ["B24", "T24"], ["wid0_v"]),
        helper.make_node("Where", ["BC", "T24", "u30"], ["lo_v"]),
        helper.make_node("Where", ["BC", "wid0_v", "u0"], ["wid_v"]),
        helper.make_node("Where", ["BR", "u100", "row_coord"], ["rowb"]),
        helper.make_node("Sub", ["colb", "lo_h"], ["Dh"]),
        helper.make_node("LessOrEqual", ["Dh", "wid_h"], ["Hm"]),
        helper.make_node("Sub", ["rowb", "lo_v"], ["Dv"]),
        helper.make_node("LessOrEqual", ["Dv", "wid_v"], ["Vm"]),
        helper.make_node("Or", ["Hm", "Vm"], ["fill24"]),
        helper.make_node("Pad", ["fill24", "pad_pads"], ["fill30"], mode="constant"),
        helper.make_node("Where", ["fill30", "mk_sel", "input"], ["output"]),
    ]

    # suffix[0:18] ends at mk_sel after removing the redundant cc2 Reshape.
    # replaced by compact_tail.
    del graph.node[:]
    graph.node.extend(prefix + replacement + suffix[:18] + compact_tail)

    # Drop initializers used only by the removed mask/ArgMax/Cast chain.
    used = {name for node in graph.node for name in node.input if name}
    kept = [x for x in graph.initializer if x.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)

    model.ir_version = 10
    del model.opset_import[:]
    model.opset_import.extend([helper.make_opsetid("", 21)])
    onnx.checker.check_model(model, full_check=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Some repository artifacts may be hard-linked.  Unlink the destination
    # before saving so writing a candidate can never mutate repairs/task064.onnx.
    if OUTPUT.exists():
        OUTPUT.unlink()
    onnx.save(model, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
