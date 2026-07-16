from pathlib import Path

import onnx
import numpy as np
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repairs" / "task341.onnx"
DESTINATION = ROOT / "other_model_onnx" / "task341.onnx"


def main() -> None:
    source = onnx.load(SOURCE)
    old = list(source.graph.node)

    nodes = list(old[:16])

    # row_bg is bounded by 8. Therefore max(row_bg) == 8 is exactly the
    # existing "any empty row" test, but it avoids bool->uint8->bool casts.
    nodes.append(
        helper.make_node(
            "ReduceMax",
            ["row_bg", "axis2"],
            ["row_max"],
            name="row_has_empty_reduce",
            keepdims=1,
        )
    )
    nodes.append(
        helper.make_node(
            "Equal",
            ["row_max", "eight_f"],
            ["any_row_empty"],
            name="row_has_empty",
        )
    )
    nodes.extend(old[19:24])
    nodes.append(old[25])  # col_empty -> col_empty6

    # Exact boolean mux, with no separately materialized horizontal flag:
    # selected = fallback XOR (vertical AND (primary XOR fallback)).
    nodes.extend(
        [
            helper.make_node(
                "Xor",
                ["row_empty6", "row_inner6"],
                ["row_choice_delta"],
                name="row_choice_delta",
            ),
            helper.make_node(
                "And",
                ["vertical", "row_choice_delta"],
                ["row_vertical_delta"],
                name="row_vertical_delta",
            ),
            helper.make_node(
                "Xor",
                ["row_inner6", "row_vertical_delta"],
                ["bridge_rows6"],
                name="select_bridge_rows",
            ),
            helper.make_node(
                "Xor",
                ["col_inner6", "col_empty6"],
                ["col_choice_delta"],
                name="col_choice_delta",
            ),
            helper.make_node(
                "And",
                ["vertical", "col_choice_delta"],
                ["col_vertical_delta"],
                name="col_vertical_delta",
            ),
            helper.make_node(
                "Xor",
                ["col_empty6", "col_vertical_delta"],
                ["bridge_cols6"],
                name="select_bridge_cols",
            ),
        ]
    )
    # Avoid the 900-byte full-grid boolean mask entirely. The final Einsum
    # emits logits directly (and is therefore exempt as the named output):
    # normal colors use a rank-2 exact argmax code, while output color 8 gets
    # logit 1 only at the Cartesian product of the selected bridge vectors.
    nodes.extend(
        [
            helper.make_node(
                "Cast",
                ["bridge_rows6"],
                ["bridge_rows_f"],
                name="bridge_rows_float",
                to=onnx.TensorProto.FLOAT,
            ),
            helper.make_node(
                "Concat",
                ["eight_f", "bridge_rows_f"],
                ["row_features"],
                name="build_row_features",
                axis=2,
            ),
            helper.make_node(
                "Cast",
                ["bridge_cols6"],
                ["bridge_cols_f"],
                name="bridge_cols_float",
                to=onnx.TensorProto.FLOAT,
            ),
            helper.make_node(
                "Concat",
                ["eight_f", "bridge_cols_f"],
                ["col_features"],
                name="build_col_features",
                axis=3,
            ),
            helper.make_node(
                "Einsum",
                [
                    "input",
                    "output_code",
                    "input_code",
                    "state_weights",
                    "state_features",
                    "row_features",
                    "spatial_map",
                    "state_features",
                    "col_features",
                    "spatial_map",
                ],
                ["output"],
                name="direct_argmax_output",
                equation="bchw,ok,ck,ks,sj,bxjy,jh,sl,bxyl,lw->bohw",
            ),
        ]
    )

    colors = np.arange(10, dtype=np.float32) + 1
    # 0.5 - (output_color - input_color)^2 is positive only for the exact
    # color. At a bridge cell the three input features [1, 1, 1] are scaled
    # to [1, 9, 81], precisely the feature code for color 8.
    input_code = np.stack([np.ones(10), colors, colors**2], axis=1).astype(np.float32)
    output_code = np.stack(
        [0.5 - colors**2, 2 * colors, -np.ones(10)], axis=1
    ).astype(np.float32)
    state_weights = np.array([[1, 0], [1, 8], [1, 80]], dtype=np.float32)
    state_features = np.zeros((2, 7), dtype=np.float32)
    state_features[0, 0] = 1
    state_features[1, 1:] = 1
    spatial_map = np.zeros((7, 30), dtype=np.float32)
    # Feature zero is the existing value-8 initializer, so 1/8 normalizes it
    # to the constant spatial factor without adding another parameter.
    spatial_map[0, :] = 1 / 8
    for i in range(6):
        spatial_map[i + 1, i + 2] = 1

    final_initializers = [
        numpy_helper.from_array(input_code, "input_code"),
        numpy_helper.from_array(output_code, "output_code"),
        numpy_helper.from_array(state_weights, "state_weights"),
        numpy_helper.from_array(state_features, "state_features"),
        numpy_helper.from_array(spatial_map, "spatial_map"),
    ]

    # Broadcast-compatible everywhere it is used, and now also rank-compatible
    # with the two feature Concat nodes.
    reshaped_initializers = []
    for initializer in source.graph.initializer:
        if initializer.name == "eight_f":
            reshaped_initializers.append(
                numpy_helper.from_array(np.full((1, 1, 1, 1), 8, dtype=np.float32), "eight_f")
            )
        elif initializer.name not in {"cyan", "pad18_pads_33_13", "pad18_axes_33_14"}:
            reshaped_initializers.append(initializer)

    produced = {name for node in nodes for name in node.output}
    value_info = [vi for vi in source.graph.value_info if vi.name in produced]
    graph = helper.make_graph(
        nodes,
        source.graph.name + "_exact_compact",
        list(source.graph.input),
        list(source.graph.output),
        initializer=reshaped_initializers + final_initializers,
        value_info=value_info,
    )
    candidate = helper.make_model(
        graph,
        producer_name="codex-task341-exact-compact",
        opset_imports=list(source.opset_import),
        ir_version=source.ir_version,
    )
    candidate.doc_string = (
        "Exact task341 rewrite: direct row maximum, XOR boolean muxes, and "
        "an atomic factorized argmax output with no full-grid mask."
    )
    candidate = onnx.shape_inference.infer_shapes(candidate)
    onnx.checker.check_model(candidate, full_check=True)
    onnx.save(candidate, DESTINATION)
    print(DESTINATION)


if __name__ == "__main__":
    main()
