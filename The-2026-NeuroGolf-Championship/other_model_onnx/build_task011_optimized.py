from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repairs" / "task011.onnx"
DESTINATION = ROOT / "other_model_onnx" / "task011.onnx"


def main() -> None:
    source = onnx.load(SOURCE)
    initializers = {x.name: x for x in source.graph.initializer}

    # In each three-row/column data band there are 27 tile cells and six
    # separator-color cells. Weights background=2, separator=1, other=-2 give
    #   2*background - 2*(27-background) + 6 = 4*(background-12),
    # hence exactly zero for ordinary bands and +4 for the unique 13-zero band.
    # The all-separator group remains positive as required by the output grid.
    channel_weights = np.full(10, -2, dtype=np.float32)
    channel_weights[0] = 2
    channel_weights[5] = 1

    # The two band selectors and the selected source pixel are all linear in
    # the input. Contract them together in the final exempt output so neither
    # selector is materialized as a charged tensor.
    node = helper.make_node(
        "Einsum",
        [
            "input",
            "channel_weights",
            "EXP4",
            "input",
            "channel_weights",
            "EXP4",
            "EXP4",
            "EXP4",
            "OFF4",
            "EXP4",
            "EXP4",
            "OFF4",
            "input",
        ],
        ["output"],
        name="direct_selected_tile_expansion",
        equation="npxy,p,xR,nquv,q,vC,rI,aR,Ia,cJ,bC,Jb,nkab->nkrc",
    )
    graph = helper.make_graph(
        [node],
        "task011_atomic_selection_and_expansion",
        list(source.graph.input),
        list(source.graph.output),
        initializer=[
            initializers["EXP4"],
            initializers["OFF4"],
            numpy_helper.from_array(channel_weights, "channel_weights"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="codex-task011-atomic-einsum",
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=source.ir_version,
    )
    model.doc_string = (
        "Exact task011 selection and separator-grid expansion in one Einsum; "
        "band selectors remain internal contractions."
    )
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, DESTINATION)
    print(DESTINATION)


if __name__ == "__main__":
    main()
