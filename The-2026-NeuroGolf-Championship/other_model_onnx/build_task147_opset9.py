"""Build task147 with attribute Slice and the standard opset-9 Scatter.

The graph still applies the exact local rule: a color-3 cell becomes color 8
iff it has an orthogonal color-3 neighbor.  Scatter indices are unique for all
meaningful updates, so legacy Scatter is equivalent to ScatterElements(add)
here while allowing Slice bounds to live in free node attributes.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submissions" / "neurogolf7300" / "task147.onnx"
OUT = ROOT / "other_model_onnx" / "task147_opset9.onnx"


def build():
    model = onnx.load(SOURCE)
    old = list(model.graph.node)
    conv = next(n for n in old if n.op_type == "Conv")
    scatter = next(n for n in old if n.op_type == "ScatterElements")

    nodes = [
        helper.make_node(
            "Slice", ["input"], ["c3_f"],
            starts=[0, 3, 0, 0], ends=[1, 4, 6, 6], axes=[0, 1, 2, 3],
        ),
        conv,
        helper.make_node("Scatter", list(scatter.input), ["output"], axis=1),
    ]
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    used = {name for node in nodes for name in node.input}
    kept = [x for x in model.graph.initializer if x.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    kernel = next(x for x in model.graph.initializer if x.name == "delta_kernel")
    values = numpy_helper.to_array(kernel).copy()
    values[0, 0, 1, 1] = np.float32(0.5)
    model.graph.initializer.remove(kernel)
    model.graph.initializer.append(numpy_helper.from_array(values, "delta_kernel"))
    del model.opset_import[:]
    model.opset_import.extend([helper.make_opsetid("", 9)])
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUT)
    print(OUT)
