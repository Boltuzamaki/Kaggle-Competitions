"""Build the compact exact task128 vertical-shift renderer.

Task rule: every bottom-aligned solid rectangle of height h is moved upward
by h rows.  The 16 possible vertical input states are represented by a
shared 2x2 latent row basis.  Transposing those latent indices on the output
side supplies the required non-symmetric mapping without a dense 4x4 core.
"""

from pathlib import Path
import base64

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path.cwd()
DESTINATION = ROOT / "other_model_onnx" / "task128.onnx"

ROW_BASIS_B64 = (
    "zZjFP2RXuD+q7cA/CXGlvjx2AD8n9ry+I++lP5M5Uj/FzwA/9HHKvgMTkT/gADo/"
    "809FP0mULL/adoQ/FV2YP7YATz/LqTK/Zid4P4axkz86lm4/Qldzv+lU2j7TRp8/"
    "kfd/Pz5qfL+Hh9I+AvujP+V1cz8DM22/oIqNvlaKXj/nru0/tvqVv7MIEMCDnG6/"
    "BFm1PxvDJ79+/gjAhXqUv1tNjT/BP9u+TFLdv+1Rmb+UVUe/1CsQQGUkrr3DeSvA"
    "sIakvwCqyT+faNM/QwJ6v55ze79Lewk/PoX3P2HnLD8iToG/P0BOPyw+PEB9i6K/"
    "yhWhv0L4l7/DmUQ/XeKGPGiGnL/PQ3e/HltJP8BGKj2kdHu/Z/6RvymrBD947jC9"
    "F2Wpv8ldZb/HjUM/a9vVvRyeYr8Etq+/+FXCPgLDyr3gQoG/rgLNv5oDCz/JR1W8"
    "BYyKv2+Wa79KsCQ/LSgDvP1+ib8vyJy/CD8aP28WSrzSGa6/g+SQvzpMXz8qqDM9"
    "zml0vxlVq7/QIus+IVCKvVD3t7+DPJC/23ZwP8kcWz12SXC/lCasv0ve9T5Wxhq9"
    "a7TFv0WieL/2Pl4/3QwuvkA9l7/aK7W/fAkaP+1Gb71WY7q/fMOVv+ZUBz8wknC+"
)
LATENT_WEIGHT_B64 = "qhmTP3YaNcA="


def build() -> onnx.ModelProto:
    row_basis = np.frombuffer(
        base64.b64decode(ROW_BASIS_B64), dtype="<f4"
    ).reshape(30, 2, 2)
    latent_weight = (
        np.frombuffer(base64.b64decode(LATENT_WEIGHT_B64), dtype="<f4")
        * np.float32(8.0)
    )

    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 10, 30, 30]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 10, 30, 30]
    )
    node = helper.make_node(
        "Einsum",
        ["input", "row_basis", "latent_weight", "row_basis"],
        ["output"],
        equation="bchw,hij,j,rji->bcrw",
    )
    graph = helper.make_graph(
        [node],
        "task128_compact",
        [input_info],
        [output_info],
        [
            numpy_helper.from_array(row_basis, "row_basis"),
            numpy_helper.from_array(latent_weight, "latent_weight"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="task128_compact",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


model = build()


if __name__ == "__main__":
    onnx.save(model, DESTINATION)
    print(DESTINATION)
