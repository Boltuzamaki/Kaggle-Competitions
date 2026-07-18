"""Build the optimized one-Einsum task327 model.

The graph uses a six-component separable spatial basis.  States 0..5 encode
the down-right ray offsets and state 6 supplies a tiled background baseline.
A fitted rank-2 colour code turns those spatial responses into ten logits with
a wide sign margin.  The 6x6 result is projected directly into the 30x30 graph
output, so the scorer charges no intermediate tensor memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "other_model_onnx" / "task327.onnx"


def tensor(name: str, value) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def build_model() -> onnx.ModelProto:
    # A 6-position coordinate is represented by residue modulo 3 and block.
    # R is safely reused on the input because encoded positions >=3 are zero.
    residue = np.zeros((3, 30), dtype=np.float32)
    for position in range(6):
        residue[position % 3, position] = 1.0
    block = np.zeros((2, 30), dtype=np.float32)
    block[0, :3] = 1.0
    block[1, 3:6] = 1.0

    # Four-way rank-6 approximation of A[input_pos, state, block, residue].
    # Its sign noise is absorbed by the fitted colour-code margins below.
    spatial_input = [
        [-0.3502574563, 0.2266811579, 0.4915367067, 0.3936238885, 2.7064161301, 0.3851709962],
        [-0.0141068343, -0.0496874377, -0.0686645061, -0.0137380594, 0.5986787677, 0.0509290732],
        [0.0056656352, 0.0068576271, 0.0008435627, -0.0065265489, 0.0842648074, -0.0001206640],
    ]
    spatial_state = [
        [283.9879150391, 49.8161315918, 52.9732780457, 134.9868469238, -62.0058555603, -165.8793029785],
        [11.4593706131, -10.9150905609, -7.3955664635, -4.7005209923, -13.7203407288, -21.9452304840],
        [-4.8627181053, 1.4587810040, 0.0400407985, -2.3667156696, -1.8728330135, 0.2079680264],
        [-0.3504474163, -0.1840087920, 0.0857315734, 0.1370321363, -0.2391957780, 0.3099912405],
        [0.0043056863, 0.0141325807, -0.0206974056, 0.0044262949, -0.0202350989, 0.0630165562],
        [0.0038493043, -0.0013135602, 0.0013605310, -0.0025848136, -0.0017452778, 0.0042801020],
        [283.6374816895, 49.6321220398, 53.0590095520, 135.1238708496, -62.2450523376, -165.5693206787],
    ]
    spatial_block = [
        [-0.0253355093, -0.0463947281, 0.0483404696, -0.0337010957, 0.0081362901, -0.0200670082],
        [18.2343254089, 21.2848587036, 18.4408264160, -25.0715332031, 3.2141554356, 7.6640329361],
    ]
    spatial_residue = [
        [0.1526766121, -0.2138635963, 0.0242841206, -0.2002477944, -0.0880684033, -0.0048955861],
        [-0.3801601827, 1.5496102571, -1.9768047333, -0.4215204120, -0.6257199645, 2.0663018227],
        [-9.4391727448, -7.0697298050, 14.1513919830, 12.0778007507, -2.8287308216, 15.6275987625],
    ]

    kind = np.zeros((2, 7), dtype=np.float32)
    kind[0, 6] = 1.0
    kind[1, :6] = 1.0
    feature_kind = np.zeros((2, 3), dtype=np.float32)
    feature_kind[0, 0] = 1.0
    feature_kind[1, 1:] = 1.0

    learned_feature = [
        [0.0091095641, 0.0051420629],
        [-0.0484815463, -3.9424631596],
        [1.1195175648, -2.7003459930],
        [2.8449566364, 6.0393338203],
        [4.6786723137, 19.3945732117],
        [6.1260943413, 33.5269927979],
        [7.2511730194, 49.5771255493],
        [7.5295720100, 63.7450141907],
        [6.6848912239, 67.0176620483],
        [5.3038253784, 60.5884704590],
    ]
    learned_classifier = [
        [-17.4713706970, -9.7451114655, 12.0080394745, 15.3059291840, 14.3985738754,
         12.3919944763, 9.5664968491, 5.8400554657, -1.8496739864, -15.7378587723],
        [1.4511255026, -4.9598522186, -3.9572947025, -2.4672498703, -1.6862126589,
         -1.0573582649, -0.4453016818, 0.2092693895, 1.2499908209, 2.7424144745],
    ]
    learned_bias = [
        3.0234730244, -8.8307676315, -18.4999294281, -25.8578968048, -33.2691154480,
        -39.1710319519, -45.7458572388, -55.5872688293, -68.2994613647, -80.6077575684,
    ]
    feature = np.ones((10, 3), dtype=np.float32)
    feature[:, 1:] = np.asarray(learned_feature, dtype=np.float32)
    classifier = np.empty((3, 10), dtype=np.float32)
    classifier[0] = np.asarray(learned_bias, dtype=np.float32)
    classifier[1:] = np.asarray(learned_classifier, dtype=np.float32)

    initializers = [
        tensor("R", residue),
        tensor("B", block),
        tensor("X", spatial_input),
        tensor("Y", spatial_state),
        tensor("T", spatial_block),
        tensor("I", spatial_residue),
        tensor("Q", kind),
        tensor("M", feature_kind),
        tensor("F", feature),
        tensor("G", classifier),
    ]
    equation = (
        "abcd,ec,fd,ez,kz,tz,iz,fp,kp,wp,jp,qk,ql,bl,lh,ir,tr,js,ws->ahrs"
    )
    node = helper.make_node(
        "Einsum",
        [
            "input", "R", "R", "X", "Y", "T", "I", "X", "Y", "T", "I",
            "Q", "M", "F", "G", "R", "B", "R", "B",
        ],
        ["output"],
        equation=equation,
    )
    graph = helper.make_graph(
        [node],
        "task327_direct_low_rank_einsum",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="task327_compact_exact",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    onnx.save(model, OUTPUT)
    parameters = sum(int(np.prod(initializer.dims)) for initializer in model.graph.initializer)
    print(f"saved {OUTPUT}")
    print(f"nodes={len(model.graph.node)} parameters={parameters}")
