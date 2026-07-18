"""Factor task001's selector tensor without changing its function.

The source graph stores a selector ``m`` with shape [2, 30, 3].  Its two
branch matrices have ranks 3 and 2, so the complete tensor has an exact
five-component CP decomposition.  Expanding that decomposition inside the
existing Einsum reduces the parameter count from 190 to 185 while retaining
a single direct-to-output node.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_DIR / "repairs" / "task001.onnx"
DEFAULT_OUTPUT = PROJECT_DIR / "other_model_onnx" / "task001_cp5.onnx"
EXPECTED_EQUATION = "ncab,ndpq,c,d,uri,vpi,usj,wqj->ncrs"
FACTORED_EQUATION = (
    "ncab,ndpq,c,d,uk,rk,ik,vl,pl,il,um,sm,jm,wo,qo,jo->ncrs"
)


def cp5_factors(selector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an exact rank-3 plus rank-2 decomposition of the selector."""
    if selector.shape != (2, 30, 3):
        raise ValueError(f"Expected selector shape (2, 30, 3), got {selector.shape}")

    branch_factor = np.zeros((2, 5), dtype=np.float32)
    output_factor = np.zeros((30, 5), dtype=np.float32)
    input_factor = np.zeros((3, 5), dtype=np.float32)

    offset = 0
    for branch, rank in ((0, 3), (1, 2)):
        left, singular, right = np.linalg.svd(
            selector[branch].astype(np.float64), full_matrices=False
        )
        columns = slice(offset, offset + rank)
        branch_factor[branch, columns] = 1.0
        output_factor[:, columns] = (left[:, :rank] * singular[:rank]).astype(
            np.float32
        )
        input_factor[:, columns] = right[:rank].T.astype(np.float32)
        offset += rank

    reconstructed = np.einsum(
        "uk,rk,ik->uri", branch_factor, output_factor, input_factor
    )
    error = float(np.max(np.abs(reconstructed - selector)))
    if error > 2e-6:
        raise RuntimeError(f"CP-5 reconstruction error is too large: {error}")
    return branch_factor, output_factor, input_factor


def build(source_path: Path) -> onnx.ModelProto:
    source = onnx.load(source_path)
    if len(source.graph.node) != 1 or source.graph.node[0].op_type != "Einsum":
        raise ValueError("Source must be the one-node task001 Einsum model")

    equation = helper.get_attribute_value(source.graph.node[0].attribute[0]).decode()
    if equation != EXPECTED_EQUATION:
        raise ValueError(f"Unexpected source equation: {equation}")

    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in source.graph.initializer
    }
    color = initializers["v"].astype(np.float32)
    selector = initializers["m"].astype(np.float32)
    branch, output, input_basis = cp5_factors(selector)

    graph_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 10, 30, 30]
    )
    graph_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 10, 30, 30]
    )
    operands = [
        "input",
        "input",
        "v",
        "v",
        "branch",
        "position",
        "basis",
        "branch",
        "position",
        "basis",
        "branch",
        "position",
        "basis",
        "branch",
        "position",
        "basis",
    ]
    node = helper.make_node(
        "Einsum", operands, ["output"], equation=FACTORED_EQUATION
    )
    graph = helper.make_graph(
        [node],
        "task001_cp5",
        [graph_input],
        [graph_output],
        [
            numpy_helper.from_array(color, "v"),
            numpy_helper.from_array(branch, "branch"),
            numpy_helper.from_array(output, "position"),
            numpy_helper.from_array(input_basis, "basis"),
        ],
    )
    return helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 13)],
    )


def smoke_test(source_path: Path, candidate: onnx.ModelProto) -> None:
    source_session = ort.InferenceSession(
        source_path.read_bytes(), providers=["CPUExecutionProvider"]
    )
    candidate_session = ort.InferenceSession(
        candidate.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(1)
    for color in range(1, 10):
        for _ in range(16):
            mask = rng.integers(0, 2, size=(3, 3), dtype=np.int8)
            value = np.zeros((1, 10, 30, 30), dtype=np.float32)
            for row in range(3):
                for col in range(3):
                    value[0, color if mask[row, col] else 0, row, col] = 1.0
            expected = source_session.run(None, {"input": value})[0] > 0
            actual = candidate_session.run(None, {"input": value})[0] > 0
            if not np.array_equal(actual, expected):
                raise RuntimeError("Candidate sign output differs from source")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model = build(args.source)
    onnx.checker.check_model(model, full_check=True)
    smoke_test(args.source, model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)

    parameters = sum(int(np.prod(initializer.dims)) for initializer in model.graph.initializer)
    print(f"saved={args.output}")
    print(f"parameters={parameters}")


if __name__ == "__main__":
    main()
