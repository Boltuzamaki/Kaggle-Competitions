"""Build task212 with the source channel transform absorbed into `chan`.

The original one-node tensor network stores `ain @ chan` indirectly.  Replacing
`chan` by that exact product removes the 4x4 `ain` tensor.  The divider channel
projection is recovered from the transformed basis through the existing
`aout` and `rule` tensors plus a three-value selector.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


EQUATION = (
    "biXC,bmYD,gm,Jg,JvN,v,zY,elz,qi,qh,ho,qpa,aty,yu,uR,tw,wX,"
    "sX,rR,pls,plr->boRC"
)


def build(source: Path) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load(source))
    graph = model.graph
    if len(graph.node) != 1 or graph.node[0].op_type != "Einsum":
        raise ValueError("expected the one-node task212 Einsum repair")

    old = {init.name: onnx.numpy_helper.to_array(init) for init in graph.initializer}
    chan = (old["ain"] @ old["chan"]).astype(np.float32)
    aout = np.asarray(
        [
            [2.0, 1.0, -1.0, 0.0],
            [1.0, 2.0, -1.0, 0.0],
            [-1.0 / 16.0, -1.0 / 16.0, 1.0 / 16.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    divider_selector = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    replacements = {
        "chan": numpy_helper.from_array(chan, "chan"),
        "aout": numpy_helper.from_array(aout, "aout"),
        "divider_selector": numpy_helper.from_array(divider_selector, "divider_selector"),
    }
    initializers = []
    for init in graph.initializer:
        if init.name == "ain":
            continue
        initializers.append(replacements.pop(init.name, init))
    initializers.extend(replacements.values())
    del graph.initializer[:]
    graph.initializer.extend(initializers)

    node = graph.node[0]
    del node.input[:]
    node.input.extend(
        [
            "input",
            "input",
            "chan",
            "aout",
            "rule",
            "divider_selector",
            "row_basis",
            "side_coeff",
            "chan",
            "aout",
            "chan",
            "rule",
            "coef",
            "feat",
            "row_basis",
            "feat",
            "row_basis",
            "row_basis",
            "row_basis",
            "side_coeff",
            "side_coeff",
        ]
    )
    for attr in node.attribute:
        if attr.name == "equation":
            attr.s = EQUATION.encode("ascii")
            break
    else:
        node.attribute.append(helper.make_attribute("equation", EQUATION))

    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=Path("repairs/task212.onnx"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build(args.source), args.output)


if __name__ == "__main__":
    main()
