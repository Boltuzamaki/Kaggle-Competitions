"""Build the exact task254 extrema-bar model with a shared selector factor.

The puzzle has four or five bottom-aligned gray bars of distinct heights.  The
tallest bar becomes color 1, the shortest becomes color 2, and all other bars
become background.  This builder starts from the verified repair graph but
changes the five-slot polynomial coordinate so that ``rowmap.T`` is one of its
four Lagrange factors.  The separate ``l0`` initializer can then be removed.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repairs" / "task254.onnx"
DEST = ROOT / "other_model_onnx" / "task254.onnx"


def factor_rows(coordinates: list[float], roots: list[list[int]]) -> list[np.ndarray]:
    """Return three affine-factor tables after the shared rowmap factor."""
    tables = [np.empty((5, 2), dtype=np.float32) for _ in range(3)]
    for slot in range(5):
        denominator = 1.0
        for other in range(5):
            if other != slot:
                denominator *= coordinates[slot] - coordinates[other]
        scale = 24.0 / denominator
        for factor, root_slot in enumerate(roots[slot]):
            lead = scale if factor == 0 else 1.0
            tables[factor][slot] = [lead, -lead * coordinates[root_slot]]
    return tables


def build() -> None:
    model = onnx.load(SOURCE)

    # rowmap.T has factors [z - 1] for slots 0..3 and [z + 9] for
    # slot 4.  Map coordinate 1 to slot 4 and -9 to slot 0, making each
    # shared factor vanish at a slot other than the one it selects.
    coordinates = [-9.0, 2.0, 3.0, 4.0, 1.0]
    shared_root = [4, 4, 4, 4, 0]
    remaining_roots = [
        [other for other in range(5) if other not in (slot, shared_root[slot])]
        for slot in range(5)
    ]
    l1, l2, l3 = factor_rows(coordinates, remaining_roots)

    arrays = {init.name: numpy_helper.to_array(init).copy() for init in model.graph.initializer}
    x = arrays["x"]
    for spatial_index in range(10):
        x[0, spatial_index] = coordinates[spatial_index // 2]
    arrays["x"] = x
    arrays["l1"], arrays["l2"], arrays["l3"] = l1, l2, l3

    # The decoupled selector evaluates the intended quadratic directly.  Give
    # output channel 0 a strict negative margin on either selected extreme:
    #   middle bars: d0*d1 - eps*(d0^2+d1^2) > 0
    #   extrema:                         - eps*d^2 < 0
    # Distinct heights in [1, 9] make eps=0.005 safely below the worst middle
    # ratio.  It is also strictly below 1/100, leaving a negative margin in
    # the non-target extreme-color logit after the output projection.
    eps = np.float32(0.005)
    quad = arrays["quad"]
    quad[0, 0, 0, 0] = 1.0 - eps
    quad[1, 1, 0, 0] = 1.0 - eps
    quad[0, 0, 1, 0] = -eps
    quad[1, 1, 1, 0] = -eps
    arrays["quad"] = quad

    kept = []
    for init in model.graph.initializer:
        if init.name == "l0":
            continue
        kept.append(numpy_helper.from_array(arrays[init.name].astype(np.float32), init.name))
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    score = model.graph.node[0]
    score.input[:] = [
        "input", "kc", "rowmap",
        "x", "x", "x", "x", "rowmap", "l1", "l2", "l3",
        "x", "x", "x", "x", "rowmap", "l1", "l2", "l3",
    ]
    score.attribute[0].s = (
        b"nchw,kc,mq,th,uh,vh,sh,tq,qu,qv,qs,aw,bw,dw,ew,aj,jb,jd,je->mj"
    )

    final = model.graph.node[3]
    final.input[:] = [
        "input", "colmask", "colmask",
        "x", "x", "x", "x", "rowmap", "l1", "l2", "l3",
        "kc", "quad", "po",
    ]
    final.attribute[0].s = (
        b"nchw,aj,bj,tw,uw,vw,sw,tj,ju,jv,js,kc,abkp,po->nohw"
    )

    onnx.checker.check_model(model, full_check=True)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, DEST)


if __name__ == "__main__":
    build()
