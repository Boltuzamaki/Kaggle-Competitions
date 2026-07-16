"""Build and fully verify a compact, negative-pad-free task352 model."""

import copy
import json
import math
import os
import string
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "neurogolf_utils"))
import neurogolf_utils as ngu  # noqa: E402


def tensor(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name)


def build_model():
    # Quadratic monomials are enough to express (x-y)^2-k^2.  The same
    # coordinate table and the same eight factors are reused for both axes.
    coord = np.arange(30, dtype=np.float32)
    spatial_features = np.stack([np.ones(30, np.float32), coord, coord * coord])
    channel = np.arange(10, dtype=np.float32)
    channel_features = np.stack([np.ones(10, np.float32), channel])

    # A binary scale keeps all polynomial roots exactly representable in fp32.
    divisor = 256.0
    quadratics = []
    for k in range(2, 10):
        q = np.zeros((3, 3), np.float32)
        q[0, 0] = -(k * k) / divisor
        q[2, 0] = 1.0 / divisor
        q[0, 2] = 1.0 / divisor
        q[1, 1] = -2.0 / divisor
        quadratics.append(q)

    def common(d):
        return math.prod((d * d - k * k) / divisor for k in range(2, 10))

    # mode 0: Kronecker delta (center only); mode 1: |x-y| <= 1.
    mode = np.zeros((2, 3, 3), np.float32)
    center_scale = -1.0 / common(0)
    mode[0, 0, 0] = -center_scale
    mode[0, 2, 0] = center_scale
    mode[0, 0, 2] = center_scale
    mode[0, 1, 1] = -2.0 * center_scale
    neighbor_scale = 1.0 / min(common(0), common(1))
    mode[1, 0, 0] = neighbor_scale

    # mode 0 score: 0.5-(output_color-input_color)^2, positive only when equal.
    # mode 1 score: [1-2*(output_color-1)^2] * (1-input_color/2).  The task
    # generator keeps non-trigger colors outside every trigger neighborhood,
    # while triggers are mutually non-adjacent.  Thus the linear input-color
    # factor is 1 on the background cells that must change and 0 on the color-2
    # trigger itself.  This absorbs a 2x10 center gate into the classifier.
    # Factor both mode-dependent channel polynomials into three affine terms.
    # This needs 20 feature values + 3*8 coefficients instead of 30+18.
    channel_factors = [np.zeros((2, 2, 2), np.float32) for _ in range(3)]

    def affine(dst, mode_index, constant=0.0, output=0.0, input_=0.0):
        dst[mode_index, 0, 0] = constant
        dst[mode_index, 1, 0] = output
        dst[mode_index, 0, 1] = input_

    root = math.sqrt(0.5)
    affine(channel_factors[0], 0, root, -1.0, 1.0)
    affine(channel_factors[1], 0, root, 1.0, -1.0)
    affine(channel_factors[2], 0, 1.0)

    affine(channel_factors[0], 1, -math.sqrt(2.0) * (1.0 - root), math.sqrt(2.0))
    affine(channel_factors[1], 1, math.sqrt(2.0) * (1.0 + root), -math.sqrt(2.0))
    affine(channel_factors[2], 1, 1.0, input_=-0.5)

    neighbor_gate = np.zeros((2, 10), np.float32)
    neighbor_gate[0, :] = 1.0
    neighbor_gate[1, 2] = 1.0

    initializers = [
        tensor("P", spatial_features),
        tensor("M", mode),
        tensor("Z", channel_features),
        tensor("J", neighbor_gate),
    ]
    for k, factor in enumerate(channel_factors):
        initializers.append(tensor(f"L{k}", factor))
    for k, q in enumerate(quadratics, 2):
        initializers.append(tensor(f"Q{k}", q))

    # Operand order matters to ORT's pairwise contraction planner.  Reduce the
    # trigger copy through the two spatial relations first; joining both input
    # copies up front creates a multi-gigabyte temporary.
    inputs = ["input", "J"]
    terms = ["njuv", "sj"]
    used = set("niorcjuvs")
    labels = iter(ch for ch in string.ascii_letters if ch not in used)

    # Product of eight common roots and one mode-dependent factor per axis.
    for output_axis, source_axis in (("r", "u"), ("c", "v")):
        for k in range(2, 10):
            a, b = next(labels), next(labels)
            inputs.extend(["P", f"Q{k}", "P"])
            terms.extend([a + output_axis, a + b, b + source_axis])
        a, b = next(labels), next(labels)
        inputs.extend(["P", "M", "P"])
        terms.extend([a + output_axis, "s" + a + b, b + source_axis])

    # Join the center cell only after the trigger copy has been reduced to a
    # small (mode,row,col) map, then apply the compact channel classifier.
    inputs.append("input")
    terms.append("nirc")
    for k in range(3):
        a, b = next(labels), next(labels)
        inputs.extend(["Z", f"L{k}", "Z"])
        terms.extend([a + "o", "s" + a + b, b + "i"])
    equation = ",".join(terms) + "->norc"

    node = helper.make_node("Einsum", inputs, ["output"], equation=equation)
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph([node], "task352_compact_exact", [x], [y], initializers)
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 12)],
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def verify(model):
    sanitized = ngu.sanitize_model(copy.deepcopy(model))
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(ROOT / "task352_profile")
    session = ort.InferenceSession(sanitized.SerializeToString(), options)

    examples = json.loads((ROOT / "data" / "task352.json").read_text())
    failures = []
    count = 0
    for section in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[section]):
            benchmark = ngu.convert_to_numpy(example)
            if benchmark is None:
                continue
            count += 1
            result = ngu.run_network(session, benchmark["input"])
            if not np.array_equal(result, benchmark["output"]):
                failures.append((section, index, int(np.count_nonzero(result != benchmark["output"]))))

    profile = session.end_profiling()
    memory, params = ngu.score_network(sanitized, profile)
    try:
        os.remove(profile)
    except OSError:
        pass
    cost = memory + params if memory is not None and params is not None else None
    points = 25.0 - math.log(cost) if cost else None
    return count, failures, memory, params, cost, points


if __name__ == "__main__":
    candidate = ROOT / "other_model_onnx" / "task352.onnx"
    model = build_model()
    onnx.save(model, candidate)
    count, failures, memory, params, cost, points = verify(model)
    print(f"examples={count} failures={len(failures)}")
    if failures:
        print("first_failures=", failures[:20])
    print(f"memory={memory} params={params} cost={cost} points={points}")
    print(f"saved={candidate}")
    if failures:
        raise SystemExit(1)
