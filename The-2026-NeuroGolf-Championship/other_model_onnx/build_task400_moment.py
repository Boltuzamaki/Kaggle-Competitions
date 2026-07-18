"""Build and fully verify a compact, negative-pad-free task400 model."""

import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "neurogolf_utils"))
import neurogolf_utils as ngu  # noqa: E402


def tensor(name, values, dtype=None):
    array = np.asarray(values, dtype=dtype)
    return numpy_helper.from_array(array, name=name)


def build_model():
    # Color 1 is a unique solid 5x5 marker.  Summing (25-coordinate)/25 over
    # its 25 pixels gives 23-marker_top_left directly.  A small positive bias
    # keeps the subsequent float-to-int cast safely above the exact integer.
    channel_one = np.zeros(10, np.float32)
    channel_one[1] = 1.0
    coordinate = (25.0 - np.arange(30, dtype=np.float32)) / 25.0 + 1e-3

    initializers = [
        tensor("C", channel_one),
        tensor("R", coordinate),
        tensor("off35", 35, np.int32),
        tensor("slice_axes", [2, 3], np.int32),
        tensor("pad_axes", [2, 3], np.int64),
        tensor("steps", [-1, -1], np.int32),
        tensor("pads", [0, 0, 25, 25], np.int64),
    ]
    nodes = [
        helper.make_node("Einsum", ["input", "C", "R"], ["sr"], equation="nchw,c,h->n"),
        helper.make_node("Einsum", ["input", "C", "R"], ["sc"], equation="nchw,c,w->n"),
        helper.make_node("Concat", ["sr", "sc"], ["sf"], axis=0),
        helper.make_node("Cast", ["sf"], ["starts"], to=TensorProto.INT32),
        helper.make_node("Sub", ["starts", "off35"], ["ends"]),
        helper.make_node("Slice", ["input", "starts", "ends", "slice_axes", "steps"], ["patch"]),
        helper.make_node("Pad", ["patch", "pads", "", "pad_axes"], ["output"]),
    ]
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    patch = helper.make_tensor_value_info("patch", TensorProto.FLOAT, [1, 10, 5, 5])
    graph = helper.make_graph(
        nodes,
        "task400_moment_locator",
        [x],
        [y],
        initializers,
        value_info=[patch],
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def verify(model):
    sanitized = ngu.sanitize_model(copy.deepcopy(model))
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(ROOT / "task400_profile")
    session = ort.InferenceSession(sanitized.SerializeToString(), options)

    examples = json.loads((ROOT / "data" / "task400.json").read_text())
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
    candidate = ROOT / "other_model_onnx" / "task400.onnx"
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
