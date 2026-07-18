"""Build and exhaustively verify a compact exact task126 renderer."""

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
    return numpy_helper.from_array(np.asarray(values, dtype=dtype), name=name)


def build_model():
    background = np.zeros(10, np.float32)
    background[0] = 1.0
    initializers = [
        tensor("background", background),
        tensor("one", 1.0, np.float32),
        tensor("deltas", [[-1.0], [1.0]], np.float32),
        tensor("zero", 0.0, np.float32),
        tensor("prefix", [[0, 0], [0, 4]], np.int64),
    ]
    nodes = [
        # A center column has exactly height-1 background cells: only the top
        # cell of its inverted-U glyph is non-background.
        helper.make_node("Einsum", ["input", "background"], ["column_sums"], equation="nchw,c->nw"),
        helper.make_node("ReduceMax", ["column_sums"], ["height"], keepdims=1),
        helper.make_node("Sub", ["height", "one"], ["bottom_f"]),
        helper.make_node("Equal", ["column_sums", "bottom_f"], ["centers"]),
        # At the selected bottom-row cells, subtract the existing background
        # bit and add the yellow bit.  All other columns receive additive zero.
        helper.make_node("Where", ["centers", "deltas", "zero"], ["updates"]),
        helper.make_node("Cast", ["bottom_f"], ["bottom_i"], to=TensorProto.INT64),
        helper.make_node("Concat", ["bottom_i", "bottom_i"], ["bottom_i2"], axis=0),
        helper.make_node("Concat", ["prefix", "bottom_i2"], ["indices"], axis=1),
        helper.make_node("ScatterND", ["input", "indices", "updates"], ["output"], reduction="add"),
    ]
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task126_compact", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model, full_check=True)
    return model


def verify(model):
    sanitized = ngu.sanitize_model(copy.deepcopy(model))
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(ROOT / "task126_profile")
    session = ort.InferenceSession(sanitized.SerializeToString(), options)
    examples = json.loads((ROOT / "data" / "task126.json").read_text())
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
    candidate = ROOT / "other_model_onnx" / "task126.onnx"
    model = build_model()
    onnx.save(model, candidate)
    result = verify(model)
    print("examples=%d failures=%d" % (result[0], len(result[1])))
    if result[1]:
        print("first_failures=", result[1][:20])
    print("memory=%s params=%s cost=%s points=%s" % result[2:])
    print(f"saved={candidate}")
    if result[1]:
        raise SystemExit(1)
