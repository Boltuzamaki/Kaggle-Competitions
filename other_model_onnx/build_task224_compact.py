"""Build and exhaustively verify a compact exact renderer for task224."""

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
    # Color 5 is the four-point guide.  All other nonzero colors identify the
    # paint channel.  The extrema are unique and the other three markers are
    # at least one cell away, so an exponent bias of 0.2 keeps Log's result in
    # [coordinate, coordinate + 1) and Cast recovers it without a division.
    paint_guide = np.ones(10, np.float32)
    paint_guide[0] = 0.0
    paint_guide[5] = -1.0
    marker = np.zeros(10, np.float32)
    marker[5] = 1.0
    i = np.arange(30, dtype=np.float32)
    max_minus_two_weight = np.exp(i - 1.8).astype(np.float32)
    reverse_weight = np.exp(16.2 - i).astype(np.float32)

    initializers = [
        tensor("paint_guide", paint_guide.reshape(10, 1, 1)),
        tensor("marker", marker),
        tensor("max_m2_weight", max_minus_two_weight),
        tensor("reverse_weight", reverse_weight),
        tensor("u0", 0, np.uint8),
        tensor("u1", 1, np.uint8),
        tensor("u2", 2, np.uint8),
        tensor("u16", 16, np.uint8),
        tensor("idx_r", np.arange(14, dtype=np.uint8).reshape(14, 1)),
        tensor("idx_c", np.arange(14, dtype=np.uint8)),
        tensor("pad_span", [1, 15], np.int64),
        tensor("axis0", [0], np.int64),
    ]
    nodes = [
        helper.make_node("Einsum", ["input", "paint_guide"], ["paint"], equation="nchw,cab->cab"),
        helper.make_node("Einsum", ["input", "marker", "max_m2_weight"], ["rmax_code"], equation="nchw,c,h->n"),
        helper.make_node("Einsum", ["input", "marker", "max_m2_weight"], ["cmax_code"], equation="nchw,c,w->n"),
        helper.make_node("Einsum", ["input", "marker", "reverse_weight"], ["rrev_code"], equation="nchw,c,h->n"),
        helper.make_node("Einsum", ["input", "marker", "reverse_weight"], ["crev_code"], equation="nchw,c,w->n"),
        helper.make_node("Log", ["rmax_code"], ["rmax_log"]),
        helper.make_node("Log", ["cmax_code"], ["cmax_log"]),
        helper.make_node("Log", ["rrev_code"], ["rrev_log"]),
        helper.make_node("Log", ["crev_code"], ["crev_log"]),
        helper.make_node("Cast", ["rmax_log"], ["rmax_m2"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["cmax_log"], ["cmax_m2"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["rrev_log"], ["rrev"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["crev_log"], ["crev"], to=TensorProto.UINT8),
        helper.make_node("Sub", ["u16", "rrev"], ["rmin"]),
        helper.make_node("Sub", ["u16", "crev"], ["cmin"]),
        helper.make_node("Sub", ["rmax_m2", "rmin"], ["rspan"]),
        helper.make_node("Sub", ["cmax_m2", "cmin"], ["cspan"]),
        helper.make_node("Sub", ["idx_r", "rmin"], ["rdist"]),
        helper.make_node("Sub", ["idx_c", "cmin"], ["cdist"]),
        helper.make_node("LessOrEqual", ["rdist", "rspan"], ["rrange"]),
        helper.make_node("LessOrEqual", ["cdist", "cspan"], ["crange"]),
        helper.make_node("Mod", ["rdist", "rspan"], ["rmod"], fmod=0),
        helper.make_node("Mod", ["cdist", "cspan"], ["cmod"], fmod=0),
        helper.make_node("Equal", ["rmod", "u0"], ["hrow"]),
        helper.make_node("Equal", ["cmod", "u0"], ["vcol"]),
        helper.make_node("Where", ["rrange", "u1", "u0"], ["row_base"]),
        helper.make_node("Where", ["hrow", "u2", "row_base"], ["row_code"]),
        helper.make_node("Where", ["crange", "u1", "u2"], ["col_base"]),
        helper.make_node("Where", ["vcol", "u0", "col_base"], ["col_code"]),
        helper.make_node("Pad", ["row_code", "pad_span", "u0", "axis0"], ["row_code_p"]),
        helper.make_node("Pad", ["col_code", "pad_span", "u2"], ["col_code_p"]),
        helper.make_node("Greater", ["row_code_p", "col_code_p"], ["border"]),
        helper.make_node("Where", ["border", "paint", "input"], ["output"]),
    ]
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task224_compact", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model, full_check=True)
    return model


def verify(model):
    sanitized = ngu.sanitize_model(copy.deepcopy(model))
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(ROOT / "task224_profile")
    session = ort.InferenceSession(sanitized.SerializeToString(), options)
    examples = json.loads((ROOT / "data" / "task224.json").read_text())
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
    candidate = ROOT / "other_model_onnx" / "task224.onnx"
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
