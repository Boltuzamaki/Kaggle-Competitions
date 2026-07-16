"""Build and verify a compact exact task300 crop renderer."""

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
    base = 13.0
    # The selected object is at most 4x3, so its leading row/column contributes
    # less than log_base(3.25).  A 0.75 exponent bias makes truncation recover
    # the exact minimum coordinate directly, eliminating two subtract nodes.
    position = np.power(base, -np.arange(30, dtype=np.float32) - 0.75).astype(np.float32)
    non_background = np.ones(10, np.float32)
    non_background[0] = 0.0

    ids = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    initializers = [
        tensor("nonbg", non_background),
        tensor("depth", 10, np.int64),
        tensor("hot_values", [0.0, 1.0], np.float32),
        tensor("ids", ids),
        tensor("position", position),
        tensor("log_base", -math.log(base), np.float32),
        tensor("slice_sizes", [1, 4, 3], np.int32),
        tensor("slice_axes", [1, 2, 3], np.int32),
        tensor("axis3", [3], np.int64),
        tensor("axis2", [2], np.int64),
        tensor("outside", 10, np.uint8),
        tensor("pads", [0, 0, 26, 27], np.int64),
        tensor("pad_axes", [2, 3], np.int64),
    ]
    nodes = [
        helper.make_node("Einsum", ["input", "nonbg"], ["counts"], equation="bchw,c->c"),
        helper.make_node("ArgMax", ["counts"], ["winner"], axis=0, keepdims=1),
        helper.make_node("Cast", ["winner"], ["winner_u8"], to=TensorProto.UINT8),
        helper.make_node("OneHot", ["winner", "depth", "hot_values"], ["selector_f"], axis=-1),
        helper.make_node(
            "Einsum",
            ["input", "selector_f", "position"],
            ["row_code"],
            equation="bchw,bc,h->b",
        ),
        helper.make_node(
            "Einsum",
            ["input", "selector_f", "position"],
            ["col_code"],
            equation="bchw,bc,w->b",
        ),
        helper.make_node("Log", ["row_code"], ["row_log"]),
        helper.make_node("Log", ["col_code"], ["col_log"]),
        helper.make_node("Div", ["row_log", "log_base"], ["row_f"]),
        helper.make_node("Div", ["col_log", "log_base"], ["col_f"]),
        helper.make_node("Cast", ["row_f"], ["top"], to=TensorProto.INT32),
        helper.make_node("Cast", ["col_f"], ["left"], to=TensorProto.INT32),
        helper.make_node("Cast", ["winner"], ["winner_i32"], to=TensorProto.INT32),
        helper.make_node("Concat", ["winner_i32", "top", "left"], ["starts"], axis=0),
        helper.make_node("Add", ["starts", "slice_sizes"], ["ends"]),
        helper.make_node("Slice", ["input", "starts", "ends", "slice_axes"], ["crop"]),
        helper.make_node("Cast", ["crop"], ["object_u8"], to=TensorProto.UINT8),
        helper.make_node("ReduceMax", ["object_u8", "axis3"], ["row_u8"], keepdims=1),
        helper.make_node("ReduceMax", ["object_u8", "axis2"], ["col_u8"], keepdims=1),
        helper.make_node("Cast", ["row_u8"], ["row_mask"], to=TensorProto.BOOL),
        helper.make_node("Cast", ["col_u8"], ["col_mask"], to=TensorProto.BOOL),
        helper.make_node("And", ["row_mask", "col_mask"], ["bbox"]),
        helper.make_node("Mul", ["object_u8", "winner_u8"], ["inside"]),
        helper.make_node("Where", ["bbox", "inside", "outside"], ["classes"]),
        helper.make_node("Equal", ["classes", "ids"], ["patch"]),
        helper.make_node("Pad", ["patch", "pads", "", "pad_axes"], ["output"]),
    ]
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    crop = helper.make_tensor_value_info("crop", TensorProto.FLOAT, [1, 1, 4, 3])
    graph = helper.make_graph(
        nodes,
        "task300_compact",
        [x],
        [y],
        initializers,
        value_info=[crop],
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model, full_check=True)
    return model


def verify(model):
    sanitized = ngu.sanitize_model(copy.deepcopy(model))
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(ROOT / "task300_profile")
    session = ort.InferenceSession(sanitized.SerializeToString(), options)
    examples = json.loads((ROOT / "data" / "task300.json").read_text())
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
    candidate = ROOT / "other_model_onnx" / "task300.onnx"
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
