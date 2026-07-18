"""Build and exhaustively verify a compact exact task162 detector."""

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
    # The signed 4x4 filter detects an all-background 3x3 window while
    # suppressing a lower-right window that overlaps an already selected one.
    # Its integer scores separate at 73.  Dividing the kernel by 73 lets a
    # direct saturating uint8 Cast replace QuantizeLinear and its scale.
    weights = np.zeros((10, 1, 4, 4), np.float32)
    weights[0, 0] = np.asarray(
        [[9, 9, 9, 0], [9, 8, 8, -1], [9, 8, 8, -1], [0, -1, -1, -1]],
        dtype=np.float32,
    ) / 73.0
    blue = np.zeros((1, 10, 1, 1), np.float32)
    blue[0, 1, 0, 0] = 1.0
    initializers = [
        tensor("weights", weights),
        tensor("blue", blue),
        tensor("pads", [0, 0, 10, 10], np.int64),
        tensor("pad_axes", [2, 3], np.int64),
    ]
    nodes = [
        helper.make_node(
            "ConvTranspose", ["input", "weights"], ["score"],
            pads=[2, 2, 13, 13],
        ),
        helper.make_node("Cast", ["score"], ["hit"], to=TensorProto.UINT8),
        helper.make_node(
            "MaxPool", ["hit"], ["spread"],
            kernel_shape=[3, 3], pads=[2, 2, 2, 2], strides=[1, 1],
        ),
        helper.make_node("Cast", ["spread"], ["mask20"], to=TensorProto.BOOL),
        helper.make_node("Pad", ["mask20", "pads", "", "pad_axes"], ["mask30"]),
        helper.make_node("Where", ["mask30", "blue", "input"], ["output"]),
    ]
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task162_compact", [x], [y], initializers)
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model, full_check=True)
    return model


def verify(model):
    sanitized = ngu.sanitize_model(copy.deepcopy(model))
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(ROOT / "task162_profile")
    session = ort.InferenceSession(sanitized.SerializeToString(), options)
    examples = json.loads((ROOT / "data" / "task162.json").read_text())
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
    candidate = ROOT / "other_model_onnx" / "task162.onnx"
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
