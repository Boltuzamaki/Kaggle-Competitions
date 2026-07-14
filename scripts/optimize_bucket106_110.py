"""Exact, no-negative-pad golf candidates for the task106--110 bucket.

The task109 rewrite factors the dense 2x10x10 color-routing tensor in the
final Einsum into two explicit color modes:

* mode 0: background channel 0
* mode 1: the input-dependent line color

The source-channel selector then chooses exactly one of those modes.  This is
an algebraic refactor of the existing graph; spatial routing is unchanged.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
SOURCE_109 = ROOT / "repairs" / "task109.onnx"
OUTPUT_109 = ROOT / "other_model_onnx" / "task109_bucket_optimized.onnx"


def optimize_task109() -> onnx.ModelProto:
    model = onnx.load(SOURCE_109)
    graph = model.graph
    final = graph.node[-1]
    if final.op_type != "Einsum" or final.output != ["output"]:
        raise RuntimeError("task109 final node is not the expected Einsum")

    init_by_name = {init.name: init for init in graph.initializer}
    old_core = init_by_name.get("output_core")
    source_modes = init_by_name.get("source_modes")
    if old_core is None or list(old_core.dims) != [2, 10, 10]:
        raise RuntimeError("task109 output_core layout changed")
    if source_modes is None or list(source_modes.dims) != [10, 2]:
        raise RuntimeError("task109 source_modes layout changed")

    # A source pixel is either background (channel 0) or foreground.  Select
    # exactly one color mode instead of adding/cancelling through a dense core.
    selector = np.zeros((10, 2), dtype=np.float32)
    selector[0, 0] = 1.0
    selector[1:, 1] = 1.0
    source_modes.CopyFrom(numpy_helper.from_array(selector, "source_modes"))

    graph.initializer.remove(old_core)
    background = np.zeros((1, 10, 1, 1), dtype=np.float32)
    background[0, 0, 0, 0] = 1.0
    graph.initializer.append(numpy_helper.from_array(background, "background_mode"))

    color_modes = helper.make_node(
        "Concat",
        ["background_mode", "line_pixel"],
        ["color_modes"],
        name="color_modes",
        axis=0,
    )
    graph.node.insert(len(graph.node) - 1, color_modes)

    inputs = list(final.input)
    if inputs[:7] != [
        "input",
        "line_pixel",
        "case_gate",
        "source_gate",
        "source_gate",
        "source_modes",
        "output_core",
    ]:
        raise RuntimeError("task109 final Einsum inputs changed")
    del inputs[1]  # line_pixel is now consumed by color_modes
    inputs[5] = "color_modes"
    del final.input[:]
    final.input.extend(inputs)

    equation_attr = next(attr for attr in final.attribute if attr.name == "equation")
    equation = equation_attr.s.decode("ascii")
    prefix = "nahw,nqxy,k,kh,kw,as,soq,kt,"
    if not equation.startswith(prefix):
        raise RuntimeError("task109 final Einsum equation changed")
    equation_attr.s = ("nahw,k,kh,kw,as,soxy,kt," + equation[len(prefix) :]).encode("ascii")

    onnx.checker.check_model(model, full_check=True)
    OUTPUT_109.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUTPUT_109)
    return model


def verify_task109_shard(shard: int, shards: int) -> None:
    sys.path.insert(0, str(ROOT / "data" / "neurogolf_utils"))
    import neurogolf_utils as ngu

    model = ngu.sanitize_model(copy.deepcopy(onnx.load(OUTPUT_109)))
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    task = json.loads((ROOT / "data" / "task109.json").read_text())
    examples = task["train"] + task["test"] + task["arc-gen"]
    failures = []
    checked = 0
    for index, example in enumerate(examples):
        if index % shards != shard:
            continue
        benchmark = ngu.convert_to_numpy(example)
        if not benchmark:
            continue
        actual = ngu.run_network(session, benchmark["input"])
        checked += 1
        if not np.array_equal(actual, benchmark["output"]):
            failures.append(index)
    print(f"shard={shard}/{shards} checked={checked} failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-task109-shard", type=int)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.verify_task109_shard is None:
        optimize_task109()
        print(OUTPUT_109)
    else:
        verify_task109_shard(args.verify_task109_shard, args.shards)
