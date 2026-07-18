"""Exact task400 renderer using moment localization and a binary adder router."""

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


def tensor(name, values, dtype=None):
    return numpy_helper.from_array(np.asarray(values, dtype=dtype), name=name)


def build_model():
    # Recover start_row=23-marker_row and start_col=23-marker_col from the
    # unique solid 5x5 color-1 marker.  The epsilon makes truncating Cast robust
    # to fp32 summation roundoff without changing the integer result.
    channel_one = np.zeros(10, np.float32)
    channel_one[1] = 1.0
    coordinate = (25.0 - np.arange(30, dtype=np.float32)) / 25.0 + 1e-3

    # Five binary feature tables for coordinates 0..29.
    bit_tables = []
    values = np.arange(30, dtype=np.int32)
    for bit in range(5):
        state = ((values >> bit) & 1).astype(np.int64)
        bit_tables.append(np.eye(2, dtype=np.float32)[state].T)

    # Full adder: x + y + carry_in = sum_bit + 2*carry_out.
    full_adder = np.zeros((2, 2, 2, 2, 2), np.float32)
    for x in range(2):
        for y in range(2):
            for sum_bit in range(2):
                for carry_in in range(2):
                    for carry_out in range(2):
                        full_adder[x, y, sum_bit, carry_in, carry_out] = float(
                            x + y + carry_in == sum_bit + 2 * carry_out
                        )

    output_mask = np.zeros(30, np.float32)
    output_mask[:5] = 1.0
    initializers = [
        tensor("C", channel_one),
        tensor("R", coordinate),
        tensor("powers", [[[1]], [[2]], [[4]], [[8]], [[16]]], np.int32),
        tensor(
            "bit_targets",
            [[[0, 1]], [[0, 2]], [[0, 4]], [[0, 8]], [[0, 16]]],
            np.int32,
        ),
        tensor("axes02", [0, 2], np.int64),
        tensor("F", full_adder),
        tensor("F00", full_adder[:, 0, :, :, 0]),
        tensor("zero", [1.0, 0.0], np.float32),
        tensor("mask", output_mask),
        tensor("row", [1.0, 0.0], np.float32),
        tensor("col", [0.0, 1.0], np.float32),
    ]
    for bit, table in enumerate(bit_tables):
        initializers.append(tensor(f"B{bit}", table))
        if bit < 3:
            initializers.append(tensor(f"T{bit}", table[:, :5]))

    nodes = [
        helper.make_node("Einsum", ["input", "C", "R"], ["sr"], equation="nchw,c,h->n"),
        helper.make_node("Einsum", ["input", "C", "R"], ["sc"], equation="nchw,c,w->n"),
        helper.make_node("Concat", ["sr", "sc"], ["sf"], axis=0),
        helper.make_node("Cast", ["sf"], ["starts"], to=TensorProto.INT32),
        helper.make_node("Unsqueeze", ["starts", "axes02"], ["startsx"]),
        helper.make_node("BitwiseAnd", ["startsx", "powers"], ["maskedx"]),
        helper.make_node("Equal", ["maskedx", "bit_targets"], ["state_bool"]),
        helper.make_node("Cast", ["state_bool"], ["states"], to=TensorProto.FLOAT),
        helper.make_node(
            "Split",
            ["states"],
            [f"s{k}x" for k in range(5)],
            axis=0,
            num_outputs=5,
        ),
    ]
    # A five-bit full-adder enforces source+output=start independently on rows
    # and columns.  Final carry zero prevents modulo-32 aliases.
    # Split retains a leading singleton dimension.  Reusing label t for that
    # size-one axis avoids five extra Squeeze tensors and remains under the
    # 52-label Einsum limit.
    reserved = set("hwabqptuv")
    labels = iter(ch for ch in string.ascii_letters if ch not in reserved)
    # Operand order is deliberately row-first, then column.  ORT's pairwise
    # planner can then contract away source row h before output column b is
    # introduced, avoiding a very large h*a*w*b temporary.
    inputs = ["input"]
    terms = ["...hw"]
    boundary_start = next(labels)

    def add_dimension(source_label, route_label, selector_label, selector_name):
        # Carry-in is zero in both dimensions.  Bit 4 also has fixed output bit
        # and carry-out zero, handled by the small F00 slice below.
        carries = [boundary_start] + [next(labels) for _ in range(4)]
        inputs.extend([selector_name, "zero"])
        terms.extend([selector_label, carries[0]])
        for bit in range(5):
            if bit < 4:
                x, y, z = next(labels), next(labels), next(labels)
                output_bit = f"T{bit}" if bit < 3 else "zero"
                inputs.extend([f"B{bit}", output_bit, f"s{bit}x", "F"])
                terms.extend(
                    [
                        x + source_label,
                        y + route_label if bit < 3 else y,
                        "t" + selector_label + z,
                        x + y + z + carries[bit] + carries[bit + 1],
                    ]
                )
            else:
                x, z = next(labels), next(labels)
                inputs.extend(["B4", "s4x", "F00"])
                terms.extend(
                    [
                        x + source_label,
                        "t" + selector_label + z,
                        x + z + carries[4],
                    ]
                )

    def expand_route(route_label, output_label):
        inputs.append("mask")
        terms.append(output_label)
        for bit in range(3):
            state = next(labels)
            inputs.extend([f"B{bit}", f"T{bit}"])
            terms.extend([state + output_label, state + route_label])

    add_dimension("h", "u", "q", "row")
    expand_route("u", "a")
    add_dimension("w", "v", "p", "col")
    expand_route("v", "b")
    equation = ",".join(terms) + "->...ab"
    nodes.append(helper.make_node("Einsum", inputs, ["output"], equation=equation))

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task400_binary_router", [x], [y], initializers)
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
    options.profile_file_prefix = str(ROOT / "task400_binary_profile")
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
    candidate = ROOT / "other_model_onnx" / "task400_binary.onnx"
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
