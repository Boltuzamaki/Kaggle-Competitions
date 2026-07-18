from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_DIR / "repairs" / "task175.onnx"
OUTPUT = PROJECT_DIR / "other_model_onnx" / "task175_compact.onnx"


def build() -> Path:
    model = onnx.load(SOURCE)
    graph = model.graph
    initializers = {item.name: item for item in graph.initializer}

    slice_node = next(node for node in graph.node if node.op_type == "Slice")
    if list(slice_node.input) != [
        "input",
        "probe_starts",
        "probe_ends",
        "probe_axes",
    ]:
        raise RuntimeError(f"Unexpected task175 Slice inputs: {list(slice_node.input)}")

    # This stride-30 1x1 convolution samples the top-left one-hot pixel and
    # collapses its channels to the actual color index (1..k).
    color_weights = numpy_helper.from_array(
        np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1),
        "top_left_color_weights",
    )
    graph.initializer.append(color_weights)
    conv_node = helper.make_node(
        "Conv",
        ["input", "top_left_color_weights"],
        ["a0_float"],
        kernel_shape=[1, 1],
        strides=[30, 30],
        name="top_left_probe",
    )
    slice_index = list(graph.node).index(slice_node)
    graph.node.remove(slice_node)
    graph.node.insert(slice_index, conv_node)
    for name in ("probe_starts", "probe_ends", "probe_axes"):
        graph.initializer.remove(initializers[name])

    phase_argmax = next(
        node for node in graph.node
        if node.op_type == "ArgMax" and "phase_probe" in node.input
    )
    argmax_index = list(graph.node).index(phase_argmax)
    graph.node.remove(phase_argmax)
    graph.node.insert(
        argmax_index,
        helper.make_node(
            "Cast", ["a0_float"], ["a0_rank4"], to=TensorProto.INT8
        ),
    )
    phase_cast = next(
        node for node in graph.node
        if node.op_type == "Cast" and "a0_idx" in node.input
    )
    cast_index = list(graph.node).index(phase_cast)
    graph.node.remove(phase_cast)
    graph.node.insert(
        cast_index,
        helper.make_node("Flatten", ["a0_rank4"], ["a0_i8"], axis=1),
    )

    # Every generated input contains at least one erased (color-zero) cell and
    # all real colors are contiguous 1..k. The presence vector therefore sums
    # to k+1, which is cheaper to reduce and adjust than an int64 ArgMax.
    k_argmax = next(
        node for node in graph.node
        if node.op_type == "ArgMax" and "color_present" in node.input
    )
    k_argmax_index = list(graph.node).index(k_argmax)
    graph.node.remove(k_argmax)
    graph.node.insert(
        k_argmax_index,
        helper.make_node(
            "ReduceSum", ["color_present"], ["k_plus_one_float"], keepdims=0
        ),
    )
    k_cast = next(
        node for node in graph.node
        if node.op_type == "Cast" and "k_max" in node.input
    )
    k_cast_index = list(graph.node).index(k_cast)
    graph.node.remove(k_cast)
    graph.node.insert(
        k_cast_index,
        helper.make_node(
            "Cast", ["k_plus_one_float"], ["k_plus_one_i8"], to=TensorProto.INT8
        ),
    )
    graph.node.insert(
        k_cast_index + 1,
        helper.make_node("Sub", ["k_plus_one_i8", "one_i8"], ["k_i8"]),
    )
    graph.initializer.append(
        numpy_helper.from_array(np.asarray(1, dtype=np.int8), "one_i8")
    )

    # The old probe omitted channel zero, so its ArgMax was color-1. Shifting
    # the tiny base table down by one preserves every subsequent value exactly.
    base = numpy_helper.to_array(initializers["base_i8"]).copy()
    base -= 1
    initializers["base_i8"].CopyFrom(
        numpy_helper.from_array(base, "base_i8")
    )

    onnx.checker.check_model(model, full_check=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
