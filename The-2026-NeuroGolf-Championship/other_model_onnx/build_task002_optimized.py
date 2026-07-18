"""Build the direction-pruned exact bitset flood fill for task002.

The repair encodes each 20-cell row in one uint32 and propagates exterior
reachability through a fixed sequence of masked shifts.  Exhaustive search of
all direction subsets that can still reach the baseline fixed point showed
that six stages each contain one redundant direction across the full corpus.
"""

from pathlib import Path
import copy
import math

import onnx
from onnx import helper


ROOT = Path.cwd()
SOURCE = ROOT / "repairs" / "task002.onnx"
DESTINATION = ROOT / "other_model_onnx" / "task002.onnx"


# stage: (incoming state, masked stage result, retained direction tensors)
SCHEDULE = {
    "85": ("safe_name_78", "safe_name_85", ["safe_name_79", "safe_name_80"]),
    "103": (
        "safe_name_94",
        "safe_name_103",
        ["safe_name_95", "safe_name_97", "safe_name_98"],
    ),
    "121": (
        "safe_name_112",
        "safe_name_121",
        ["safe_name_114", "safe_name_115", "safe_name_116"],
    ),
    "139": (
        "safe_name_130",
        "safe_name_139",
        ["safe_name_131", "safe_name_132", "safe_name_133"],
    ),
    "148": (
        "safe_name_139",
        "safe_name_148",
        ["safe_name_140", "safe_name_141", "safe_name_143"],
    ),
    "157": (
        "safe_name_148",
        "safe_name_157",
        ["safe_name_149", "safe_name_151", "safe_name_152"],
    ),
}


def build() -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load(SOURCE))
    nodes = list(model.graph.node)
    insert_before = {}

    for stage, (state, result, directions) in SCHEDULE.items():
        result_index = next(
            index for index, node in enumerate(nodes) if result in node.output
        )
        previous = state
        replacements = []
        for index, direction in enumerate(directions):
            output = f"optimized_stage_{stage}_{index}"
            replacements.append(
                helper.make_node("BitwiseOr", [previous, direction], [output])
            )
            previous = output
        nodes[result_index].input[0] = previous
        insert_before[result_index] = replacements

    rewritten = []
    for index, node in enumerate(nodes):
        rewritten.extend(insert_before.get(index, []))
        rewritten.append(node)

    # Remove every now-dead direction and OR tensor by tracing back from output.
    required = {output.name for output in model.graph.output}
    kept_reversed = []
    for node in reversed(rewritten):
        if any(output and output in required for output in node.output):
            kept_reversed.append(node)
            required.update(name for name in node.input if name)
    kept_nodes = list(reversed(kept_reversed))
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)

    referenced = {name for node in kept_nodes for name in node.input}
    kept_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in referenced
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

    produced = {name for node in kept_nodes for name in node.output if name}
    kept_value_info = [
        item for item in model.graph.value_info if item.name in produced
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)

    onnx.checker.check_model(model, full_check=True)
    return model


model = build()


if __name__ == "__main__":
    onnx.save(model, DESTINATION)
    parameters = sum(math.prod(item.dims) for item in model.graph.initializer)
    print(
        f"{DESTINATION} nodes={len(model.graph.node)} "
        f"initializers={len(model.graph.initializer)} params={parameters}"
    )
