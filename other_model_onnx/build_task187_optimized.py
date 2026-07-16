"""Build an exact lower-cost decoder for the repaired task187 bitset model."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repairs" / "task187.onnx"
OUTPUT = ROOT / "other_model_onnx" / "task187.onnx"

source = onnx.load(SOURCE)
graph = source.graph

# Keep the exact packed flood fill through its 25-bit enclosed-background word.
nodes = [deepcopy(node) for node in graph.node[:72]]

# Fold the fixed 30->25 row crop into the packing Conv. A two-row kernel with
# dilation 5 has effective height 6; its zero second row preserves the original
# calculation while shrinking every packing tensor and eliminating the Slice.
nodes[0].attribute.append(helper.make_attribute("dilations", [5, 1]))
del nodes[6]
for existing_node in nodes:
    for index, name in enumerate(existing_node.input):
        if name == "safe_name_36":
            existing_node.input[index] = "safe_name_35"


def add_node(op, inputs, output, **attrs):
    nodes.append(helper.make_node(
        op, inputs, [output], name=f"opt_{len(nodes):03d}_{op.lower()}", **attrs
    ))
    return output


# Decode the low three bytes independently. This is cheaper than broadcasting
# four shifts, casting the 25x4 tensor, and slicing it back into four bytes.
byte0 = add_node("Cast", ["safe_name_101"], "dec_byte0", to=TensorProto.UINT8)
shift8 = add_node("BitShift", ["safe_name_101", "safe_name_5"], "dec_shift8", direction="RIGHT")
byte1 = add_node("Cast", [shift8], "dec_byte1", to=TensorProto.UINT8)
shift16 = add_node("BitShift", [shift8, "safe_name_5"], "dec_shift16", direction="RIGHT")
byte2 = add_node("Cast", [shift16], "dec_byte2", to=TensorProto.UINT8)

decoded = []
for index, byte in enumerate((byte0, byte1, byte2)):
    masked = add_node("BitwiseAnd", [byte, "safe_name_19"], f"dec_masked{index}")
    decoded.append(add_node("Greater", [masked, "safe_name_21"], f"dec_bits{index}"))

# Only column 24 remains. The packed word has that bit set exactly when its
# unsigned value exceeds 2^24-1, so no fourth Shift/Cast/Slice is necessary.
decoded.append(add_node("Greater", ["safe_name_101", "dec_low24"], "dec_bits3"))
add_node("Concat", decoded, "safe_name_116", axis=3)

# Retain the scalar-color Conv, but select color 2 directly with the boolean
# enclosure mask. This replaces both the bool->uint8 Cast and uint8 Sub.
nodes.extend(deepcopy(node) for node in graph.node[87:89])
add_node("Where", ["safe_name_116", "dec_color2", "safe_name_119"], "safe_name_120")
nodes.extend(deepcopy(node) for node in graph.node[90:])

required = {name for node in nodes for name in node.input if name}
initializers = []
for item in graph.initializer:
    if item.name not in required:
        continue
    if item.name == "safe_name_0":
        old_weight = onnx.numpy_helper.to_array(item)
        packed_weight = np.zeros((1, 10, 2, 15), dtype=np.float32)
        packed_weight[:, :, 0, :] = old_weight[:, :, 0, :]
        initializers.append(numpy_helper.from_array(packed_weight, name=item.name))
    else:
        initializers.append(deepcopy(item))
initializers.extend([
    numpy_helper.from_array(np.asarray((1 << 24) - 1, dtype=np.uint32), name="dec_low24"),
    numpy_helper.from_array(np.asarray(2, dtype=np.uint8), name="dec_color2"),
])

produced = {out for node in nodes for out in node.output if out and out != "output"}
value_info = [deepcopy(vi) for vi in graph.value_info if vi.name in produced]

# Override shapes changed by the folded packing crop.
folded_names = {"safe_name_30", "safe_name_31", "safe_name_32", "safe_name_33", "safe_name_34", "safe_name_35"}
value_info = [item for item in value_info if item.name not in folded_names]


def vi(name, dtype, shape):
    value_info.append(helper.make_tensor_value_info(name, dtype, shape))


vi("safe_name_30", TensorProto.FLOAT, [1, 1, 25, 2])
vi("safe_name_31", TensorProto.UINT32, [1, 1, 25, 2])
for name in ("safe_name_32", "safe_name_33", "safe_name_34", "safe_name_35"):
    vi(name, TensorProto.UINT32, [1, 1, 25, 1])
for name in ("dec_byte0", "dec_byte1", "dec_byte2"):
    vi(name, TensorProto.UINT8, [1, 1, 25, 1])
for name in ("dec_shift8", "dec_shift16"):
    vi(name, TensorProto.UINT32, [1, 1, 25, 1])
for index in range(3):
    vi(f"dec_masked{index}", TensorProto.UINT8, [1, 1, 25, 8])
    vi(f"dec_bits{index}", TensorProto.BOOL, [1, 1, 25, 8])
vi("dec_bits3", TensorProto.BOOL, [1, 1, 25, 1])

new_graph = helper.make_graph(
    nodes,
    "task187_compact_decoder",
    [deepcopy(graph.input[0])],
    [deepcopy(graph.output[0])],
    initializers,
    value_info=value_info,
)
model = helper.make_model(new_graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.checker.check_model(model, full_check=True)
onnx.save(model, OUTPUT)
print(OUTPUT)
