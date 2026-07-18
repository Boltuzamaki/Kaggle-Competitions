import onnx
from onnx import helper, TensorProto
import numpy as np

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, 10, 10])
Y = helper.make_tensor_value_info('conv_out', TensorProto.FLOAT, [1, 1, 19, 19])

nodes = []
initializers = []

def make_init(name, data_type, dims, vals):
    initializers.append(helper.make_tensor(name, data_type, dims, vals))

make_init("starts_1", TensorProto.INT64, [1], [1])
make_init("ends_2", TensorProto.INT64, [1], [2])
make_init("axes_1", TensorProto.INT64, [1], [1])
nodes.append(helper.make_node("Slice", ["input", "starts_1", "ends_2", "axes_1"], ["c1"]))

make_init("starts_0", TensorProto.INT64, [1], [0])
make_init("ends_1", TensorProto.INT64, [1], [1])
nodes.append(helper.make_node("Slice", ["input", "starts_0", "ends_1", "axes_1"], ["c0"]))

make_init("pads", TensorProto.INT64, [8], [0, 0, 9, 9, 0, 0, 9, 9])
make_init("pad_val", TensorProto.FLOAT, [], [0.0])
nodes.append(helper.make_node("Pad", ["c0", "pads", "pad_val"], ["padded_c0"]))

nodes.append(helper.make_node("Conv", ["padded_c0", "c1"], ["conv_out"]))

graph = helper.make_graph(nodes, 'test', [X], [Y], initializer=initializers)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
onnx.checker.check_model(model)
model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
import sys
sys.path.insert(0, 'data/neurogolf_utils')
import neurogolf_utils as ngu
try:
    print('mem:', ngu.calculate_memory(model, 'dummy.json'))
except Exception as e:
    import traceback
    traceback.print_exc()
