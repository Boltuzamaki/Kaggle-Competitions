# Source: predicted/test_onnx_task237.py — ONNX graph construction code
# Verified model: repairs/task237.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    I64 = TensorProto.INT64
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H_out', 'W_out'])
    nodes = []
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    make_const('c0_11', [[0]])
    make_const('axes_0', [0])
    nodes.append(helper.make_node('Shape', ['input'], ['in_shape']))
    make_const('c0_1d', [0])
    make_const('c1_1d', [1])
    nodes.append(helper.make_node('Slice', ['in_shape', 'c0_1d', 'c1_1d', 'axes_0'], ['batch_dim']))
    nodes.append(helper.make_node('Expand', ['c0_11', 'batch_dim'], ['output_2d']))
    make_const('shape_b11', [-1, 1, 1])
    nodes.append(helper.make_node('Reshape', ['output_2d', 'shape_b11'], ['output']))
    graph = helper.make_graph(nodes, 'task237_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task237_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task237.onnx')


# Build the model (the function saves it internally, so we load the result)
create_onnx_model()
import glob
model = onnx.load("/project/repairs/task237.onnx")
