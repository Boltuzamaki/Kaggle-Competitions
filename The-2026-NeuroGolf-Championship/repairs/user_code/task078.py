# Source: predicted/test_onnx_task078.py — ONNX graph construction code
# Verified model: repairs/task078.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

def create_model():
    I64 = TensorProto.INT64
    x = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    y = helper.make_tensor_value_info('output', I64, ['batch', 'H', 'W'])
    
    def K(name, arr, dtype=np.int64): 
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)
        
    inits = [
        K('c1', [1]),
        K('c2', [2]),
        K('c0', [0]),
        K('axes_1', [1]),
        K('axes_0', [0]),
        K('starts_1', [1]), K('ends_2', [2]),
        K('shape_H1', [-1, 1]),
        K('shape_1HW', [1, -1, -1]),
    ]
    
    nodes = []
    
    nodes.append(helper.make_node('Equal', ['input', 'c1'], ['is_1_bool']))
    nodes.append(helper.make_node('Cast', ['is_1_bool'], ['is_1'], to=I64))
    
    nodes.append(helper.make_node('Equal', ['input', 'c2'], ['is_2_bool']))
    nodes.append(helper.make_node('Cast', ['is_2_bool'], ['is_2'], to=I64))
    
    nodes.append(helper.make_node('ReduceSum', ['is_1', 'axes_1'], ['N1'], keepdims=0))
    nodes.append(helper.make_node('ReduceSum', ['is_2', 'axes_1'], ['N2'], keepdims=0))
    
    nodes.append(helper.make_node('Shape', ['input'], ['shape']))
    nodes.append(helper.make_node('Slice', ['shape', 'starts_1', 'ends_2', 'axes_0'], ['H_tensor']))
    nodes.append(helper.make_node('Squeeze', ['H_tensor', 'axes_0'], ['H_scalar']))
    
    nodes.append(helper.make_node('Range', ['c0', 'H_scalar', 'c1'], ['R_1d']))
    nodes.append(helper.make_node('Reshape', ['R_1d', 'shape_H1'], ['R']))
    
    nodes.append(helper.make_node('Less', ['R', 'N1'], ['is_out_1_bool']))
    nodes.append(helper.make_node('Cast', ['is_out_1_bool'], ['out_1'], to=I64))
    
    nodes.append(helper.make_node('Add', ['N1', 'N2'], ['N1_plus_N2']))
    nodes.append(helper.make_node('Less', ['R', 'N1_plus_N2'], ['is_out_12_bool']))
    
    nodes.append(helper.make_node('Not', ['is_out_1_bool'], ['not_out_1_bool']))
    nodes.append(helper.make_node('And', ['is_out_12_bool', 'not_out_1_bool'], ['is_out_2_bool']))
    
    nodes.append(helper.make_node('Cast', ['is_out_2_bool'], ['out_2_raw'], to=I64))
    nodes.append(helper.make_node('Mul', ['out_2_raw', 'c2'], ['out_2']))
    
    nodes.append(helper.make_node('Add', ['out_1', 'out_2'], ['out_2d']))
    
    nodes.append(helper.make_node('Shape', ['out_2d'], ['out_2d_shape']))
    nodes.append(helper.make_node('Concat', ['c1', 'out_2d_shape'], ['final_shape'], axis=0))
    nodes.append(helper.make_node('Reshape', ['out_2d', 'final_shape'], ['output']))
    
    graph = helper.make_graph(nodes, 'task078', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])

def check_task():
    with open('task078.json', 'r') as f:
        task = json.load(f)
        
    model = create_model()
    onnx.save(model, 'task078.onnx')
    session = onnxruntime.InferenceSession('task078.onnx')
    
    for split in ['train', 'test']:
        for i, ex in enumerate(task[split]):
            inp = np.array([ex['input']], dtype=np.int64)
            out = np.array([ex['output']], dtype=np.int64)
            
            res = session.run(['output'], {'input': inp})[0]
            
            if np.array_equal(res, out):
                print(f"{split} {i}: ONNX MATCH")
            else:
                print(f"{split} {i}: ONNX FAIL")
                print('Expected:', out)
                print('Got:', res)


# Build the model
model = create_model()
