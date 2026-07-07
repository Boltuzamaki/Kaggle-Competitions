# Source: predicted/test_onnx_task079.py — ONNX graph construction code
# Verified model: repairs/task079.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

def create_model():
    I64 = TensorProto.INT64
    F = TensorProto.FLOAT
    x = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    y = helper.make_tensor_value_info('output', I64, ['batch', 3, 3])
    
    def K(name, arr, dtype=np.int64): 
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)
        
    inits = [
        K('c1_int', np.array(1, dtype=np.int64)), K('c2_int', np.array(2, dtype=np.int64)), K('c3_int', np.array(3, dtype=np.int64)),
        K('c1_float', np.array(1.0, dtype=np.float32), dtype=np.float32),
        K('axes_0', [0]), K('axes_1', [1]), K('axes_1_2', [1, 2]), K('axes_2_3', [2, 3]),
        K('shape_B11', [-1, 1, 1]),
        K('W_conv9', np.ones((9, 1, 3, 3), dtype=np.float32), dtype=np.float32),
        K('W_conv1', np.ones((1, 1, 3, 3), dtype=np.float32), dtype=np.float32),
        K('starts_2', [2]), K('ends_3', [3]),
    ]
    
    nodes = []
    
    is_c_list = []
    for c_val in range(1, 10):
        inits.append(K(f'color_{c_val}', [c_val]))
        nodes.append(helper.make_node('Equal', ['input', f'color_{c_val}'], [f'is_c_{c_val}_bool']))
        nodes.append(helper.make_node('Cast', [f'is_c_{c_val}_bool'], [f'is_c_{c_val}_f'], to=F))
        nodes.append(helper.make_node('Unsqueeze', [f'is_c_{c_val}_f', 'axes_1'], [f'is_c_{c_val}_4d']))
        is_c_list.append(f'is_c_{c_val}_4d')
        
    nodes.append(helper.make_node('Concat', is_c_list, ['is_all_c'], axis=1))
    
    nodes.append(helper.make_node('Conv', ['is_all_c', 'W_conv9'], ['window_counts'], group=9))
    
    nodes.append(helper.make_node('ReduceMax', ['window_counts'], ['max_window'], axes=[2, 3], keepdims=0))
    nodes.append(helper.make_node('ReduceSum', ['is_all_c', 'axes_2_3'], ['total_pixels'], keepdims=0))
    
    nodes.append(helper.make_node('Max', ['max_window', 'c1_float'], ['max_window_safe']))
    nodes.append(helper.make_node('Div', ['total_pixels', 'max_window_safe'], ['num_shapes']))
    
    nodes.append(helper.make_node('ArgMax', ['num_shapes'], ['best_c_idx'], axis=1, keepdims=0))
    nodes.append(helper.make_node('Add', ['best_c_idx', 'c1_int'], ['best_c']))
    
    nodes.append(helper.make_node('Reshape', ['best_c', 'shape_B11'], ['best_c_reshaped']))
    nodes.append(helper.make_node('Equal', ['input', 'best_c_reshaped'], ['is_C_bool']))
    nodes.append(helper.make_node('Cast', ['is_C_bool'], ['is_C_float'], to=F))
    nodes.append(helper.make_node('Unsqueeze', ['is_C_float', 'axes_1'], ['is_C_4d']))
    
    nodes.append(helper.make_node('Conv', ['is_C_4d', 'W_conv1'], ['window_counts_best']))
    nodes.append(helper.make_node('Flatten', ['window_counts_best'], ['window_counts_flat'], axis=1))
    nodes.append(helper.make_node('ArgMax', ['window_counts_flat'], ['best_window_idx'], axis=1, keepdims=0))
    
    nodes.append(helper.make_node('Shape', ['input'], ['shape']))
    nodes.append(helper.make_node('Slice', ['shape', 'starts_2', 'ends_3', 'axes_0'], ['W_tensor']))
    nodes.append(helper.make_node('Sub', ['W_tensor', 'c2_int'], ['W_out']))
    
    nodes.append(helper.make_node('Div', ['best_window_idx', 'W_out'], ['r']))
    nodes.append(helper.make_node('Mod', ['best_window_idx', 'W_out'], ['c']))
    
    nodes.append(helper.make_node('Add', ['r', 'c3_int'], ['r_end']))
    nodes.append(helper.make_node('Add', ['c', 'c3_int'], ['c_end']))
    
    nodes.append(helper.make_node('Concat', ['r', 'c'], ['starts'], axis=0))
    nodes.append(helper.make_node('Concat', ['r_end', 'c_end'], ['ends'], axis=0))
    
    nodes.append(helper.make_node('Slice', ['input', 'starts', 'ends', 'axes_1_2'], ['output']))
    
    graph = helper.make_graph(nodes, 'task079', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])

def check_task():
    with open('task079.json', 'r') as f:
        task = json.load(f)
        
    model = create_model()
    onnx.save(model, 'task079.onnx')
    session = onnxruntime.InferenceSession('task079.onnx')
    
    for split in ['train', 'test']:
        for i, ex in enumerate(task[split]):
            inp = np.array([ex['input']], dtype=np.int64)
            out = np.array([ex['output']], dtype=np.int64)
            
            res = session.run(['output'], {'input': inp})[0]
            
            if np.array_equal(res, out):
                print(f"{split} {i}: ONNX MATCH")
            else:
                print(f"{split} {i}: ONNX FAIL")
                print('Expected:\n', out)
                print('Got:\n', res)


# Build the model
model = create_model()
