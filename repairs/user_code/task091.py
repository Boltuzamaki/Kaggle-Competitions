# Source: predicted/test_onnx_task091.py — ONNX graph construction code
# Verified model: repairs/task091.onnx
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
    
    make_const('c0_1d', [0])
    make_const('c1_1d', [1])
    make_const('c2_1d', [2])
    make_const('c5', [5])
    make_const('c9999', [9999])
    make_const('cm1', [-1])
    make_const('axes_12', [1, 2])
    make_const('axes_0', [0])
    
    nodes.append(helper.make_node('Shape', ['input'], ['shape']))
    nodes.append(helper.make_node('Slice', ['shape', 'c1_1d', 'c2_1d', 'axes_0'], ['H_1d']))
    make_const('c3_1d', [3])
    nodes.append(helper.make_node('Slice', ['shape', 'c2_1d', 'c3_1d', 'axes_0'], ['W_1d']))
    
    nodes.append(helper.make_node('Squeeze', ['H_1d', 'axes_0'], ['H_s']))
    nodes.append(helper.make_node('Squeeze', ['W_1d', 'axes_0'], ['W_s']))
    
    nodes.append(helper.make_node('Concat', ['H_1d', 'c1_1d'], ['shape_H1'], axis=0))
    nodes.append(helper.make_node('Concat', ['c1_1d', 'W_1d'], ['shape_1W'], axis=0))
    
    nodes.append(helper.make_node('Range', ['c0_1d', 'H_s', 'c1_1d'], ['Y_1d']))
    nodes.append(helper.make_node('Reshape', ['Y_1d', 'shape_H1'], ['Y_2d']))
    nodes.append(helper.make_node('Range', ['c0_1d', 'W_s', 'c1_1d'], ['X_1d']))
    nodes.append(helper.make_node('Reshape', ['X_1d', 'shape_1W'], ['X_2d']))
    
    nodes.append(helper.make_node('Equal', ['input', 'c5'], ['mask_bool']))
    
    nodes.append(helper.make_node('Where', ['mask_bool', 'Y_2d', 'c9999'], ['Y_masked']))
    nodes.append(helper.make_node('ReduceMin', ['Y_masked'], ['min_y'], axes=[1, 2], keepdims=0))
    nodes.append(helper.make_node('Where', ['mask_bool', 'Y_2d', 'cm1'], ['Y_masked_max']))
    nodes.append(helper.make_node('ReduceMax', ['Y_masked_max'], ['max_y'], axes=[1, 2], keepdims=0))
    
    nodes.append(helper.make_node('Where', ['mask_bool', 'X_2d', 'c9999'], ['X_masked']))
    nodes.append(helper.make_node('ReduceMin', ['X_masked'], ['min_x'], axes=[1, 2], keepdims=0))
    nodes.append(helper.make_node('Where', ['mask_bool', 'X_2d', 'cm1'], ['X_masked_max']))
    nodes.append(helper.make_node('ReduceMax', ['X_masked_max'], ['max_x'], axes=[1, 2], keepdims=0))
    
    nodes.append(helper.make_node('Reshape', ['min_y', 'c1_1d'], ['min_y_1d']))
    nodes.append(helper.make_node('Reshape', ['max_y', 'c1_1d'], ['max_y_1d']))
    nodes.append(helper.make_node('Reshape', ['min_x', 'c1_1d'], ['min_x_1d']))
    nodes.append(helper.make_node('Reshape', ['max_x', 'c1_1d'], ['max_x_1d']))
    
    nodes.append(helper.make_node('Sub', ['min_y_1d', 'c1_1d'], ['start_y_tmp']))
    nodes.append(helper.make_node('Max', ['start_y_tmp', 'c0_1d'], ['start_y']))
    
    nodes.append(helper.make_node('Add', ['max_y_1d', 'c2_1d'], ['end_y']))
    
    nodes.append(helper.make_node('Identity', ['min_x_1d'], ['start_x']))
    
    nodes.append(helper.make_node('Add', ['max_x_1d', 'c1_1d'], ['end_x']))
    
    nodes.append(helper.make_node('Concat', ['start_y', 'start_x'], ['starts'], axis=0))
    nodes.append(helper.make_node('Concat', ['end_y', 'end_x'], ['ends'], axis=0))
    
    nodes.append(helper.make_node('Slice', ['input', 'starts', 'ends', 'axes_12'], ['output']))
    
    graph = helper.make_graph(nodes, 'task091_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task091_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task091.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task091.onnx')
    
    with open('task091.json', 'r') as f:
        task = json.load(f)
        
    for split in ['train', 'test']:
        for i, ex in enumerate(task[split]):
            inp = np.array(ex['input'], dtype=np.int64)[np.newaxis, ...]
            if 'output' in ex:
                out = np.array(ex['output'], dtype=np.int64)[np.newaxis, ...]
            else:
                out = None
                
            res = session.run(['output'], {'input': inp})[0]
            
            if out is not None:
                if np.array_equal(res, out):
                    print(f"{split} {i}: ONNX MATCH")
                else:
                    print(f"{split} {i}: ONNX FAIL")
                    print("RES:")
                    for r in res[0]: print(''.join(str(x) for x in r))
                    print("OUT:")
                    for r in out[0]: print(''.join(str(x) for x in r))


# Build the model (the function saves it internally, so we load the result)
create_onnx_model()
import glob
model = onnx.load("/project/repairs/task091.onnx")
