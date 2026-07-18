# Source: predicted/test_onnx_task121.py — ONNX graph construction code
# Verified model: repairs/task121.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    I64 = TensorProto.INT64
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', '3', '3'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('c0', 0)
    make_const('c1', 1)
    make_const('c2', 2)
    make_const('c8', 8)
    make_const('cm1000', -1000)
    
    make_const('axes_0', [0])
    make_const('axes_1', [1])
    make_const('axes_2', [2])
    make_const('axes_1_2', [1, 2])
    make_const('c0_1d', [0])
    make_const('c1_1d', [1])
    make_const('c2_1d', [2])
    make_const('c3_1d', [3])
    
    nodes.append(helper.make_node('Shape', ['input'], ['in_shape']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c0_1d', 'c1_1d', 'axes_0'], ['batch_dim']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c1_1d', 'c2_1d', 'axes_0'], ['H_dim']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c2_1d', 'c3_1d', 'axes_0'], ['W_dim']))
    nodes.append(helper.make_node('Squeeze', ['H_dim', 'axes_0'], ['H']))
    nodes.append(helper.make_node('Squeeze', ['W_dim', 'axes_0'], ['W']))
    
    nodes.append(helper.make_node('Range', ['c0', 'H', 'c1'], ['r_indices']))
    nodes.append(helper.make_node('Range', ['c0', 'W', 'c1'], ['c_indices']))
    
    make_const('shape_H_col', [1, -1, 1])
    make_const('shape_W_row', [1, 1, -1])
    nodes.append(helper.make_node('Reshape', ['r_indices', 'shape_H_col'], ['r_indices_col']))
    nodes.append(helper.make_node('Reshape', ['c_indices', 'shape_W_row'], ['c_indices_row']))
    
    # Find 8
    nodes.append(helper.make_node('Equal', ['input', 'c8'], ['mask8']))
    nodes.append(helper.make_node('Where', ['mask8', 'r_indices_col', 'cm1000'], ['mask8_r']))
    nodes.append(helper.make_node('ReduceMax', ['mask8_r'], ['r8_3d'], axes=[1, 2], keepdims=1)) # [batch, 1, 1]
    
    nodes.append(helper.make_node('Where', ['mask8', 'c_indices_row', 'cm1000'], ['mask8_c']))
    nodes.append(helper.make_node('ReduceMax', ['mask8_c'], ['c8_3d'], axes=[1, 2], keepdims=1)) # [batch, 1, 1]
    
    # Pad
    make_const('pads', [0, 1, 1, 0, 1, 1])
    nodes.append(helper.make_node('Pad', ['input', 'pads'], ['padded']))
    
    # r_grid
    make_const('offset_r', [[[0], [1], [2]]])
    nodes.append(helper.make_node('Add', ['r8_3d', 'offset_r'], ['r_grid']))
    
    nodes.append(helper.make_node('Add', ['W', 'c2'], ['W_plus_2']))
    nodes.append(helper.make_node('Unsqueeze', ['W_plus_2', 'axes_0'], ['W_plus_2_1d']))
    nodes.append(helper.make_node('Concat', ['batch_dim', 'c3_1d', 'W_plus_2_1d'], ['shape_r'], axis=0))
    nodes.append(helper.make_node('Expand', ['r_grid', 'shape_r'], ['r_grid_expanded']))
    
    nodes.append(helper.make_node('GatherElements', ['padded', 'r_grid_expanded'], ['gathered_rows'], axis=1))
    
    # c_grid
    make_const('offset_c', [[[0, 1, 2]]])
    nodes.append(helper.make_node('Add', ['c8_3d', 'offset_c'], ['c_grid']))
    
    nodes.append(helper.make_node('Concat', ['batch_dim', 'c3_1d', 'c3_1d'], ['shape_c'], axis=0))
    nodes.append(helper.make_node('Expand', ['c_grid', 'shape_c'], ['c_grid_expanded']))
    
    nodes.append(helper.make_node('GatherElements', ['gathered_rows', 'c_grid_expanded'], ['region'], axis=2))
    
    # Replace 8
    nodes.append(helper.make_node('Equal', ['region', 'c8'], ['is_8']))
    nodes.append(helper.make_node('Not', ['is_8'], ['not_8']))
    nodes.append(helper.make_node('Where', ['not_8', 'region', 'c0'], ['region_no_8']))
    nodes.append(helper.make_node('ReduceMax', ['region_no_8'], ['C'], axes=[1, 2], keepdims=1))
    
    nodes.append(helper.make_node('Where', ['is_8', 'C', 'region'], ['output']))
    
    graph = helper.make_graph(nodes, 'task121_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task121_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task121.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task121.onnx')
    
    with open('task121.json', 'r') as f:
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
model = onnx.load("/project/repairs/task121.onnx")
