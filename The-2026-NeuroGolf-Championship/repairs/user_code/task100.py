# Source: predicted/test_onnx_task100.py — ONNX graph construction code
# Verified model: repairs/task100.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    I64 = TensorProto.INT64
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 2, 2])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('c0_1d', [0])
    make_const('c1_1d', [1])
    make_const('c2_1d', [2])
    make_const('c3_1d', [3])
    make_const('cm1', [-1])
    make_const('c999', [999])
    
    make_const('axes_0', [0])
    make_const('axes_12', [1, 2])
    make_const('axes_1_1d', [1])
    
    make_const('shape_y', [1, -1, 1])
    make_const('shape_x', [1, 1, -1])
    make_const('shape_b_1_1', [-1, 1, 1])
    
    nodes.append(helper.make_node('Shape', ['input'], ['shape']))
    nodes.append(helper.make_node('Slice', ['shape', 'c0_1d', 'c1_1d', 'axes_0'], ['batch_dim']))
    nodes.append(helper.make_node('Slice', ['shape', 'c1_1d', 'c2_1d', 'axes_0'], ['H_1d']))
    nodes.append(helper.make_node('Slice', ['shape', 'c2_1d', 'c3_1d', 'axes_0'], ['W_1d']))
    
    nodes.append(helper.make_node('Range', ['c0_1d', 'H_1d', 'c1_1d'], ['y_range']))
    nodes.append(helper.make_node('Reshape', ['y_range', 'shape_y'], ['y_col']))
    nodes.append(helper.make_node('Expand', ['y_col', 'shape'], ['y_grid']))
    
    nodes.append(helper.make_node('Range', ['c0_1d', 'W_1d', 'c1_1d'], ['x_range']))
    nodes.append(helper.make_node('Reshape', ['x_range', 'shape_x'], ['x_row']))
    nodes.append(helper.make_node('Expand', ['x_row', 'shape'], ['x_grid']))
    
    areas = []
    for c in range(1, 10):
        make_const(f'cc_{c}', [c])
        nodes.append(helper.make_node('Equal', ['input', f'cc_{c}'], [f'mask_{c}']))
        nodes.append(helper.make_node('Cast', [f'mask_{c}'], [f'mask_{c}_i64'], to=I64))
        
        nodes.append(helper.make_node('ReduceMax', [f'mask_{c}_i64'], [f'has_{c}_dim'], axes=[1, 2], keepdims=0))
        
        nodes.append(helper.make_node('Where', [f'mask_{c}', 'y_grid', 'cm1'], [f'y_max_grid_{c}']))
        nodes.append(helper.make_node('Where', [f'mask_{c}', 'y_grid', 'c999'], [f'y_min_grid_{c}']))
        nodes.append(helper.make_node('Where', [f'mask_{c}', 'x_grid', 'cm1'], [f'x_max_grid_{c}']))
        nodes.append(helper.make_node('Where', [f'mask_{c}', 'x_grid', 'c999'], [f'x_min_grid_{c}']))
        
        nodes.append(helper.make_node('ReduceMax', [f'y_max_grid_{c}'], [f'y_max_{c}'], axes=[1, 2], keepdims=0))
        nodes.append(helper.make_node('ReduceMin', [f'y_min_grid_{c}'], [f'y_min_{c}'], axes=[1, 2], keepdims=0))
        nodes.append(helper.make_node('ReduceMax', [f'x_max_grid_{c}'], [f'x_max_{c}'], axes=[1, 2], keepdims=0))
        nodes.append(helper.make_node('ReduceMin', [f'x_min_grid_{c}'], [f'x_min_{c}'], axes=[1, 2], keepdims=0))
        
        nodes.append(helper.make_node('Sub', [f'y_max_{c}', f'y_min_{c}'], [f'h_m1_{c}']))
        nodes.append(helper.make_node('Add', [f'h_m1_{c}', 'c1_1d'], [f'h_{c}']))
        
        nodes.append(helper.make_node('Sub', [f'x_max_{c}', f'x_min_{c}'], [f'w_m1_{c}']))
        nodes.append(helper.make_node('Add', [f'w_m1_{c}', 'c1_1d'], [f'w_{c}']))
        
        nodes.append(helper.make_node('Mul', [f'h_{c}', f'w_{c}'], [f'area_{c}_raw']))
        
        nodes.append(helper.make_node('Greater', [f'has_{c}_dim', 'c0_1d'], [f'has_{c}_bool']))
        nodes.append(helper.make_node('Where', [f'has_{c}_bool', f'area_{c}_raw', 'c0_1d'], [f'area_{c}']))
        
        nodes.append(helper.make_node('Unsqueeze', [f'area_{c}', 'axes_1_1d'], [f'area_{c}_unsq']))
        areas.append(f'area_{c}_unsq')
        
    nodes.append(helper.make_node('Concat', areas, ['all_areas'], axis=1))
    nodes.append(helper.make_node('ArgMax', ['all_areas'], ['best_idx'], axis=1, keepdims=0))
    nodes.append(helper.make_node('Add', ['best_idx', 'c1_1d'], ['best_color']))
    
    nodes.append(helper.make_node('Reshape', ['best_color', 'shape_b_1_1'], ['best_color_reshaped']))
    nodes.append(helper.make_node('Concat', ['batch_dim', 'c2_1d', 'c2_1d'], ['out_shape'], axis=0))
    nodes.append(helper.make_node('Expand', ['best_color_reshaped', 'out_shape'], ['output']))
    
    graph = helper.make_graph(nodes, 'task100_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task100_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task100.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task100.onnx')
    
    with open('task100.json', 'r') as f:
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
model = onnx.load("/project/repairs/task100.onnx")
