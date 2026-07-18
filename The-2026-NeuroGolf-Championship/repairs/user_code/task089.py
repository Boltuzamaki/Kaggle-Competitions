# Source: predicted/test_onnx_task089.py — ONNX graph construction code
# Verified model: repairs/task089.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    I64 = TensorProto.INT64
    F = TensorProto.FLOAT
    BOOL = TensorProto.BOOL
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H', 'W'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('c0', 0)
    make_const('c0_f', 0.0, F)
    make_const('c1', 1)
    make_const('c1_1d', [1])
    make_const('c2_1d', [2])
    make_const('cm1_1d', [-1])
    make_const('axes_0', [0])
    make_const('axes_1', [1])
    make_const('axes_01', [0, 1])
    
    # H and W
    nodes.append(helper.make_node('Shape', ['input'], ['shape']))
    nodes.append(helper.make_node('Slice', ['shape', 'c1_1d', 'c2_1d', 'axes_0'], ['H_1d']))
    make_const('c3_1d', [3])
    nodes.append(helper.make_node('Slice', ['shape', 'c2_1d', 'c3_1d', 'axes_0'], ['W_1d']))
    nodes.append(helper.make_node('Squeeze', ['H_1d', 'axes_0'], ['H_s']))
    nodes.append(helper.make_node('Squeeze', ['W_1d', 'axes_0'], ['W_s']))
    
    # mask_all
    nodes.append(helper.make_node('Squeeze', ['input', 'axes_0'], ['input_2d']))
    nodes.append(helper.make_node('Greater', ['input_2d', 'c0'], ['mask_all']))
    nodes.append(helper.make_node('Cast', ['mask_all'], ['mask_all_f'], to=F))
    
    # Shapes
    nodes.append(helper.make_node('Concat', ['c1_1d', 'c1_1d', 'H_1d', 'W_1d'], ['shape_11HW'], axis=0))
    nodes.append(helper.make_node('Concat', ['H_1d', 'W_1d'], ['shape_HW'], axis=0))
    nodes.append(helper.make_node('Concat', ['H_1d', 'c1_1d'], ['shape_H1'], axis=0))
    nodes.append(helper.make_node('Concat', ['c1_1d', 'W_1d'], ['shape_1W'], axis=0))
    
    # Conv to count neighbors
    nodes.append(helper.make_node('Reshape', ['mask_all_f', 'shape_11HW'], ['input_f_4d']))
    make_const('conv_weight', np.ones((1, 1, 3, 3), dtype=np.float32), F)
    nodes.append(helper.make_node('Conv', ['input_f_4d', 'conv_weight'], ['neighbors_4d'], pads=[1,1,1,1]))
    nodes.append(helper.make_node('Reshape', ['neighbors_4d', 'shape_HW'], ['neighbors_f']))
    
    make_const('f1_5', 1.5, F)
    nodes.append(helper.make_node('Less', ['neighbors_f', 'f1_5'], ['is_isolated']))
    nodes.append(helper.make_node('And', ['mask_all', 'is_isolated'], ['isolated']))
    nodes.append(helper.make_node('Not', ['isolated'], ['not_isolated']))
    nodes.append(helper.make_node('And', ['mask_all', 'not_isolated'], ['non_isolated']))
    
    # Y and X grids
    nodes.append(helper.make_node('Range', ['c0', 'H_s', 'c1'], ['Y_1d']))
    nodes.append(helper.make_node('Range', ['c0', 'W_s', 'c1'], ['X_1d']))
    nodes.append(helper.make_node('Reshape', ['Y_1d', 'shape_H1'], ['Y_2d']))
    nodes.append(helper.make_node('Reshape', ['X_1d', 'shape_1W'], ['X_2d']))
    
    make_const('shape_1111', [1, 1, 1, 1])
    nodes.append(helper.make_node('Concat', ['H_1d', 'c1_1d', 'c1_1d', 'c1_1d'], ['shape_Y_out'], axis=0))
    nodes.append(helper.make_node('Concat', ['c1_1d', 'W_1d', 'c1_1d', 'c1_1d'], ['shape_X_out'], axis=0))
    nodes.append(helper.make_node('Concat', ['c1_1d', 'c1_1d', 'H_1d', 'c1_1d'], ['shape_Y_tgt'], axis=0))
    nodes.append(helper.make_node('Concat', ['c1_1d', 'c1_1d', 'c1_1d', 'W_1d'], ['shape_X_tgt'], axis=0))
    
    nodes.append(helper.make_node('Reshape', ['Y_1d', 'shape_Y_out'], ['Y_out']))
    nodes.append(helper.make_node('Reshape', ['X_1d', 'shape_X_out'], ['X_out']))
    nodes.append(helper.make_node('Reshape', ['Y_1d', 'shape_Y_tgt'], ['Y_tgt']))
    nodes.append(helper.make_node('Reshape', ['X_1d', 'shape_X_tgt'], ['X_tgt']))
    
    nodes.append(helper.make_node('Sub', ['Y_out', 'Y_tgt'], ['Y_diff']))
    nodes.append(helper.make_node('Reshape', ['H_1d', 'shape_1111'], ['H_1111']))
    nodes.append(helper.make_node('Add', ['H_1111', 'Y_diff'], ['Y_idx']))
    
    nodes.append(helper.make_node('Sub', ['X_out', 'X_tgt'], ['X_diff']))
    nodes.append(helper.make_node('Reshape', ['W_1d', 'shape_1111'], ['W_1111']))
    
    nodes.append(helper.make_node('Add', ['H_1d', 'H_1d'], ['H2_1d']))
    nodes.append(helper.make_node('Add', ['W_1d', 'W_1d'], ['W2_1d']))
    nodes.append(helper.make_node('Reshape', ['W2_1d', 'shape_1111'], ['W2_1111']))
    
    color_outs = []
    
    for c in range(1, 10):
        make_const(f'color_{c}', c)
        nodes.append(helper.make_node('Equal', ['input_2d', f'color_{c}'], [f'is_{c}']))
        
        nodes.append(helper.make_node('And', [f'is_{c}', 'non_isolated'], [f'source_bool_{c}']))
        nodes.append(helper.make_node('And', [f'is_{c}', 'isolated'], [f'target_{c}']))
        
        mask_name = f'source_f_{c}'
        nodes.append(helper.make_node('Cast', [f'source_bool_{c}'], [mask_name], to=F))
        
        for i in range(15):
            nodes.append(helper.make_node('Reshape', [mask_name, 'shape_11HW'], [f'mask_4d_{c}_{i}']))
            nodes.append(helper.make_node('MaxPool', [f'mask_4d_{c}_{i}'], [f'pool_{c}_{i}'], kernel_shape=[3,3], pads=[1,1,1,1]))
            nodes.append(helper.make_node('Reshape', [f'pool_{c}_{i}', 'shape_HW'], [f'pool_2d_{c}_{i}']))
            nodes.append(helper.make_node('Greater', [f'pool_2d_{c}_{i}', 'c0_f'], [f'pool_bool_{c}_{i}']))
            nodes.append(helper.make_node('And', [f'pool_bool_{c}_{i}', 'mask_all'], [f'mask_bool_{c}_{i+1}']))
            mask_name = f'mask_f_{c}_{i+1}'
            nodes.append(helper.make_node('Cast', [f'mask_bool_{c}_{i+1}'], [mask_name], to=F))
            
        nodes.append(helper.make_node('Cast', [mask_name], [f'T_mask_{c}'], to=I64))
        nodes.append(helper.make_node('Mul', ['input_2d', f'T_mask_{c}'], [f'T_{c}']))
        
        nodes.append(helper.make_node('Cast', [f'source_bool_{c}'], [f'source_i64_{c}'], to=I64))
        nodes.append(helper.make_node('Reshape', [f'source_i64_{c}', 'cm1_1d'], [f'src_flat_{c}']))
        nodes.append(helper.make_node('ArgMax', [f'src_flat_{c}'], [f'idx_first_{c}'], axis=0, keepdims=1))
        nodes.append(helper.make_node('Div', [f'idx_first_{c}', 'W_1d'], [f'y_s_{c}']))
        nodes.append(helper.make_node('Mod', [f'idx_first_{c}', 'W_1d'], [f'x_s_{c}']))
        
        nodes.append(helper.make_node('Concat', ['H_1d', 'W_1d', 'H_1d', 'W_1d'], [f'pads_T_{c}'], axis=0))
        nodes.append(helper.make_node('Pad', [f'T_{c}', f'pads_T_{c}'], [f'T_pad_{c}']))
        
        nodes.append(helper.make_node('Concat', [f'y_s_{c}', f'x_s_{c}'], [f'starts_{c}'], axis=0))
        nodes.append(helper.make_node('Add', [f'y_s_{c}', 'H2_1d'], [f'end_y_{c}']))
        nodes.append(helper.make_node('Add', [f'x_s_{c}', 'W2_1d'], [f'end_x_{c}']))
        nodes.append(helper.make_node('Concat', [f'end_y_{c}', f'end_x_{c}'], [f'ends_{c}'], axis=0))
        
        nodes.append(helper.make_node('Slice', [f'T_pad_{c}', f'starts_{c}', f'ends_{c}', 'axes_01'], [f'T_cent_{c}']))
        nodes.append(helper.make_node('Reshape', [f'T_cent_{c}', 'cm1_1d'], [f'T_flat_{c}']))
        
        if c % 2 == 0:
            nodes.append(helper.make_node('Sub', ['W_1111', 'X_diff'], [f'X_idx_{c}']))
        else:
            nodes.append(helper.make_node('Add', ['W_1111', 'X_diff'], [f'X_idx_{c}']))
            
        nodes.append(helper.make_node('Mul', ['Y_idx', 'W2_1111'], [f'idx_part1_{c}']))
        nodes.append(helper.make_node('Add', [f'idx_part1_{c}', f'X_idx_{c}'], [f'idx_1d_{c}']))
        
        nodes.append(helper.make_node('Gather', [f'T_flat_{c}', f'idx_1d_{c}'], [f'gathered_{c}']))
        
        nodes.append(helper.make_node('Cast', [f'target_{c}'], [f'target_i64_{c}'], to=I64))
        nodes.append(helper.make_node('Reshape', [f'target_i64_{c}', 'shape_11HW'], [f'target_4d_{c}']))
        nodes.append(helper.make_node('Mul', [f'gathered_{c}', f'target_4d_{c}'], [f'gathered_mask_{c}']))
        nodes.append(helper.make_node('ReduceMax', [f'gathered_mask_{c}'], [f'out_c_{c}'], axes=[2, 3], keepdims=0))
        color_outs.append(f'out_c_{c}')

    nodes.append(helper.make_node('Max', color_outs + ['input_2d'], ['output_2d']))
    nodes.append(helper.make_node('Reshape', ['output_2d', 'shape'], ['output']))
    
    graph = helper.make_graph(nodes, 'task089_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task089_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task089.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task089.onnx')
    
    with open('task089.json', 'r') as f:
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
model = onnx.load("/project/repairs/task089.onnx")
