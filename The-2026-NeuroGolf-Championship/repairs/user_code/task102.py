# Source: predicted/test_onnx_task102.py — ONNX graph construction code
# Verified model: repairs/task102.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    I64 = TensorProto.INT64
    F32 = TensorProto.FLOAT
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H', 'W'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
        
    def make_const_scalar(name, val, dtype=I64):
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, [], [val])))
    
    make_const('c0_1d', [0])
    make_const('c1_1d', [1])
    make_const('c2_1d', [2])
    make_const('c3_1d', [3])
    make_const('c5_1d', [5])
    make_const('c1000_f', [1000.0], dtype=F32)
    make_const_scalar('c0_f_scalar', 0.0, dtype=F32)
    
    make_const('axes_0', [0])
    make_const('axes_2_1d', [2])
    make_const('axes_3_1d', [3])
    
    nodes.append(helper.make_node('Shape', ['input'], ['shape']))
    nodes.append(helper.make_node('Slice', ['shape', 'c0_1d', 'c1_1d', 'axes_0'], ['batch_dim']))
    nodes.append(helper.make_node('Slice', ['shape', 'c1_1d', 'c2_1d', 'axes_0'], ['H_1d']))
    nodes.append(helper.make_node('Slice', ['shape', 'c2_1d', 'c3_1d', 'axes_0'], ['W_1d']))
    nodes.append(helper.make_node('Concat', ['batch_dim', 'c1_1d', 'H_1d', 'W_1d'], ['shape_b_1_H_W'], axis=0))
    
    nodes.append(helper.make_node('Equal', ['input', 'c0_1d'], ['is_0']))
    nodes.append(helper.make_node('Cast', ['is_0'], ['is_0_f'], to=F32))
    
    nodes.append(helper.make_node('Equal', ['input', 'c5_1d'], ['is_5']))
    nodes.append(helper.make_node('Cast', ['is_5'], ['is_5_f'], to=F32))
    
    nodes.append(helper.make_node('Mul', ['is_5_f', 'c1000_f'], ['is_5_f_1000']))
    nodes.append(helper.make_node('Add', ['is_0_f', 'is_5_f_1000'], ['combined_f']))
    nodes.append(helper.make_node('Reshape', ['combined_f', 'shape_b_1_H_W'], ['combined_4d']))
    
    make_const('pads_30', [0, 0, 0, 0, 0, 0, 30, 30], dtype=I64)
    nodes.append(helper.make_node('Pad', ['combined_4d', 'pads_30', 'c0_f_scalar'], ['combined_padded']))
    
    mask_names = []
    for K in range(1, 29):
        K_size = K + 2
        kernel = np.full((1, 1, K_size, K_size), 1.0, dtype=np.float32)
        kernel[0, 0, 1:-1, 1:-1] = 1000000.0
        make_const(f'kernel_{K}', kernel, dtype=F32)
        
        expected = float((4 * K + 4) * 1000.0 + K * K * 1000000.0)
        make_const_scalar(f'expected_{K}', expected, dtype=F32)
        
        nodes.append(helper.make_node('Conv', ['combined_padded', f'kernel_{K}'], [f'conv_{K}'], pads=[0,0,0,0]))
        
        nodes.append(helper.make_node('Sub', [f'conv_{K}', f'expected_{K}'], [f'diff_{K}']))
        nodes.append(helper.make_node('Abs', [f'diff_{K}'], [f'abs_diff_{K}']))
        
        make_const_scalar(f'c0_5_scalar_{K}', 0.5, dtype=F32)
        nodes.append(helper.make_node('Less', [f'abs_diff_{K}', f'c0_5_scalar_{K}'], [f'match_{K}']))
        nodes.append(helper.make_node('Cast', [f'match_{K}'], [f'match_f_{K}'], to=F32))
        
        pads_match = [0, 0, K, K, 0, 0, K, K]
        make_const(f'pads_match_{K}', pads_match, dtype=I64)
        nodes.append(helper.make_node('Pad', [f'match_f_{K}', f'pads_match_{K}', 'c0_f_scalar'], [f'match_padded_{K}']))
        
        kernel_expand = np.ones((1, 1, K, K), dtype=np.float32)
        make_const(f'kernel_expand_{K}', kernel_expand, dtype=F32)
        nodes.append(helper.make_node('Conv', [f'match_padded_{K}', f'kernel_expand_{K}'], [f'hole_mask_{K}'], pads=[0, 0, 0, 0]))
        
        mask_names.append(f'hole_mask_{K}')
        
    curr = mask_names[0]
    for i in range(1, len(mask_names)):
        nodes.append(helper.make_node('Add', [curr, mask_names[i]], [f'sum_{i}']))
        curr = f'sum_{i}'
        
    nodes.append(helper.make_node('Greater', [curr, 'c0_f_scalar'], ['final_hole_mask_padded']))
    
    nodes.append(helper.make_node('Slice', ['final_hole_mask_padded', 'c0_1d', 'H_1d', 'axes_2_1d'], ['sliced_H']))
    nodes.append(helper.make_node('Slice', ['sliced_H', 'c0_1d', 'W_1d', 'axes_3_1d'], ['sliced_HW']))
    nodes.append(helper.make_node('Reshape', ['sliced_HW', 'shape'], ['final_mask']))
    
    nodes.append(helper.make_node('Where', ['final_mask', 'c2_1d', 'input'], ['output']))
    
    graph = helper.make_graph(nodes, 'task102_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task102_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task102.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task102.onnx')
    
    with open('task102.json', 'r') as f:
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
model = onnx.load("/project/repairs/task102.onnx")
