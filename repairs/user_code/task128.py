# Source: predicted/test_onnx_task128.py — ONNX graph construction code
# Verified model: repairs/task128.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    """
    Task 128: Shift each column up by the number of non-zero elements in that column.
    H_c = count_nonzero(input[:, c])
    output[r, c] = input[r + H_c, c] if r + H_c < H else 0
    Strategy: 
    - Compute H_c = ReduceSum(input != 0, axis=1) [Wait, axis=1 of input[batch, H, W] is rows, so axis=1]
    - Pad input at bottom with H zeros -> shape [batch, 2H, W]
    - Compute source indices: r + H_c
    - GatherElements along axis=1
    """
    I64 = TensorProto.INT64
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H', 'W'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('c0', 0)
    make_const('c1', 1)
    make_const('axes_0', [0])
    make_const('axes_1', [1])
    make_const('c0_1d', [0])
    make_const('c1_1d', [1])
    make_const('c2_1d', [2])
    make_const('c3_1d', [3])
    
    # Get H
    nodes.append(helper.make_node('Shape', ['input'], ['in_shape']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c1_1d', 'c2_1d', 'axes_0'], ['H_dim']))
    nodes.append(helper.make_node('Squeeze', ['H_dim', 'axes_0'], ['H']))
    
    # H_c = sum(input != 0) along axis 1
    nodes.append(helper.make_node('Equal', ['input', 'c0'], ['is_zero']))
    nodes.append(helper.make_node('Not', ['is_zero'], ['not_zero']))
    nodes.append(helper.make_node('Cast', ['not_zero'], ['not_zero_i64'], to=int(I64)))
    nodes.append(helper.make_node('ReduceSum', ['not_zero_i64', 'axes_1'], ['H_c'], keepdims=1)) # [batch, 1, W]
    
    # Pad input with H zeros at the bottom.
    # pads format for 3D: [pad_b_0, pad_b_1, pad_b_2, pad_e_0, pad_e_1, pad_e_2]
    # We want pad_e_1 = H, others 0.
    # Concat [0, 0, 0, 0, H, 0]
    nodes.append(helper.make_node('Concat', ['c0_1d', 'c0_1d', 'c0_1d', 'c0_1d', 'H_dim', 'c0_1d'], ['pads'], axis=0))
    nodes.append(helper.make_node('Pad', ['input', 'pads'], ['padded_input'])) # [batch, 2H, W]
    
    # Create grid of r: [batch, H, W]
    nodes.append(helper.make_node('Range', ['c0', 'H', 'c1'], ['r_range'])) # [H]
    make_const('shape_r', [1, -1, 1])
    nodes.append(helper.make_node('Reshape', ['r_range', 'shape_r'], ['r_col'])) # [1, H, 1]
    
    # We need to expand r_col and H_c to [batch, H, W], then add them.
    # Actually, Add broadcasts automatically!
    # r_col is [1, H, 1], H_c is [batch, 1, W].
    # Add(r_col, H_c) -> [batch, H, W]
    nodes.append(helper.make_node('Add', ['r_col', 'H_c'], ['gather_indices']))
    
    # GatherElements
    nodes.append(helper.make_node('GatherElements', ['padded_input', 'gather_indices'], ['output'], axis=1))
    
    graph = helper.make_graph(nodes, 'task128_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task128_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task128.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task128.onnx')
    
    with open('task128.json', 'r') as f:
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
                    print(str(split)+' '+str(i)+': ONNX MATCH')
                else:
                    print(str(split)+' '+str(i)+': ONNX FAIL')
                    print("RES:")
                    for r in res[0]: print(''.join(str(x) for x in r))
                    print("OUT:")
                    for r in out[0]: print(''.join(str(x) for x in r))


# Build the model (the function saves it internally, so we load the result)
create_onnx_model()
import glob
model = onnx.load("/project/repairs/task128.onnx")
