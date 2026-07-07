# Source: predicted/test_onnx_task131.py — ONNX graph construction code
# Verified model: repairs/task131.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    """
    Task 131: Shift '3' shape towards '2' line, place '8' line behind.
    - min/max bounds for '3' and '2'.
    - determine if '2' is horiz or vert.
    - calculate shift dr, dc.
    - calculate r8 or c8.
    - output construction.
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
    make_const('c2', 2)
    make_const('c3', 3)
    make_const('c8_val', 8)
    make_const('cm1', -1)
    make_const('c1000', 1000)
    make_const('axes_0', [0])
    make_const('axes_1', [1])
    make_const('axes_2', [2])
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
    
    # Range vectors
    nodes.append(helper.make_node('Range', ['c0', 'H', 'c1'], ['range_H'])) # [H]
    nodes.append(helper.make_node('Range', ['c0', 'W', 'c1'], ['range_W'])) # [W]
    
    make_const('shape_1H', [1, -1])
    make_const('shape_1W', [1, -1])
    nodes.append(helper.make_node('Reshape', ['range_H', 'shape_1H'], ['range_H_2d'])) # [1, H]
    nodes.append(helper.make_node('Reshape', ['range_W', 'shape_1W'], ['range_W_2d'])) # [1, W]
    
    # Helper to get min/max bounds
    def get_bounds(val, prefix):
        # val is scalar 2 or 3
        make_const(f'{prefix}_val', val)
        nodes.append(helper.make_node('Equal', ['input', f'{prefix}_val'], [f'{prefix}_mask'])) # [batch, H, W]
        nodes.append(helper.make_node('Cast', [f'{prefix}_mask'], [f'{prefix}_mask_i64'], to=int(I64)))
        
        # has_r: [batch, H]
        nodes.append(helper.make_node('ReduceSum', [f'{prefix}_mask_i64', 'axes_2'], [f'{prefix}_sum_r'], keepdims=0))
        nodes.append(helper.make_node('Greater', [f'{prefix}_sum_r', 'c0'], [f'{prefix}_has_r']))
        nodes.append(helper.make_node('Cast', [f'{prefix}_has_r'], [f'{prefix}_has_r_i64'], to=int(I64)))
        
        # has_c: [batch, W]
        nodes.append(helper.make_node('ReduceSum', [f'{prefix}_mask_i64', 'axes_1'], [f'{prefix}_sum_c'], keepdims=0))
        nodes.append(helper.make_node('Greater', [f'{prefix}_sum_c', 'c0'], [f'{prefix}_has_c']))
        nodes.append(helper.make_node('Cast', [f'{prefix}_has_c'], [f'{prefix}_has_c_i64'], to=int(I64)))
        
        # max_r, min_r [batch, 1]
        nodes.append(helper.make_node('Mul', [f'{prefix}_has_r_i64', 'range_H_2d'], [f'{prefix}_r_mul']))
        nodes.append(helper.make_node('ReduceMax', [f'{prefix}_r_mul'], [f'{prefix}_max_r'], axes=[1], keepdims=1))
        nodes.append(helper.make_node('Where', [f'{prefix}_has_r', 'range_H_2d', 'c1000'], [f'{prefix}_r_inv']))
        nodes.append(helper.make_node('ReduceMin', [f'{prefix}_r_inv'], [f'{prefix}_min_r'], axes=[1], keepdims=1))
        
        # max_c, min_c [batch, 1]
        nodes.append(helper.make_node('Mul', [f'{prefix}_has_c_i64', 'range_W_2d'], [f'{prefix}_c_mul']))
        nodes.append(helper.make_node('ReduceMax', [f'{prefix}_c_mul'], [f'{prefix}_max_c'], axes=[1], keepdims=1))
        nodes.append(helper.make_node('Where', [f'{prefix}_has_c', 'range_W_2d', 'c1000'], [f'{prefix}_c_inv']))
        nodes.append(helper.make_node('ReduceMin', [f'{prefix}_c_inv'], [f'{prefix}_min_c'], axes=[1], keepdims=1))
    
    get_bounds(3, 't3')
    get_bounds(2, 't2')
    
    # is_horiz2 = (t2_min_r == t2_max_r)
    nodes.append(helper.make_node('Equal', ['t2_min_r', 't2_max_r'], ['is_horiz2'])) # [batch, 1]
    
    # dr = is_horiz2 ? (t3_max_r < t2_min_r ? t2_min_r - 1 - t3_max_r : t2_min_r + 1 - t3_min_r) : 0
    nodes.append(helper.make_node('Less', ['t3_max_r', 't2_min_r'], ['t3_above']))
    nodes.append(helper.make_node('Sub', ['t2_min_r', 'c1'], ['t2_min_r_m1']))
    nodes.append(helper.make_node('Sub', ['t2_min_r_m1', 't3_max_r'], ['dr_above']))
    nodes.append(helper.make_node('Add', ['t2_min_r', 'c1'], ['t2_min_r_p1']))
    nodes.append(helper.make_node('Sub', ['t2_min_r_p1', 't3_min_r'], ['dr_below']))
    nodes.append(helper.make_node('Where', ['t3_above', 'dr_above', 'dr_below'], ['dr_horiz']))
    nodes.append(helper.make_node('Where', ['is_horiz2', 'dr_horiz', 'c0'], ['dr'])) # [batch, 1]
    
    # dc = !is_horiz2 ? (t3_max_c < t2_min_c ? t2_min_c - 1 - t3_max_c : t2_min_c + 1 - t3_min_c) : 0
    nodes.append(helper.make_node('Not', ['is_horiz2'], ['is_vert2']))
    nodes.append(helper.make_node('Less', ['t3_max_c', 't2_min_c'], ['t3_left']))
    nodes.append(helper.make_node('Sub', ['t2_min_c', 'c1'], ['t2_min_c_m1']))
    nodes.append(helper.make_node('Sub', ['t2_min_c_m1', 't3_max_c'], ['dc_left']))
    nodes.append(helper.make_node('Add', ['t2_min_c', 'c1'], ['t2_min_c_p1']))
    nodes.append(helper.make_node('Sub', ['t2_min_c_p1', 't3_min_c'], ['dc_right']))
    nodes.append(helper.make_node('Where', ['t3_left', 'dc_left', 'dc_right'], ['dc_vert']))
    nodes.append(helper.make_node('Where', ['is_vert2', 'dc_vert', 'c0'], ['dc'])) # [batch, 1]
    
    # r8 = is_horiz2 ? (t3_max_r < t2_min_r ? t3_min_r + dr - 1 : t3_max_r + dr + 1) : -1
    nodes.append(helper.make_node('Add', ['t3_min_r', 'dr'], ['t3_min_r_dr']))
    nodes.append(helper.make_node('Sub', ['t3_min_r_dr', 'c1'], ['r8_above']))
    nodes.append(helper.make_node('Add', ['t3_max_r', 'dr'], ['t3_max_r_dr']))
    nodes.append(helper.make_node('Add', ['t3_max_r_dr', 'c1'], ['r8_below']))
    nodes.append(helper.make_node('Where', ['t3_above', 'r8_above', 'r8_below'], ['r8_horiz']))
    nodes.append(helper.make_node('Where', ['is_horiz2', 'r8_horiz', 'cm1'], ['r8'])) # [batch, 1]
    
    # c8 = is_vert2 ? (t3_max_c < t2_min_c ? t3_min_c + dc - 1 : t3_max_c + dc + 1) : -1
    nodes.append(helper.make_node('Add', ['t3_min_c', 'dc'], ['t3_min_c_dc']))
    nodes.append(helper.make_node('Sub', ['t3_min_c_dc', 'c1'], ['c8_left']))
    nodes.append(helper.make_node('Add', ['t3_max_c', 'dc'], ['t3_max_c_dc']))
    nodes.append(helper.make_node('Add', ['t3_max_c_dc', 'c1'], ['c8_right']))
    nodes.append(helper.make_node('Where', ['t3_left', 'c8_left', 'c8_right'], ['c8_vert']))
    nodes.append(helper.make_node('Where', ['is_vert2', 'c8_vert', 'cm1'], ['c8'])) # [batch, 1]
    
    # Grid of r, c: [batch, H, W]
    make_const('shape_H_1', [1, -1, 1])
    make_const('shape_1_W', [1, 1, -1])
    nodes.append(helper.make_node('Reshape', ['range_H', 'shape_H_1'], ['grid_r']))
    nodes.append(helper.make_node('Reshape', ['range_W', 'shape_1_W'], ['grid_c']))
    
    # target '8' locations
    # Reshape r8 to [batch, 1, 1], c8 to [batch, 1, 1]
    make_const('shape_b_1_1', [-1, 1, 1])
    nodes.append(helper.make_node('Reshape', ['r8', 'shape_b_1_1'], ['r8_3d']))
    nodes.append(helper.make_node('Reshape', ['c8', 'shape_b_1_1'], ['c8_3d']))
    nodes.append(helper.make_node('Equal', ['grid_r', 'r8_3d'], ['is_8_r']))
    nodes.append(helper.make_node('Equal', ['grid_c', 'c8_3d'], ['is_8_c']))
    nodes.append(helper.make_node('Or', ['is_8_r', 'is_8_c'], ['is_8'])) # [batch, H, W]
    
    # source for '3': src_r = r - dr, src_c = c - dc
    # dr, dc are [batch, 1], reshape to [batch, 1, 1]
    nodes.append(helper.make_node('Reshape', ['dr', 'shape_b_1_1'], ['dr_3d']))
    nodes.append(helper.make_node('Reshape', ['dc', 'shape_b_1_1'], ['dc_3d']))
    nodes.append(helper.make_node('Sub', ['grid_r', 'dr_3d'], ['src_r'])) # [batch, H, W]
    nodes.append(helper.make_node('Sub', ['grid_c', 'dc_3d'], ['src_c'])) # [batch, H, W]
    
    # validity
    nodes.append(helper.make_node('GreaterOrEqual', ['src_r', 'c0'], ['src_r_ge_0']))
    nodes.append(helper.make_node('Less', ['src_r', 'H'], ['src_r_lt_H']))
    nodes.append(helper.make_node('GreaterOrEqual', ['src_c', 'c0'], ['src_c_ge_0']))
    nodes.append(helper.make_node('Less', ['src_c', 'W'], ['src_c_lt_W']))
    nodes.append(helper.make_node('And', ['src_r_ge_0', 'src_r_lt_H'], ['valid_r']))
    nodes.append(helper.make_node('And', ['src_c_ge_0', 'src_c_lt_W'], ['valid_c']))
    nodes.append(helper.make_node('And', ['valid_r', 'valid_c'], ['valid_src']))
    
    # Gather '3's from input
    # flat_idx = src_r * W + src_c
    nodes.append(helper.make_node('Mul', ['src_r', 'W'], ['src_r_W']))
    nodes.append(helper.make_node('Add', ['src_r_W', 'src_c'], ['flat_idx'])) # [batch, H, W]
    
    # Clamp flat_idx to 0 to avoid gather errors on invalid spots
    nodes.append(helper.make_node('Where', ['valid_src', 'flat_idx', 'c0'], ['safe_flat_idx']))
    
    # Flatten input to [batch, H*W]
    nodes.append(helper.make_node('Mul', ['H', 'W'], ['HW']))
    nodes.append(helper.make_node('Unsqueeze', ['HW', 'axes_0'], ['HW_1d']))
    nodes.append(helper.make_node('Concat', ['batch_dim', 'HW_1d'], ['flat_shape'], axis=0))
    nodes.append(helper.make_node('Reshape', ['input', 'flat_shape'], ['flat_input']))
    
    # Gather flat_idx from flat_input
    # GatherElements needs indices of same rank as data.
    # safe_flat_idx is [batch, H, W], we want to gather from [batch, H*W].
    # So reshape safe_flat_idx to [batch, H*W], gather, reshape back to [batch, H, W]
    nodes.append(helper.make_node('Reshape', ['safe_flat_idx', 'flat_shape'], ['safe_flat_idx_2d']))
    nodes.append(helper.make_node('GatherElements', ['flat_input', 'safe_flat_idx_2d'], ['gathered_2d'], axis=1))
    nodes.append(helper.make_node('Reshape', ['gathered_2d', 'in_shape'], ['gathered']))
    
    # Check if gathered is 3
    nodes.append(helper.make_node('Equal', ['gathered', 'c3'], ['gathered_is_3']))
    nodes.append(helper.make_node('And', ['valid_src', 'gathered_is_3'], ['is_shifted_3']))
    
    # Combine:
    # out = Where(is_8, 8, Where(is_shifted_3, 3, Where(input == 2, 2, 0)))
    nodes.append(helper.make_node('Equal', ['input', 'c2'], ['is_2']))
    nodes.append(helper.make_node('Where', ['is_2', 'c2', 'c0'], ['out_2']))
    nodes.append(helper.make_node('Where', ['is_shifted_3', 'c3', 'out_2'], ['out_3']))
    nodes.append(helper.make_node('Where', ['is_8', 'c8_val', 'out_3'], ['output']))
    
    graph = helper.make_graph(nodes, 'task131_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task131_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task131.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task131.onnx')
    
    with open('task131.json', 'r') as f:
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
model = onnx.load("/project/repairs/task131.onnx")
